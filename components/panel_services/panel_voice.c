/* Standalone voice using the Agent Key delivered by this panel's login. */
#include "panel_voice.h"
#include "panel_voice_audio.h"
#include "esp_aec.h"
#include "esp_heap_caps.h"
#include "product_support.h"
#include "nyx_creds.h"
#include "lcd_renderer.h"
#include "settings.h"
#include "network.h"
#include "cJSON.h"
#include "nvs.h"
#include "esp_crt_bundle.h"
#include "esp_http_client.h"
#include "esp_websocket_client.h"
#include "esp_peer.h"
#include "esp_peer_default.h"
#include "media_lib_adapter.h"
#include "esp_opus_enc.h"
#include "esp_opus_dec.h"
#include "esp_timer.h"
#include "esp_random.h"
#include "esp_app_desc.h"
#include "mbedtls/sha256.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/event_groups.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <math.h>

#define FRAME_SAMPLES 320
#define MAX_PCM_SAMPLES 1920
#define MAX_CONTROL_BYTES (96 * 1024)
// esp_audio_codec documents ~40 KiB for encoders and ~20 KiB for decoders.
// Include the PCM/RTP buffers and caller frames in each task's budget.
#define MIC_STACK_BYTES (48 * 1024)
#define SPEAKER_STACK_BYTES (24 * 1024)
#define DONE_PEER 1
#define DONE_MIC 2
#define DONE_SPEAKER 4
typedef struct { char key_id[40]; panel_voice_bot_t bot; } voice_config_t;
typedef struct { unsigned len; uint8_t bytes[1500]; } audio_packet_t;
typedef struct { char session[48]; panel_voice_remote_t request; } remote_message_t;
typedef struct { char id[40]; unsigned char digest[32]; } remote_record_t;
typedef struct {
    voice_config_t config;
    panel_voice_input_mode_t input_mode;
    panel_voice_remote_t remote;
    nyx_agent_credential_t credential;
    esp_peer_handle_t peer;
    esp_websocket_client_handle_t socket;
    SemaphoreHandle_t peer_lock, reference_lock;
    panel_voice_reference_t reference;
    QueueHandle_t audio, offers, messages;
    EventGroupHandle_t done;
    atomic_bool running, connected, controls, provider_ready, muted;
    atomic_uint played_samples;
    atomic_int_fast64_t last_audio_us, last_speech_us;
    char session[48];
    unsigned generation, control_revision, heartbeat_revision, provider_frames;
    char *socket_text;
    size_t socket_used, socket_size;
    panel_voice_captions_t last_captions;
} voice_call_t;

static SemaphoreHandle_t state_lock;
static voice_config_t configuration;
static panel_voice_snapshot_t status;
static panel_voice_bot_t bots[PANEL_VOICE_MAX_BOTS];
static size_t bot_count;
static char bots_key_id[40];
static atomic_bool busy, stopping, holding, microphone_muted;
static atomic_uint playback_gain=4;
static atomic_uint input_mode=PANEL_VOICE_INPUT_HOLD;
static atomic_uint active_mode=PANEL_VOICE_INPUT_HOLD;
static QueueHandle_t remote_messages;
static SemaphoreHandle_t remote_lock;
// Bounded, persisted retry protection. Records are written before any side effect.
static remote_record_t remote_records[8];

static void state(const char *name, const char *message)
{
    xSemaphoreTake(state_lock, portMAX_DELAY);
    snprintf(status.state,sizeof(status.state),"%s",name);
    snprintf(status.message,sizeof(status.message),"%s",message);
    xSemaphoreGive(state_lock);
}
void panel_voice_snapshot(panel_voice_snapshot_t *out)
{
    if (!out) return;
    memset(out,0,sizeof(*out));
    if (!state_lock) return;
    xSemaphoreTake(state_lock,portMAX_DELAY);
    *out=status;
    out->configured=configuration.bot.agent_id[0]!=0;
    xSemaphoreGive(state_lock);
    out->configured=out->configured && nyx_creds_have_agent();
    out->busy=atomic_load(&busy);
    out->holding=atomic_load(&holding);
    out->playback_gain=atomic_load(&playback_gain);
    out->input_mode=(panel_voice_input_mode_t)atomic_load(&active_mode);
    out->preferred_input_mode=(panel_voice_input_mode_t)atomic_load(&input_mode);
    out->microphone_muted=atomic_load(&microphone_muted);
}
static bool valid_config(const voice_config_t *cfg)
{
    return memchr(cfg->key_id,0,sizeof(cfg->key_id)) &&
        memchr(cfg->bot.agent_id,0,sizeof(cfg->bot.agent_id)) &&
        memchr(cfg->bot.conversation_id,0,sizeof(cfg->bot.conversation_id)) &&
        memchr(cfg->bot.name,0,sizeof(cfg->bot.name)) &&
        panel_voice_uuid(cfg->key_id) && panel_voice_uuid(cfg->bot.agent_id) &&
        panel_voice_conversation(cfg->bot.conversation_id);
}
void panel_voice_init(void)
{
    if (state_lock) return;
    state_lock=xSemaphoreCreateMutex();
    if (!state_lock) return;
    remote_messages=xQueueCreate(4,sizeof(remote_message_t));
    remote_lock=xSemaphoreCreateMutex();
    media_lib_add_default_adapter();
    nvs_handle_t h;
    if (nvs_open(NYX_VOICE_NVS_NAMESPACE,NVS_READONLY,&h)==ESP_OK) {
        size_t len=sizeof(configuration);
        if (nvs_get_blob(h,"selection_v1",&configuration,&len)!=ESP_OK || len!=sizeof(configuration) ||
            !valid_config(&configuration)) memset(&configuration,0,sizeof(configuration));
        uint8_t gain=0;
        if(nvs_get_u8(h,"output_gain",&gain)==ESP_OK && gain>=1 && gain<=8)
            atomic_store(&playback_gain,gain);
        uint8_t mode=0;
        if(nvs_get_u8(h,"input_mode",&mode)==ESP_OK && panel_voice_input_mode_name(mode))
            atomic_store(&input_mode,mode);
        uint8_t volume=0;
        if(nvs_get_u8(h,"volume",&volume)==ESP_OK && volume<=100)(void)product_codec_volume_set(volume,NULL);
        len=sizeof(remote_records);
        if(nvs_get_blob(h,"remote_ids_v1",remote_records,&len)!=ESP_OK || len!=sizeof(remote_records))
            memset(remote_records,0,sizeof(remote_records));
        for(size_t i=0;i<8;++i)
            if(!memchr(remote_records[i].id,0,sizeof(remote_records[i].id)) ||
                (remote_records[i].id[0] && !panel_voice_uuid(remote_records[i].id)))
                memset(remote_records,0,sizeof(remote_records));
        nvs_close(h);
    }
    atomic_store(&active_mode,atomic_load(&input_mode));
    snprintf(status.agent_id,sizeof(status.agent_id),"%s",configuration.bot.agent_id);
    snprintf(status.agent_name,sizeof(status.agent_name),"%s",configuration.bot.name);
    state("idle",!nyx_creds_have_agent()?"Sign in to NyxID in Settings":
        configuration.bot.agent_id[0]?"Ready":"Select a Bot in Settings");
}
size_t panel_voice_bots(panel_voice_bot_t *out,size_t cap)
{
    if (!state_lock || !out) return 0;
    xSemaphoreTake(state_lock,portMAX_DELAY);
    size_t count=bot_count<cap?bot_count:cap;
    memcpy(out,bots,count*sizeof(*out));
    xSemaphoreGive(state_lock);
    return count;
}
esp_err_t panel_voice_configure(const char *agent)
{
    if (!state_lock || !panel_voice_uuid(agent)) return ESP_ERR_INVALID_ARG;
    bool expected=false;
    if (!atomic_compare_exchange_strong(&busy,&expected,true)) return ESP_ERR_INVALID_STATE;
    voice_config_t cfg={0};nyx_agent_credential_t credential={0};
    esp_err_t err=ESP_ERR_INVALID_ARG;
    xSemaphoreTake(state_lock,portMAX_DELAY);
    for (size_t i=0;i<bot_count;i++) if (!strcmp(bots[i].agent_id,agent)) {
        cfg.bot=bots[i];snprintf(cfg.key_id,sizeof(cfg.key_id),"%s",bots_key_id);break;
    }
    xSemaphoreGive(state_lock);
    if (!valid_config(&cfg) || !nyx_creds_read_agent(&credential) || strcmp(cfg.key_id,credential.key_id)) goto done;
    nvs_handle_t h;
    err=nvs_open(NYX_VOICE_NVS_NAMESPACE,NVS_READWRITE,&h);
    if (err==ESP_OK) {
        err=nvs_set_blob(h,"selection_v1",&cfg,sizeof(cfg));
        if (err==ESP_OK) err=nvs_commit(h);
        nvs_close(h);
    }
    if (err==ESP_OK) {
        xSemaphoreTake(state_lock,portMAX_DELAY);configuration=cfg;
        snprintf(status.agent_id,sizeof(status.agent_id),"%s",cfg.bot.agent_id);
        snprintf(status.agent_name,sizeof(status.agent_name),"%s",cfg.bot.name);xSemaphoreGive(state_lock);
        state("idle","Ready");
    }
done:
    memset(&credential,0,sizeof(credential));
    atomic_store(&busy,false);
    return err;
}
void panel_voice_stop(void) { atomic_store(&holding,false);atomic_store(&stopping,true); }
void panel_voice_hold(bool pressed) { atomic_store(&holding,pressed && atomic_load(&busy)); }
void panel_voice_microphone_mute(bool muted) { atomic_store(&microphone_muted,muted); }

esp_err_t panel_voice_input_mode_set(panel_voice_input_mode_t mode)
{
    if (!state_lock || !panel_voice_input_mode_name(mode)) return ESP_ERR_INVALID_ARG;
    bool expected=false;
    if (!atomic_compare_exchange_strong(&busy,&expected,true)) return ESP_ERR_INVALID_STATE;
    nvs_handle_t h;
    esp_err_t err=nvs_open(NYX_VOICE_NVS_NAMESPACE,NVS_READWRITE,&h);
    if(err==ESP_OK) {
        err=nvs_set_u8(h,"input_mode",(uint8_t)mode);
        if(err==ESP_OK)err=nvs_commit(h);
        nvs_close(h);
    }
    if(err==ESP_OK){atomic_store(&input_mode,mode);atomic_store(&active_mode,mode);}
    atomic_store(&busy,false);
    return err;
}

esp_err_t panel_voice_gain_set(unsigned gain)
{
    if (gain<1 || gain>8) return ESP_ERR_INVALID_ARG;
    nvs_handle_t h;
    esp_err_t err=nvs_open(NYX_VOICE_NVS_NAMESPACE,NVS_READWRITE,&h);
    if(err!=ESP_OK)return err;
    err=nvs_set_u8(h,"output_gain",(uint8_t)gain);
    if(err==ESP_OK)err=nvs_commit(h);
    nvs_close(h);
    if(err==ESP_OK)atomic_store(&playback_gain,gain);
    return err;
}
esp_err_t panel_voice_volume_set(unsigned volume)
{
    if(volume>100)return ESP_ERR_INVALID_ARG;
    int previous=product_codec_volume_get();
    esp_err_t err=product_codec_volume_set((int)volume,NULL);
    if(err!=ESP_OK)return err;
    nvs_handle_t h;err=nvs_open(NYX_VOICE_NVS_NAMESPACE,NVS_READWRITE,&h);
    if(err==ESP_OK){err=nvs_set_u8(h,"volume",(uint8_t)volume);if(err==ESP_OK)err=nvs_commit(h);nvs_close(h);}
    if(err!=ESP_OK)(void)product_codec_volume_set(previous,NULL);
    return err;
}

typedef struct { char *data; size_t size,used; bool overflow; } http_buffer_t;
static esp_err_t http_event(esp_http_client_event_t *event)
{
    http_buffer_t *b=event->user_data;
    if (event->event_id==HTTP_EVENT_ON_DATA && event->data_len>0) {
        if (b->overflow || (size_t)event->data_len>=b->size-b->used) {b->overflow=true;return ESP_ERR_INVALID_SIZE;}
        memcpy(b->data+b->used,event->data,event->data_len);b->used+=event->data_len;b->data[b->used]=0;
    }
    return ESP_OK;
}
static cJSON *http(voice_call_t *c,const char *suffix,cJSON *body)
{
    char url[512],auth[160];
    snprintf(url,sizeof(url),"%s/api/v1/assistant/nyxagent%s",NYX_AGENT_BASE_URL,suffix);
    snprintf(auth,sizeof(auth),"Bearer %s",c->credential.credential);
    http_buffer_t b={.data=calloc(1,MAX_CONTROL_BYTES),.size=MAX_CONTROL_BYTES};
    if (!b.data) return NULL;
    char *encoded=body?cJSON_PrintUnformatted(body):NULL;
    esp_http_client_config_t cfg={.url=url,.crt_bundle_attach=esp_crt_bundle_attach,
        .timeout_ms=15000,.buffer_size=8192,.buffer_size_tx=8192,.disable_auto_redirect=true,
        .event_handler=http_event,.user_data=&b};
    esp_http_client_handle_t client=esp_http_client_init(&cfg);
    esp_err_t err=ESP_ERR_NO_MEM;int code=0;
    if (client && (!body || encoded)) {
        esp_http_client_set_header(client,"Authorization",auth);
        esp_http_client_set_header(client,"Content-Type","application/json");
        char agent[64];snprintf(agent,sizeof(agent),"nyx-panel/%s",esp_app_get_description()->version);
        esp_http_client_set_header(client,"User-Agent",agent);
        if (body) {esp_http_client_set_method(client,HTTP_METHOD_POST);esp_http_client_set_post_field(client,encoded,strlen(encoded));}
        err=esp_http_client_perform(client);code=esp_http_client_get_status_code(client);
    }
    if (client) esp_http_client_cleanup(client);
    free(encoded);memset(auth,0,sizeof(auth));
    cJSON *result=NULL;
    if (err==ESP_OK && !b.overflow && code>=200 && code<300) result=cJSON_ParseWithLength(b.data,b.used);
    else {
        char message[128];
        cJSON *error=err==ESP_OK && !b.overflow?cJSON_ParseWithLength(b.data,b.used):NULL;
        const char *known=panel_voice_error_message(error,code);
        if (known) snprintf(message,sizeof(message),"%s",known);
        else snprintf(message,sizeof(message),"Voice connection failed (%d / %s)",code,esp_err_to_name(err));
        cJSON_Delete(error);
        state("error",message);
    }
    free(b.data);
    return result;
}
static void refresh_task(void *arg)
{
    voice_call_t *c=arg;
    cJSON *reply=http(c,"/voice/agents",NULL);
    panel_voice_bot_t *found=calloc(PANEL_VOICE_MAX_BOTS,sizeof(*found));size_t count=0;
    if (found && reply && panel_voice_parse_agents(reply,found,&count)) {
        xSemaphoreTake(state_lock,portMAX_DELAY);
        memcpy(bots,found,count*sizeof(*bots));bot_count=count;
        snprintf(bots_key_id,sizeof(bots_key_id),"%s",c->credential.key_id);
        bool selected=false;
        for(size_t i=0;i<count;i++) if (!strcmp(configuration.key_id,bots_key_id) &&
            !strcmp(configuration.bot.agent_id,bots[i].agent_id) &&
            !strcmp(configuration.bot.conversation_id,bots[i].conversation_id)) selected=true;
        if (!selected) {
            memset(&configuration,0,sizeof(configuration));status.agent_id[0]=status.agent_name[0]=0;
        }
        status.bots_revision++;
        xSemaphoreGive(state_lock);
        state("idle",selected?"Ready":count?"Choose a Bot below":"Create a Bot voice connection in NyxID, then allow it on this Key");
    } else if (reply) state("error","Could not read the Bot list");
    free(found);cJSON_Delete(reply);memset(c,0,sizeof(*c));free(c);
    atomic_store(&busy,false);vTaskDelete(NULL);
}
esp_err_t panel_voice_refresh_agents(void)
{
    if (!state_lock || !network_is_connected()) return ESP_ERR_INVALID_STATE;
    bool expected=false;
    if (!atomic_compare_exchange_strong(&busy,&expected,true)) return ESP_ERR_INVALID_STATE;
    voice_call_t *c=calloc(1,sizeof(*c));
    if (!c) {atomic_store(&busy,false);return ESP_ERR_NO_MEM;}
    if (!nyx_creds_read_agent(&c->credential)) {
        free(c);atomic_store(&busy,false);state("error","Sign in to NyxID in Settings");return ESP_ERR_INVALID_STATE;
    }
    state("loading","Loading your Bots from NyxID");
    if (xTaskCreate(refresh_task,"voice_bots",12288,c,3,NULL)!=pdPASS) {
        memset(c,0,sizeof(*c));free(c);atomic_store(&busy,false);return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
static void uuid(char out[37])
{
    uint8_t bytes[16];esp_fill_random(bytes,sizeof(bytes));bytes[6]=(bytes[6]&15)|64;bytes[8]=(bytes[8]&63)|128;
    snprintf(out,37,"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        bytes[0],bytes[1],bytes[2],bytes[3],bytes[4],bytes[5],bytes[6],bytes[7],bytes[8],bytes[9],bytes[10],bytes[11],bytes[12],bytes[13],bytes[14],bytes[15]);
}

// Caller holds remote_lock. A duplicate never opens another microphone or repeats speech.
static esp_err_t remote_record(const panel_voice_remote_t *r,bool open,bool commit,bool *duplicate)
{
    *duplicate=false;
    unsigned char digest[32],kind[2]={(unsigned char)open,(unsigned char)(open && r->microphone_muted)};
    mbedtls_sha256_context hash;mbedtls_sha256_init(&hash);
    int rc=mbedtls_sha256_starts(&hash,0);
    if(!rc)rc=mbedtls_sha256_update(&hash,kind,sizeof(kind));
    if(!rc)rc=mbedtls_sha256_update(&hash,(const unsigned char *)r->text,strlen(r->text));
    if(!rc)rc=mbedtls_sha256_finish(&hash,digest);
    mbedtls_sha256_free(&hash);if(rc)return ESP_FAIL;
    for(size_t i=0;i<8;++i)if(!strcasecmp(remote_records[i].id,r->request_id)){
        if(memcmp(remote_records[i].digest,digest,sizeof(digest)))return ESP_ERR_INVALID_STATE;
        *duplicate=true;return ESP_OK;
    }
    if(!commit)return ESP_OK;
    remote_record_t next[8];memcpy(next+1,remote_records,sizeof(next)-sizeof(next[0]));
    memset(&next[0],0,sizeof(next[0]));snprintf(next[0].id,sizeof(next[0].id),"%s",r->request_id);
    memcpy(next[0].digest,digest,sizeof(digest));
    nvs_handle_t h;esp_err_t err=nvs_open(NYX_VOICE_NVS_NAMESPACE,NVS_READWRITE,&h);
    if(err==ESP_OK){err=nvs_set_blob(h,"remote_ids_v1",next,sizeof(next));if(err==ESP_OK)err=nvs_commit(h);nvs_close(h);}
    if(err==ESP_OK)memcpy(remote_records,next,sizeof(next));
    return err;
}
static void remote_state(const char *request,const char *value)
{
    xSemaphoreTake(state_lock,portMAX_DELAY);
    if(!strcmp(status.remote_request_id,request))snprintf(status.remote_state,sizeof(status.remote_state),"%s",value);
    xSemaphoreGive(state_lock);
}
static void send_announcement(voice_call_t *c,const panel_voice_remote_t *request)
{
    char path[160];snprintf(path,sizeof(path),"/voice/sessions/%s/announcements",c->session);
    cJSON *body=cJSON_CreateObject();cJSON_AddStringToObject(body,"request_id",request->request_id);
    cJSON_AddStringToObject(body,"text",request->text);
    cJSON *reply=http(c,path,body);cJSON_Delete(body);
    cJSON *value=cJSON_GetObjectItemCaseSensitive(reply,"state");
    remote_state(request->request_id,cJSON_IsString(value)?value->valuestring:reply?"queued":"failed");
    cJSON_Delete(reply);
}
static int on_state(esp_peer_state_t value,void *ctx)
{
    voice_call_t *c=ctx;
    if (value==ESP_PEER_STATE_CONNECTED) atomic_store(&c->connected,true);
    if (value==ESP_PEER_STATE_DATA_CHANNEL_CONNECTED) {
        esp_peer_data_channel_cfg_t channel={.type=ESP_PEER_DATA_CHANNEL_RELIABLE,.ordered=true,.label="oai-events"};
        if (esp_peer_create_data_channel(c->peer,&channel)) {
            state("error","Voice data channel failed");atomic_store(&c->running,false);
        }
    }
    if (value==ESP_PEER_STATE_CONNECT_FAILED || value==ESP_PEER_STATE_DISCONNECTED) {
        if (atomic_exchange(&c->running,false) && !atomic_load(&stopping))
            state("error","Voice media disconnected");
    }
    return 0;
}
// Print only public protocol structure, never ICE credentials, candidates or SDP.
static void log_sdp_shape(const char *kind, const char *sdp)
{
    for (const char *p=sdp; p && *p;) {
        size_t n=strcspn(p, "\r\n");
        if (!strncmp(p,"a=setup:",8) || !strncmp(p,"a=mid:",6) || !strncmp(p,"m=",2))
            printf("Voice SDP %s: %.*s\n",kind,(int)(n>120?120:n),p);
        p=strchr(p,'\n'); if(p) ++p;
    }
}
static int on_offer(esp_peer_msg_t *msg,void *ctx)
{
    voice_call_t *c=ctx;
    if (msg->type!=ESP_PEER_MSG_TYPE_SDP) return 0;
    if (msg->size<=0 || msg->size>64*1024) return -1;
    char *copy=panel_voice_prepare_offer((const char *)msg->data,msg->size);
    if (!copy || xQueueSend(c->offers,&copy,0)!=pdTRUE) {free(copy);atomic_store(&c->running,false);return -1;}
    return 0;
}
static int on_audio(esp_peer_audio_frame_t *frame,void *ctx)
{
    voice_call_t *c=ctx;
    if (frame->size<=0 || frame->size>1500) return -1;
    audio_packet_t packet={.len=frame->size};memcpy(packet.bytes,frame->data,frame->size);
    if (xQueueSend(c->audio,&packet,0)!=pdTRUE) {
        // Never accumulate stale speech indefinitely behind the loudspeaker.
        audio_packet_t dropped;xQueueReceive(c->audio,&dropped,0);xQueueSend(c->audio,&packet,0);
    }
    xSemaphoreTake(state_lock,portMAX_DELAY);status.received_packets++;xSemaphoreGive(state_lock);
    return 0;
}
static int on_data(esp_peer_data_frame_t *frame,void *ctx)
{
    voice_call_t *c=ctx;
    if (frame->type==ESP_PEER_DATA_CHANNEL_STRING && frame->size>0 && frame->size<=MAX_CONTROL_BYTES) {
        bool started=panel_voice_provider_started((const char *)frame->data,frame->size);
        if (!c->provider_frames++ || started)
            printf("Voice provider event: bytes=%d started=%d\n",frame->size,started);
        if (started) atomic_store(&c->provider_ready,true);
    }
    return 0;
}
static int on_channel(esp_peer_data_channel_info_t *channel,void *ctx)
{
    (void)ctx; printf("Voice data channel opened: stream=%u\n",channel->stream_id); return 0;
}
static void peer_task(void *arg)
{
    voice_call_t *c=arg;
    while (atomic_load(&c->running)) {
        xSemaphoreTake(c->peer_lock,portMAX_DELAY);esp_peer_main_loop(c->peer);xSemaphoreGive(c->peer_lock);
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    xEventGroupSetBits(c->done,DONE_PEER);vTaskDelete(NULL);
}
static void mic_task(void *arg)
{
    voice_call_t *c=arg;void *encoder=NULL;size_t used=0;
    esp_opus_enc_config_t cfg=ESP_OPUS_ENC_CONFIG_DEFAULT();
    cfg.sample_rate=16000;cfg.channel=1;cfg.bitrate=24000;cfg.complexity=0;cfg.enable_vbr=true;
    int16_t mono[FRAME_SAMPLES];uint8_t encoded[2048];unsigned pts=0;
    aec_config_t config={.mic_num=1,.ref_num=1,.out_num=1,.filter_length=8,.sample_rate=16000,
        .caps=MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT,.mode=AEC_MODE_FD_LOW_COST,.nlp_level=AEC_NLP_LEVEL_NORMAL};
    aec_handle_t *aec=aec_create_from_config(&config);
    int chunk=aec ? aec_get_chunksize(aec) : 0;
    int16_t *capture=NULL,*mic=NULL,*ref=NULL,*clean=NULL;
    if (chunk>0 && chunk<=1024) {
        capture=heap_caps_aligned_alloc(16,chunk*4,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
        mic=heap_caps_aligned_alloc(16,chunk*2,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
        ref=heap_caps_aligned_alloc(16,chunk*2,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
        clean=heap_caps_aligned_alloc(16,chunk*2,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
    }
    if (!capture || !mic || !ref || !clean ||
        esp_opus_enc_open(&cfg,sizeof(cfg),&encoder)!=ESP_AUDIO_ERR_OK) {
        state("error","Voice echo cancellation could not start");atomic_store(&c->running,false);
    } else printf("Voice AEC ready: chunk=%d rate=16000 filter=8 volume=%d\n",chunk,product_codec_volume_get());
    while (atomic_load(&c->running)) {
        size_t got=0;
        if (product_i2s_read(capture,chunk*4,&got,1000)!=ESP_OK || got!=(size_t)chunk*4) {
            state("error","Voice microphone read failed");atomic_store(&c->running,false);break;
        }
        for (int i=0;i<chunk;i++) mic[i]=capture[2*i];
        xSemaphoreTake(c->reference_lock,portMAX_DELAY);
        panel_voice_reference_read(&c->reference,ref,chunk);
        xSemaphoreGive(c->reference_lock);
        aec_process(aec,mic,ref,clean);
        for (int i=0;i<chunk && atomic_load(&c->running);i++) {
            mono[used++]=clean[i];
            if (used!=FRAME_SAMPLES) continue;
            used=0;
            bool transmit=panel_voice_capture_allowed(c->input_mode,atomic_load(&holding),
                atomic_load(&microphone_muted),atomic_load(&c->controls),atomic_load(&c->connected),
                atomic_load(&c->provider_ready),atomic_load(&c->muted));
            if (!transmit) memset(mono,0,sizeof(mono));
            esp_audio_enc_in_frame_t input={.buffer=(uint8_t *)mono,.len=sizeof(mono)};
            esp_audio_enc_out_frame_t output={.buffer=encoded,.len=sizeof(encoded)};
            if (esp_opus_enc_process(encoder,&input,&output)!=ESP_AUDIO_ERR_OK) {
                state("error","Voice microphone encoding failed");atomic_store(&c->running,false);break;
            }
            if (!pts) printf("Voice codec stack free: encoder=%u bytes\n",(unsigned)uxTaskGetStackHighWaterMark(NULL));
            if (atomic_load(&c->connected)) {
                esp_peer_audio_frame_t frame={.data=encoded,.size=output.encoded_bytes,.pts=pts};
                xSemaphoreTake(c->peer_lock,portMAX_DELAY);int err=esp_peer_send_audio(c->peer,&frame);xSemaphoreGive(c->peer_lock);
                if (!err) {xSemaphoreTake(state_lock,portMAX_DELAY);status.sent_packets++;xSemaphoreGive(state_lock);}
            }
            pts+=20;
        }
    }
    if (encoder) esp_opus_enc_close(encoder);
    if (aec) aec_destroy(aec);
    free(capture);free(mic);free(ref);free(clean);
    xEventGroupSetBits(c->done,DONE_MIC);vTaskDelete(NULL);
}
static void speaker_task(void *arg)
{
    voice_call_t *c=arg;void *decoder=NULL;
    int16_t *mono=malloc(MAX_PCM_SAMPLES*sizeof(int16_t)),*stereo=malloc(MAX_PCM_SAMPLES*2*sizeof(int16_t));
    esp_opus_dec_cfg_t cfg=ESP_OPUS_DEC_CONFIG_DEFAULT();cfg.sample_rate=16000;cfg.channel=1;
    if (!mono || !stereo || esp_opus_dec_open(&cfg,sizeof(cfg),&decoder)!=ESP_AUDIO_ERR_OK) atomic_store(&c->running,false);
    audio_packet_t packet;bool first_frame=true;
    while (atomic_load(&c->running)) {
        if (xQueueReceive(c->audio,&packet,pdMS_TO_TICKS(100))!=pdTRUE) continue;
        esp_audio_dec_in_raw_t input={.buffer=packet.bytes,.len=packet.len};
        esp_audio_dec_out_frame_t output={.buffer=(uint8_t *)mono,.len=MAX_PCM_SAMPLES*sizeof(int16_t)};esp_audio_dec_info_t info={0};
        if (esp_opus_dec_decode(decoder,&input,&output,&info)!=ESP_AUDIO_ERR_OK || output.decoded_size>MAX_PCM_SAMPLES*2) {
            atomic_store(&c->running,false);break;
        }
        if (first_frame) {printf("Voice codec stack free: decoder=%u bytes\n",(unsigned)uxTaskGetStackHighWaterMark(NULL));first_frame=false;}
        size_t samples=output.decoded_size/2;
        unsigned gain=atomic_load(&playback_gain),source_peak=0,output_peak=0;
        for (size_t i=0;i<samples;i++) {
            unsigned peak=mono[i]<0 ? -(int32_t)mono[i] : mono[i];
            if(peak>source_peak)source_peak=peak;
            mono[i]=panel_voice_amplify_sample(mono[i],gain);
            peak=mono[i]<0 ? -(int32_t)mono[i] : mono[i];
            if(peak>output_peak)output_peak=peak;
            stereo[2*i]=stereo[2*i+1]=mono[i];
        }
        xSemaphoreTake(state_lock,portMAX_DELAY);
        if(source_peak>status.source_peak)status.source_peak=source_peak;
        if(output_peak>status.output_peak)status.output_peak=output_peak;
        xSemaphoreGive(state_lock);
        xSemaphoreTake(c->reference_lock,portMAX_DELAY);
        panel_voice_reference_push(&c->reference,mono,samples);
        xSemaphoreGive(c->reference_lock);
        size_t written=0;
        if (product_i2s_write(stereo,samples*4,&written,1000)!=ESP_OK) {atomic_store(&c->running,false);break;}
        atomic_fetch_add(&c->played_samples,samples);atomic_store(&c->last_audio_us,esp_timer_get_time());
        if(output_peak>400)atomic_store(&c->last_speech_us,esp_timer_get_time());
    }
    if (decoder) esp_opus_dec_close(decoder);
    free(mono);free(stereo);xEventGroupSetBits(c->done,DONE_SPEAKER);vTaskDelete(NULL);
}
static void socket_event(void *arg,esp_event_base_t base,int32_t id,void *data)
{
    (void)base;voice_call_t *c=arg;
    if (id==WEBSOCKET_EVENT_DISCONNECTED || id==WEBSOCKET_EVENT_ERROR) {
        atomic_store(&c->controls,false);
        if (atomic_exchange(&c->running,false) && !atomic_load(&stopping))
            state("error","Voice control connection was lost. Start a new call");
        return;
    }
    if (id!=WEBSOCKET_EVENT_DATA) return;
    esp_websocket_event_data_t *e=data;
    if (e->op_code!=0 && e->op_code!=1) return;
    if (e->payload_len<=0 || e->payload_len>=MAX_CONTROL_BYTES || e->payload_offset<0 || e->data_len<0 ||
        e->payload_offset+e->data_len>e->payload_len) {atomic_store(&c->running,false);return;}
    if (!e->payload_offset) {
        free(c->socket_text);c->socket_text=malloc(e->payload_len+1);c->socket_size=e->payload_len;c->socket_used=0;
    }
    if (!c->socket_text || c->socket_size!=(size_t)e->payload_len || c->socket_used!=(size_t)e->payload_offset) {atomic_store(&c->running,false);return;}
    memcpy(c->socket_text+c->socket_used,e->data_ptr,e->data_len);c->socket_used+=e->data_len;
    if (c->socket_used==c->socket_size) {
        c->socket_text[c->socket_used]=0;char *text=c->socket_text;c->socket_text=NULL;
        if (xQueueSend(c->messages,&text,0)!=pdTRUE) {free(text);atomic_store(&c->running,false);}
    }
}
static bool snapshot(voice_call_t *c,const char *text)
{
    cJSON *root=cJSON_Parse(text),*session=cJSON_GetObjectItemCaseSensitive(root,"session");
    const cJSON *id=cJSON_GetObjectItemCaseSensitive(session,"id"),*value=cJSON_GetObjectItemCaseSensitive(session,"state");
    if (!cJSON_IsString(id) || strcmp(id->valuestring,c->session) || !cJSON_IsString(value)) {cJSON_Delete(root);return false;}
    c->control_revision=(unsigned)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(session,"control_revision"));
    bool active=!strcmp(value->valuestring,"active");
    atomic_store(&c->controls,active);
    atomic_store(&c->muted,cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(session,"muted")) || cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(session,"input_muted")));
    panel_voice_captions_t captions;
    if (panel_voice_parse_captions(root,&captions) && memcmp(&captions,&c->last_captions,sizeof(captions))) {
        if (lcd_renderer_push_voice_captions(captions.user,captions.bot,15000)==ESP_OK)
            c->last_captions=captions;
    }
    const cJSON *notice;
    cJSON_ArrayForEach(notice,cJSON_GetObjectItemCaseSensitive(root,"announcements")){
        const cJSON *request=cJSON_GetObjectItemCaseSensitive(notice,"request_id");
        const cJSON *stage=cJSON_GetObjectItemCaseSensitive(notice,"state");
        if(cJSON_IsString(request) && cJSON_IsString(stage))remote_state(request->valuestring,stage->valuestring);
    }
    bool live=strcmp(value->valuestring,"closed") && strcmp(value->valuestring,"failed") && strcmp(value->valuestring,"closing");
    if (!live && !atomic_load(&stopping) &&
        (!strcmp(value->valuestring,"failed") || !atomic_load(&c->provider_ready)))
        state("error","Voice ended before audio was ready. Start a new call");
    cJSON_Delete(root);return live;
}
// Opus always advertises a 48 kHz stereo RTP clock, even for 16 kHz mono PCM.
// The encoder/decoder continue to use the board codec's native 16 kHz mono.
static esp_err_t open_peer(voice_call_t *c)
{
    esp_peer_default_cfg_t extra={.agent_recv_timeout=50,.data_ch_cfg={.send_cache_size=8192,.recv_cache_size=128*1024},
        .rtp_cfg={.send_pool_size=32768,.send_queue_num=64,.audio_recv_jitter={.cache_size=32768}}};
    esp_peer_cfg_t cfg={.role=ESP_PEER_ROLE_CONTROLLING,.audio_info={.codec=ESP_PEER_AUDIO_CODEC_OPUS,.sample_rate=48000,.channel=2},
        .audio_dir=ESP_PEER_MEDIA_DIR_SEND_RECV,.video_dir=ESP_PEER_MEDIA_DIR_NONE,.enable_data_channel=true,.manual_ch_create=true,
        .no_auto_reconnect=true,.extra_cfg=&extra,.extra_size=sizeof(extra),.on_state=on_state,.on_msg=on_offer,.on_audio_data=on_audio,
        .on_data=on_data,.on_channel_open=on_channel,.ctx=c};
    if (esp_peer_open(&cfg,esp_peer_get_default_impl(),&c->peer)) return ESP_FAIL;
    return ESP_OK;
}

esp_err_t panel_voice_peer_test(char *json,size_t cap)
{
    bool expected=false;
    if (!atomic_compare_exchange_strong(&busy,&expected,true)) return ESP_ERR_INVALID_STATE;
    voice_call_t *c=calloc(1,sizeof(*c));char *offer=NULL;esp_err_t err=ESP_ERR_NO_MEM;
    if (!c) goto done;
    c->offers=xQueueCreate(2,sizeof(char *));
    if (!c->offers) goto done;
    if ((err=open_peer(c))!=ESP_OK) goto done;
    atomic_store(&c->running,true);
    if (esp_peer_new_connection(c->peer)) {err=ESP_FAIL;goto done;}
    int64_t deadline=esp_timer_get_time()+10000000;
    while (!offer && atomic_load(&c->running) && esp_timer_get_time()<deadline) {
        esp_peer_main_loop(c->peer);
        xQueueReceive(c->offers,&offer,pdMS_TO_TICKS(10));
    }
    err=ESP_ERR_TIMEOUT;
    if (offer) {
        char *opus=strstr(offer,"opus/");
        if (opus && !strncmp(opus,"opus/48000/2",12)) err=ESP_OK;
        char mapping[64]={0};
        if(opus) {size_t n=strcspn(opus,"\r\n");if(n>=sizeof(mapping))n=sizeof(mapping)-1;memcpy(mapping,opus,n);}
        cJSON *root=cJSON_CreateObject();cJSON_AddBoolToObject(root,"ok",err==ESP_OK);
        cJSON_AddStringToObject(root,"opus_rtp",mapping);cJSON_AddNumberToObject(root,"pcm_rate",16000);
        cJSON_AddNumberToObject(root,"pcm_channels",1);cJSON_AddBoolToObject(root,"provider_contacted",false);
        char *text=cJSON_PrintUnformatted(root);if(text){snprintf(json,cap,"%s",text);free(text);}cJSON_Delete(root);
    }
done:
    free(offer);
    if(c) {
        if(c->peer)esp_peer_close(c->peer);
        if(c->offers){while(xQueueReceive(c->offers,&offer,0)==pdTRUE)free(offer);vQueueDelete(c->offers);}
        free(c);
    }
    atomic_store(&busy,false);return err;
}

static void call_task(void *arg)
{
    voice_call_t *c=arg;bool owns=false;unsigned tasks=0;char *sdp=NULL;
    state("connecting","Connecting through NyxID");
    cJSON *discovery=http(c,"/voice/agents",NULL),*preferences=NULL;
    char path[256];
    if (!discovery) goto done;
    panel_voice_bot_t *found=calloc(PANEL_VOICE_MAX_BOTS,sizeof(*found));size_t count=0;bool selected=false;
    if (found && panel_voice_parse_agents(discovery,found,&count)) {
        for(size_t i=0;i<count;i++) if (!strcmp(found[i].agent_id,c->config.bot.agent_id) &&
            !strcmp(found[i].conversation_id,c->config.bot.conversation_id)) selected=true;
    }
    free(found);
    if (!selected) {cJSON_Delete(discovery);state("error","Selected Bot is no longer available. Refresh Bots in Settings");goto done;}
    snprintf(path,sizeof(path),"/voice/options?conversation_id=%s",c->config.bot.conversation_id);
    cJSON *options=http(c,path,NULL);
    if (options) preferences=panel_voice_preferences(discovery,options,c->input_mode);
    cJSON_Delete(options);cJSON_Delete(discovery);
    if (!preferences) {state("error","Select an available WebRTC voice in NyxID voice settings");goto done;}
    if (product_audio_claim()!=ESP_OK) {state("error","Close the music or spectrum app first");goto done;}
    owns=true;
    if (product_codec_dev_resume()!=ESP_OK || product_codec_mute_set(false)!=ESP_OK) {state("error","Audio codec is unavailable");goto done;}
    c->peer_lock=xSemaphoreCreateMutex();c->reference_lock=xSemaphoreCreateMutex();c->audio=xQueueCreate(24,sizeof(audio_packet_t));
    c->offers=xQueueCreate(2,sizeof(char *));c->messages=xQueueCreate(4,sizeof(char *));c->done=xEventGroupCreate();
    if (!c->peer_lock || !c->reference_lock || !c->audio || !c->offers || !c->messages || !c->done) goto done;
    if (open_peer(c)!=ESP_OK) {state("error","Voice peer setup failed");goto done;}
    atomic_store(&c->running,true);
    if (esp_peer_new_connection(c->peer)) {state("error","Voice peer setup failed");goto done;}
    if (xTaskCreate(peer_task,"voice_rtc",12288,c,4,NULL)!=pdPASS) goto done;
    tasks|=DONE_PEER;
    int64_t deadline=esp_timer_get_time()+20000000;
    while (!atomic_load(&stopping) && atomic_load(&c->running) && !sdp && esp_timer_get_time()<deadline)
        xQueueReceive(c->offers,&sdp,pdMS_TO_TICKS(100));
    if (!sdp || atomic_load(&stopping)) goto done;
    log_sdp_shape("offer",sdp);
    char request[37];uuid(request);
    cJSON *body=cJSON_CreateObject();cJSON_AddStringToObject(body,"agent_id",c->config.bot.agent_id);cJSON_AddStringToObject(body,"client_request_id",request);cJSON_AddStringToObject(body,"sdp_offer",sdp);
    cJSON_AddItemToObject(body,"preferences",preferences);preferences=NULL;
    snprintf(path,sizeof(path),"/voice/sessions");
    cJSON *reply=http(c,path,body);cJSON_Delete(body);free(sdp);sdp=NULL;
    cJSON *session=cJSON_GetObjectItemCaseSensitive(reply,"session"),*id=cJSON_GetObjectItemCaseSensitive(session,"id");
    cJSON *answer=cJSON_GetObjectItemCaseSensitive(reply,"sdp_answer");
    if (!cJSON_IsString(id) || strlen(id->valuestring)>=sizeof(c->session) || !cJSON_IsString(answer)) {cJSON_Delete(reply);goto done;}
    strcpy(c->session,id->valuestring);
    xSemaphoreTake(state_lock,portMAX_DELAY);
    snprintf(status.session_id,sizeof(status.session_id),"%s",c->session);
    xSemaphoreGive(state_lock);
    c->generation=(unsigned)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(session,"generation"));
    c->control_revision=(unsigned)cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(session,"control_revision"));
    log_sdp_shape("answer",answer->valuestring);
    esp_peer_msg_t msg={.type=ESP_PEER_MSG_TYPE_SDP,.data=(uint8_t *)answer->valuestring,.size=strlen(answer->valuestring)};
    xSemaphoreTake(c->peer_lock,portMAX_DELAY);int err=esp_peer_send_msg(c->peer,&msg);xSemaphoreGive(c->peer_lock);cJSON_Delete(reply);
    if (err || atomic_load(&stopping)) goto done;
    char url[512],headers[192];
    const char *base=NYX_AGENT_BASE_URL;
    bool secure=!strncmp(base,"https://",8);
    snprintf(url,sizeof(url),"%s%s/api/v1/assistant/nyxagent/voice/sessions/%s/stream",
        secure?"wss":"ws",base+(secure?5:4),c->session);
    snprintf(headers,sizeof(headers),"Authorization: Bearer %s\r\n",c->credential.credential);
    esp_websocket_client_config_t ws={.uri=url,.headers=headers,.crt_bundle_attach=esp_crt_bundle_attach,
        .disable_auto_reconnect=true,.buffer_size=8192,.task_stack=8192,.network_timeout_ms=10000};
    c->socket=esp_websocket_client_init(&ws);memset(headers,0,sizeof(headers));
    if (!c->socket) goto done;
    esp_websocket_register_events(c->socket,WEBSOCKET_EVENT_ANY,socket_event,c);
    if (esp_websocket_client_start(c->socket)!=ESP_OK) goto done;
    if (xTaskCreate(mic_task,"voice_mic",MIC_STACK_BYTES,c,4,NULL)!=pdPASS) goto done;
    tasks|=DONE_MIC;
    if (xTaskCreate(speaker_task,"voice_spk",SPEAKER_STACK_BYTES,c,4,NULL)!=pdPASS) goto done;
    tasks|=DONE_SPEAKER;
    int64_t heartbeat=0;deadline=esp_timer_get_time()+25000000;
    while (atomic_load(&c->running) && !atomic_load(&stopping)) {
        char *text=NULL;
        if (xQueueReceive(c->messages,&text,pdMS_TO_TICKS(100))==pdTRUE) {
            bool live=snapshot(c,text);free(text);if (!live) break;
        }
        bool active=atomic_load(&c->connected) && atomic_load(&c->controls) && atomic_load(&c->provider_ready);
        xSemaphoreTake(state_lock,portMAX_DELAY);status.active=active;status.playback_ms=atomic_load(&c->played_samples)/16;xSemaphoreGive(state_lock);
        if (active) {
            bool muted=atomic_load(&microphone_muted) || atomic_load(&c->muted);
            bool speaking=atomic_load(&c->last_speech_us)>0 &&
                esp_timer_get_time()-atomic_load(&c->last_speech_us)<250000;
            state("active",muted ? "Microphone muted" : speaking ? "Bot speaking" :
                c->input_mode==PANEL_VOICE_INPUT_AUTOMATIC ? "Listening - automatic" :
                atomic_load(&holding) ? "Listening" : "Hold to talk");
            if(c->remote.request_id[0]) {
                if(c->remote.text[0])send_announcement(c,&c->remote);
                else remote_state(c->remote.request_id,"active");
                memset(&c->remote,0,sizeof(c->remote));
            }
            remote_message_t notice;
            if(remote_messages && xQueueReceive(remote_messages,&notice,0)==pdTRUE && !strcmp(notice.session,c->session))
                send_announcement(c,&notice.request);
        }
        else if (esp_timer_get_time()>deadline) {state("error","Voice connection timed out");break;}
        if (atomic_load(&c->controls) && esp_timer_get_time()-heartbeat>1000000) {
            char event[256];snprintf(event,sizeof(event),"{\"type\":\"heartbeat\",\"generation\":%u,\"revision\":%u,\"playback_ms\":%u,\"playback_audible\":%s}",
                c->generation,++c->heartbeat_revision,atomic_load(&c->played_samples)/16,
                esp_timer_get_time()-atomic_load(&c->last_audio_us)<200000 && product_codec_volume_get()>0?"true":"false");
            if (esp_websocket_client_send_text(c->socket,event,strlen(event),pdMS_TO_TICKS(2000))<0) break;
            heartbeat=esp_timer_get_time();
        }
    }
done:
    atomic_store(&holding,false);atomic_store(&c->running,false);
    xSemaphoreTake(state_lock,portMAX_DELAY);status.active=false;
    if(!strcmp(status.remote_state,"connecting") || !strcmp(status.remote_state,"queued"))
        snprintf(status.remote_state,sizeof(status.remote_state),"%s",atomic_load(&stopping)?"cancelled":"failed");
    remote_message_t discarded;
    while(remote_messages && xQueueReceive(remote_messages,&discarded,0)==pdTRUE){}
    xSemaphoreGive(state_lock);
    if (c->session[0]) {
        char command[37];uuid(command);snprintf(path,sizeof(path),"/voice/sessions/%s/control",c->session);
        cJSON *end=cJSON_CreateObject();cJSON_AddStringToObject(end,"command_id",command);cJSON_AddNumberToObject(end,"expected_revision",c->control_revision);cJSON_AddStringToObject(end,"action","end");
        cJSON *reply=http(c,path,end);
        if (!atomic_load(&stopping) && panel_voice_normal_end(reply))
            state("idle","Call ended");
        cJSON_Delete(reply);cJSON_Delete(end);
    }
    if (c->socket) {esp_websocket_client_stop(c->socket);esp_websocket_client_destroy(c->socket);}
    if (tasks) xEventGroupWaitBits(c->done,tasks,pdFALSE,pdTRUE,portMAX_DELAY);
    if (c->peer) esp_peer_close(c->peer);
    if (c->audio) vQueueDelete(c->audio);
    for (int i=0;i<2;i++) {QueueHandle_t q=i?c->offers:c->messages;char *text; if(q){while(xQueueReceive(q,&text,0)==pdTRUE)free(text);vQueueDelete(q);}}
    if (c->done) vEventGroupDelete(c->done);
    if (c->peer_lock) vSemaphoreDelete(c->peer_lock);
    if (c->reference_lock) vSemaphoreDelete(c->reference_lock);
    free(sdp);free(c->socket_text);cJSON_Delete(preferences);
    if (owns) {product_codec_dev_resume();product_audio_release();}
    xSemaphoreTake(state_lock,portMAX_DELAY);status.active=false;bool failed=!strcmp(status.state,"error");xSemaphoreGive(state_lock);
    if (!failed || atomic_load(&stopping)) state("idle","Ready");
    atomic_store(&active_mode,atomic_load(&input_mode));
    memset(c,0,sizeof(*c));free(c);atomic_store(&busy,false);vTaskDelete(NULL);
}
static esp_err_t start_call(const panel_voice_remote_t *remote)
{
    if (!state_lock || !network_is_connected()) return ESP_ERR_INVALID_STATE;
    bool expected=false;
    if (!atomic_compare_exchange_strong(&busy,&expected,true)) return ESP_ERR_INVALID_STATE;
    voice_call_t *c=calloc(1,sizeof(*c));
    if (!c) {atomic_store(&busy,false);return ESP_ERR_NO_MEM;}
    xSemaphoreTake(state_lock,portMAX_DELAY);c->config=configuration;status.sent_packets=status.received_packets=status.playback_ms=status.source_peak=status.output_peak=0;xSemaphoreGive(state_lock);
    if (!valid_config(&c->config) || !nyx_creds_read_agent(&c->credential) || strcmp(c->config.key_id,c->credential.key_id)) {
        memset(c,0,sizeof(*c));free(c);atomic_store(&busy,false);
        state("error","Sign in and select a Bot in Settings");return ESP_ERR_INVALID_STATE;
    }
    c->input_mode=remote?PANEL_VOICE_INPUT_AUTOMATIC:(panel_voice_input_mode_t)atomic_load(&input_mode);
    if(remote){
        bool duplicate=false;esp_err_t err=remote_record(remote,true,true,&duplicate);
        if(err!=ESP_OK){memset(c,0,sizeof(*c));free(c);atomic_store(&busy,false);return err;}
        c->remote=*remote;
    }
    xSemaphoreTake(state_lock,portMAX_DELAY);
    status.session_id[0]=status.remote_request_id[0]=status.remote_state[0]=0;status.remote_initiated=remote!=NULL;
    if(remote){snprintf(status.remote_request_id,sizeof(status.remote_request_id),"%s",remote->request_id);
        snprintf(status.remote_state,sizeof(status.remote_state),"connecting");}
    xSemaphoreGive(state_lock);
    atomic_store(&active_mode,c->input_mode);
    atomic_store(&stopping,false);atomic_store(&holding,false);atomic_store(&microphone_muted,remote && remote->microphone_muted);
    state("connecting","Connecting through NyxID");
    if (xTaskCreate(call_task,"voice_call",16384,c,3,NULL)!=pdPASS) {
        if(remote)remote_state(remote->request_id,"failed");
        atomic_store(&active_mode,atomic_load(&input_mode));
        memset(c,0,sizeof(*c));free(c);atomic_store(&busy,false);
        state("error","Voice resources unavailable. Try again");return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
esp_err_t panel_voice_start(void){return start_call(NULL);}
esp_err_t panel_voice_remote_open(const panel_voice_remote_t *r,bool *duplicate)
{
    if(!remote_lock || !remote_messages || !r || !duplicate || !panel_voice_uuid(r->request_id))return ESP_ERR_INVALID_ARG;
    xSemaphoreTake(remote_lock,portMAX_DELAY);
    esp_err_t err=remote_record(r,true,false,duplicate);
    if(err==ESP_OK && !*duplicate)err=start_call(r);
    if(err==ESP_OK && !*duplicate)(void)lcd_renderer_show_voice();
    xSemaphoreGive(remote_lock);return err;
}
esp_err_t panel_voice_remote_say(const panel_voice_remote_t *r,bool *duplicate)
{
    if(!remote_lock || !remote_messages || !r || !duplicate || !panel_voice_uuid(r->request_id) || !r->text[0])return ESP_ERR_INVALID_ARG;
    xSemaphoreTake(remote_lock,portMAX_DELAY);
    esp_err_t err=remote_record(r,false,false,duplicate);
    if(err==ESP_OK && !*duplicate){
        xSemaphoreTake(state_lock,portMAX_DELAY);
        if(!status.active || atomic_load(&stopping) || uxQueueSpacesAvailable(remote_messages)==0)err=ESP_ERR_INVALID_STATE;
        else{
            remote_message_t message={.request=*r};snprintf(message.session,sizeof(message.session),"%s",status.session_id);
            err=remote_record(r,false,true,duplicate);
            if(err==ESP_OK){
                if(xQueueSend(remote_messages,&message,0)!=pdTRUE)err=ESP_ERR_INVALID_STATE;
                else{snprintf(status.remote_request_id,sizeof(status.remote_request_id),"%s",r->request_id);
                    snprintf(status.remote_state,sizeof(status.remote_state),"queued");}
            }
        }
        xSemaphoreGive(state_lock);
    }
    xSemaphoreGive(remote_lock);return err;
}

esp_err_t panel_voice_test(char *json,size_t cap)
{
    bool expected=false;
    if (!atomic_compare_exchange_strong(&busy,&expected,true)) return ESP_ERR_INVALID_STATE;
    esp_err_t err=product_audio_claim();
    if (err!=ESP_OK) {atomic_store(&busy,false);return err;}
    err=product_codec_dev_resume();
    int16_t pcm[FRAME_SAMPLES*2];unsigned peak=0;double sum=0;unsigned samples=0;
    if (err==ESP_OK) err=product_codec_mute_set(false);
    for (int frame=0;err==ESP_OK && frame<25;frame++) {
        for(int i=0;i<FRAME_SAMPLES;i++)pcm[2*i]=pcm[2*i+1]=(int16_t)(1500*sin(2*3.141592653589793*440*(frame*FRAME_SAMPLES+i)/16000));
        size_t wrote=0;err=product_i2s_write(pcm,sizeof(pcm),&wrote,1000);
    }
    for(int frame=0;err==ESP_OK && frame<50;frame++) {
        size_t got=0;err=product_i2s_read(pcm,sizeof(pcm),&got,1000);
        for(unsigned i=0;i<got/4;i++){int s=pcm[2*i];unsigned a=s<0?-s:s;if(a>peak)peak=a;sum+=(double)s*s;samples++;}
    }
    snprintf(json,cap,"{\"ok\":%s,\"sample_rate\":16000,\"channels\":1,\"samples\":%u,\"peak\":%u,\"rms\":%.1f}",err==ESP_OK?"true":"false",samples,peak,samples?sqrt(sum/samples):0);
    product_audio_release();atomic_store(&busy,false);return err;
}
