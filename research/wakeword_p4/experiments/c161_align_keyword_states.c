/* Emit PocketSphinx's live features and phone/state alignments for one marked training prefix. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pocketsphinx.h>
#include "pocketsphinx_internal.h"
#include "ps_alignment_internal.h"

static int process(ps_decoder_t *ps, const int16_t *pcm, size_t samples)
{
    if (ps_start_utt(ps) < 0)
        return -1;
    for (size_t i = 0; i < samples; i += 480) {
        size_t count = samples-i < 480 ? samples-i : 480;
        if (ps_process_raw(ps, pcm+i, count, 0, 0) < 0)
            return -1;
    }
    return ps_end_utt(ps);
}

int main(int argc, char **argv)
{
    if (argc != 4)
        return 1;
    FILE *in = fopen(argv[3], "rb");
    if (!in || fseek(in, 0, SEEK_END))
        return 1;
    long bytes = ftell(in);
    if (bytes <= 0 || bytes % 2 || fseek(in, 0, SEEK_SET))
        return 1;
    int16_t *pcm = malloc(bytes);
    if (!pcm || fread(pcm, 1, bytes, in) != (size_t)bytes)
        return 1;
    fclose(in);
    ps_config_t *config = ps_config_init(NULL);
    ps_config_set_str(config, "hmm", argv[1]);
    ps_config_set_str(config, "dict", argv[2]);
    ps_config_set_str(config, "loglevel", "ERROR");
    ps_decoder_t *ps = ps_init(config);
    if (!ps)
        return 1;
    char *cmn = strdup(ps_get_cmn(ps, 0));
    if (!cmn || ps_set_align_text(ps, "hey nicks") < 0 || process(ps, pcm, bytes/2) < 0)
        return 1;
    if (ps_set_alignment(ps, NULL) < 0)
        return 2;
    if (ps_reinit_feat(ps, NULL) < 0 || ps_set_cmn(ps, cmn) < 0)
        return 1;
    acmod_set_grow(ps->acmod, 1);
    if (process(ps, pcm, bytes/2) < 0)
        return 2;
    ps_alignment_t *al = ps_get_alignment(ps);
    if (!al)
        return 1;
    for (int i = 0; i < al->state.n_ent; i++) {
        ps_alignment_entry_t *s = &al->state.seq[i];
        ps_alignment_entry_t *phone = &al->sseq.seq[s->parent];
        ps_alignment_entry_t *word = &al->word.seq[phone->parent];
        const char *name = dict_wordstr(ps->dict, word->id.wid);
        if (strcmp(name, "hey") && strcmp(name, "nicks"))
            continue;
        for (int j = 0; j < s->duration; j++) {
            int frame = s->start+j;
            mfcc_t **f = acmod_get_frame(ps->acmod, &frame);
            int32_t header[] = {frame, phone->id.pid.cipid, i-phone->child, s->id.senid};
            if (!f || fwrite(header, sizeof(int32_t), 4, stdout) != 4)
                return 1;
            for (int k = 0; k < 3; k++) {
                if (fwrite(f[k], sizeof(float), 13, stdout) != 13)
                    return 1;
            }
        }
    }
    ps_free(ps);
    ps_config_free(config);
    free(cmn);
    free(pcm);
    return 0;
}
