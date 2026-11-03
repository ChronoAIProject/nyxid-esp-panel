# NyxID × ESP32-P4 功能审计与集成建议

本文件保留集成前的独立审计快照；当前实现和实测状态分别见
[ONBOARDING.md](ONBOARDING.md) 与 [VERIFICATION.md](VERIFICATION.md)。

审计时间：2026-11-03 15:28–15:39（Asia/Singapore，UTC+08）。
范围：上游源码、发布记录、实际部署的公开只读接口、当前板子和现有工程。未改业务源码、未创建设备/令牌/服务、未发送外部消息、未部署。

## 1. 结论与第一批范围

第一批建议完成三个用户可见结果：

1. 系统 Settings → WLAN 支持手动联网及 Camera 扫 Wi-Fi 二维码。
2. Settings → NyxID 支持在板子上显示设备码及授权二维码，用户在手机完成明确授权，板子自动注册真正的 WSS 控制节点、创建对应的自定义服务，最后仅保存节点运行凭据。
3. 独立 Nyx Panel App 专注图文消息；Camera 负责预览与扫码；系统 Settings 承担网络、账号绑定、设备状态、更新等公共入口。

**首批不要启用旧 device_code.c，不要把硬件绑定 API key 或 bootstrap token 写进 WSS 节点令牌槽。** 当前板子没有调用云服务的客户端功能，硬件绑定 API key 不会增强现有图文推送/远控，却增加第二套身份、有限有效期及尚无实现的续期问题。完整控制节点登录可以用 NyxID 现有账号 device login，不需要新增 Ed25519/libsodium 依赖；这对当前仅约 145 KiB 的固件余量尤其合适。

## 2. 权威版本与实际部署

| 对象 | 实查结果 |
| --- | --- |
| 权威上游 | https://github.com/ChronoAIProject/NyxID |
| 默认分支 | main |
| 最新 main | ff302d5ecfe8d188cf0f167995e87bc08c722296，2026-11-03 15:00 +08 |
| 最新发布 | v0.77.0，2026-11-02 13:32:18 UTC |
| 发布对应 commit | 88e101bbf24fe3812a97f8a27a9c51c37aff1d03 |
| 独立源码目录 | /Users/auric/tmp/nyxid-board-feature-audit |
| 原有 /Users/auric/nyxid | 停在 2026-10-15 的 897a89c2；本次没有更新或切换它 |
| 板子当前 WSS 主机 | nyx-api.chrono-ai.fun |
| GET /health | version 0.77.0、commit 2c0d66d66f25 |
| 该部署 commit 完整值 | 2c0d66d66f25cb591682fb2bc778238248a33343，2026-11-03 02:36:43 UTC |
| GET /api/v1/public/config | frontend_url=https://nyxid.ai，node_ws_url=wss://api.nyxid.ai/api/v1/nodes/ws；GitHub/Google/Apple 可用，email/password 关闭 |
| 当前 CLI | 本机已安装 0.71.0，不能当成最新 CLI 功能的证据 |
| 当前板子 | LAN 192.168.1.247，fw 0.1.14，Nyx state=up，last_error 为空 |
| 当前控制节点 | 名称 nyx-panel-d2ef63，在线 |
| 当前自定义服务 | slug nyx-panel-d2ef63，active，auth_method=none，endpoint=http://127.0.0.1:80，指向该节点 |

最新 main 与实际部署不是同一 commit。已将部署 commit 只 fetch 到独立审计目录，并用 git diff 对比：device_code_service、auth_device、node_service、node_admin、routes、auth guard、keys、proxy 等本报告核心协议源码无差异。后续 main 的 backend 差异只在 mcp_transport、oidc_discovery、user_tokens、cookie_domain 四个文件，不能用 main 的新版发现文档替代当前线上返回值。

只读线上验证：

- curl GET /auth/device/request、/auth/device/poll、/auth/device/v2/request、/devices/code/request、/devices/code/poll、/devices/onboard/redeem、/nodes/register-token 均返回 405，Allow: POST，证明相关路由在该部署存在。
- GET /api/v1/keys 无认证返回 401。
- 通过已有授权 CLI 的只读 node list / service show 证实上述在线节点及自定义服务路由。
- LAN GET /status 与经过 Nyx 的 GET /status 均成功，证明现有 WSS 普通代理链工作。
- 没有发起真实新登录、批准、注册、兑换或创建服务，因此“路由已部署 + 源码协议吻合”与“新设备完整绑定已实机验收”应明确区分。

## 3. 三类设备身份必须分清

| 流程 | 发起端点 | 最终交付 | 主要用途 | 能否直接连接现有节点 WSS |
| --- | --- | --- | --- | --- |
| 账号 device login | /api/v1/auth/device/request；或 v2/request | 第一方 account_session 的 access_token + refresh_token；v2 也允许人工选择 Agent Key | 临时完整账号操作，或独立受限客户端 | 否；先用账号授权创建注册令牌，再走 WSS register |
| 硬件 device-code binding | /api/v1/devices/code/request | scoped nyxid_ag_ API key、node_id、refresh_token | 设备作为客户端调用获准的 NyxID 服务 | 否；这里 node 是所有权记录 |
| WSS node-agent registration | 已授权用户 POST /api/v1/nodes/register-token，随后 WSS 首帧 register | node_id、nyx_nauth_ auth_token、signing_secret | NAT 后的控制面板/私有服务接收云端代理请求 | 是 |

这不是标准 OAuth /oauth/device_authorization → /oauth/token device grant。线上 OIDC 发现也未宣告 device_code grant；此处是 NyxID 的第一方 /api/v1/auth/device/* 协议，接口和错误码必须按其自身契约实现。

硬件绑定返回的 refresh_token 在最新源码中只生成、返回、保存哈希，没有任何实际验证/消费它来更新设备凭据的 endpoint。全文检索 refresh_token_hash 与 redeemed_refresh_token_hash 均未发现读取消费的续期实现。不能声称可自动续期。返回 expires_in=86400 是交付信封/记录生命周期，**不是 API key 实际到期时间**；以 api_key_expires_at 为准，当前新 key 的默认策略是有限期（通常 90 天）。

依据：

- backend/src/services/node_service.rs:531 明确说明 hardware device 通过 proxy API 认证，不是 node WebSocket agent。
- backend/src/services/device_code_service/mod.rs:70、129：两种硬件交付都没有 auth_token 或 signing_secret。
- backend/src/services/device_code_service/approve.rs:66、251：scope=proxy、allow_all_nodes=false、allowed_node_ids 仅刚创建的设备记录。
- backend/src/handlers/node_admin.rs:1056：非 all-nodes API key 无权创建节点注册令牌；management write 还要求 write/admin scope。
- backend/src/routes.rs:3428、3429、3432、3448：节点管理、POST /keys、自定义服务写操作受 human-only guard 约束，拒绝 API key、SA、delegated、relay。
- backend/src/mw/auth.rs:1339、1449：第一方 token 通过 sid 关联真实可撤销账号会话。

上游固定源码入口：
https://github.com/ChronoAIProject/NyxID/blob/2c0d66d66f25cb591682fb2bc778238248a33343/backend/src/services/node_service.rs#L531

## 4. 最小可靠的板上登录 → 控制节点 → 自定义服务

建议先用稳定、账号专用的 legacy request，而非让用户意外选到无法完成管理操作的受限 Agent Key。v2 可以使用，但拿到 auth_kind=agent_key 时应清楚提示权限不适合本次设备注册，停止；不能把 key 偷换成节点凭据或扩大授权。

### 4.1 登录契约

无凭据 POST https://目标API/api/v1/auth/device/request：

~~~json
{
  "client_label": "Nyx Panel",
  "client_user_agent": "nyxid-esp-panel/0.1.x",
  "client_app": "nyxid-esp-panel",
  "client_platform": "esp32p4",
  "client_model": "Waveshare ESP32-P4 10.1"
}
~~~

返回 device_code（秘密，不显示/记录）、user_code（八字符，显示 XXXX-XXXX）、verification_uri、verification_uri_complete、expires_in=600、interval=5。

板子显示大号 user_code、有效期、服务名称以及 verification_uri_complete 的二维码；使用服务器返回的可信链接。此为板子拥有的 QR 流程，不是 CLI 流程；CLI 的最新安全规则要求代码和链接分开，不应错误套用到板子二维码。批准者可用已登录网页或 NyxID app；扫码只定位请求，用户仍需确认这是自己的请求并明确批准。当前部署 email auth 关闭，UI 不应指导用户输 NyxID 密码。

无认证 POST /api/v1/auth/device/poll：

~~~json
{"device_code":"<仅RAM中的秘密>"}
~~~

当前源码成功返回：

~~~json
{
  "auth_kind":"account_session",
  "access_token":"<temporary>",
  "refresh_token":"<temporary>",
  "token_type":"Bearer",
  "expires_in":900
}
~~~

需要处理的错误：

| error_code | HTTP | 行为 |
| --- | --- | --- |
| 11202 authorization_pending | 400 | 按当前 interval 等待 |
| 11203 slow_down | 429 | 当前 interval 增加 5 秒；后续一直使用增加后的间隔 |
| 11204 access_denied | 403 | 停止并显示“本次授权已拒绝”，提供重新开始 |
| 11201 expired_token | 410 | 停止并显示到期，用户点“重新获取” |
| 11205 already_delivered | 410 | 不再轮询；若未持有交付结果，重新人工授权 |
| 11200 not_found | 404 | 本次请求无效，重新开始 |
| 11206 rate_limited | 429 | 退避；不得立即新建请求形成循环 |

首次 poll 等待 interval；不要收到 pending 就忙轮询。网络超时可重试，但 approved 交付是一次性的，响应丢失不能保证重新获取。倒计时使用单调时钟，HTTPS证书验证和 WSS 签名所需 UTC 时间应由系统联网后的 SNTP 提供。

源码：backend/src/handlers/auth_device.rs:27、59、69、74、172、290；backend/src/services/auth_device_service.rs:35；docs/API.md:503。

### 4.2 注册真正节点

临时 Bearer access_token：

~~~text
POST /api/v1/nodes/register-token
{"name":"nyx-panel-<稳定硬件后缀>"}
~~~

返回 token_id、token（nyx_nreg_ 前缀）、name、expires_at。默认注册令牌有效期 1 小时，一次使用。token_id 已保留为未来的 node_id，适合恢复日志；org 选项应由经过验证的用户选择提供 owner_user_id，不接受二维码任意注入组织 ID。

连接 public/config 返回并通过信任校验的 node_ws_url。WSS 首帧：

~~~json
{
  "type":"register",
  "token":"<nyx_nreg_...>",
  "metadata":{"agent_version":"0.1.x","os":"esp-idf","arch":"riscv32"}
}
~~~

成功首包：

~~~json
{
  "type":"register_ok",
  "node_id":"<uuid>",
  "auth_token":"<nyx_nauth_...>",
  "signing_secret":"<64 hex chars>"
}
~~~

先验证并原子提交这三项，再进入普通 auth/heartbeat 运行模式。NVS中的现有 api_key 命名实际上存节点 auth_token，建议内部重命名或至少在API层明确 node_auth_token；不要让名字促使后续开发者混用。

源码：backend/src/handlers/node_admin.rs:44、133、993；backend/src/handlers/node_ws.rs:54、962；backend/src/services/node_service.rs:435、480。

### 4.3 创建板子服务

仍使用同一个临时账号 access_token，与 NyxID CLI 当前实现一致调用 POST /api/v1/keys：

~~~json
{
  "label":"Nyx Panel",
  "slug":"nyx-panel-<稳定硬件后缀>",
  "endpoint_url":"http://127.0.0.1:80",
  "auth_method":"none",
  "auth_key_name":"",
  "node_id":"<register_ok node_id>"
}
~~~

不提供 service_slug（这是 catalog slug，面板是 custom endpoint），不上传服务 credential。响应包含 service id、slug、node_id 等；保存返回的最终 slug，不能仅假设请求值即最终路由。custom HTTP 的路由由 UserService.node_id 决定，无需再创建 catalog NodeServiceBinding。

创建后读取 /keys/{id或slug} 并确认 node_id、active、auth_method，再用临时账号 GET /proxy/s/{slug}/status 验证实际板子响应。服务端可能有人工审批策略，UI应显示“等待批准”或错误，而不是将 API 接受等同于可控。

源码：backend/src/handlers/keys.rs:289、897；backend/src/services/unified_key_service.rs:1680、1823；cli/src/commands/service.rs:865。

### 4.4 收尾、恢复与取消

- 完成后用临时 Bearer POST /api/v1/auth/logout，再擦除 RAM 中 access_token、refresh_token、device_code、注册令牌。不要把用户 token 写入 NVS。logout 撤销本次 sid 与相关 refresh tokens；不退出审批者浏览器，但源码也会结束该用户无 Bearer 的 MCP fallback sessions。
- 尚未 approved 的 Cancel/Back 停止本地 poll、清掉秘密并还原 Settings。没有可供匿名 requester 使用的 cancel endpoint，服务器请求自行到期；不要冒用 /auth/device/deny。
- 用户在 HTTP poll 未完成时取消，结果可能稍后到达；必须使用 generation/cancel 状态拦截，若已收到了账号 token，就撤销该临时 session后擦除，避免UI退出后后台继续注册。
- 注册与服务创建必须是可恢复的阶段状态机。register_ok 后断网/重启，不要再创建新节点；显示“设备已注册，服务设置尚未完成”，下次人工授权继续创建服务。
- 记录非秘密 token_id/阶段；如果注册令牌已被消费但 register_ok 丢失，用户再次授权后 GET /nodes/{token_id} 确认节点，再 POST /nodes/{id}/rotate-token 取回一对新的 auth_token/signing_secret，断开旧连接再保存/重连。不要靠重复 register 消费已使用令牌。
- 自定义 slug 409 时，GET /keys/{slug}，仅当 owner/node_id 确实符合本设备预期才复用。绝不能覆盖别的节点同名服务。
- 已经绑定且在线的 v0.1.14 板子不应自动触发新登录。换账号/重新绑定需明确用户操作，保留旧工作身份直至新流程成功；本地“断开”与云端“撤销节点”区别应在UI写清。
- 节点凭据没有 OAuth refresh；服务端可 rotate-token 或删除节点。401/auth_error时停止快速重连并引导重新授权/修复，不能循环设备登录。

源码：backend/src/handlers/auth.rs:860；backend/src/services/token_service.rs:1242；backend/src/handlers/node_admin.rs:1225。

## 5. 硬件 device-code 与 QR onboarding 真实协议（后续能力）

### 硬件设备码

POST /api/v1/devices/code/request：

- device_pubkey 必须是32字节真正 Ed25519 公钥的标准 Base64，非hex。
- hw_id 1–256字符；suggested_label 可选。
- 返回15分钟 device_code、12字符 XXXX-XXXX-XXXX user_code、verification_uri_complete、poll_interval=5。
- user_code 每30秒旋转，只接受当前一代批准；pending poll 返回 current_user_code，需要同步更新文字和授权二维码。
- poll JSON字段为 device_code、timestamp（Unix秒）、signature（64字节真正 Ed25519签名的标准Base64）。
- 签名消息：ASCII "nyxid:device-code:poll:v1" || base64url_decode(device_code)得到的32字节 || timestamp的有符号i64大端8字节。
- 时间误差限±10秒，要求 timestamp 单调增加；三次签名失败触发1小时公钥锁定。
- approved 一次性交付 api_key/node_id/refresh_token/expires_in/api_key_expires_at；后续410/9505。
- 9500不存在、9501过期、9502签名/时间错误、9506限流、9507锁定；Denied会返回403。9508 slow_down仅定义在错误枚举，在本硬件poll service里没有实际返回路径。
- **当前服务端内部矛盾：** poll_interval=5秒，但同一 /devices/code/* 专属IP限流为5次/分钟（request/poll/approve共用），公钥桶也有限流。按广告间隔持续poll会429。若以后选此流程需先修正/验证服务器限流契约；不要在固件中默认为5秒永远可成功。
- 此身份默认没有任意服务权限。审批可选择 default_services，allow_auto_connected_services也需人工明确选；没有选择时不应认为拥有所有服务。

源码：backend/src/crypto/device_code.rs:12、43、85；backend/src/services/device_code_service/mod.rs:25；poll.rs:24、160；backend/src/mw/rate_limit.rs:1590；backend/src/main.rs:993。

### QR onboarding

**上游 docs/device-code-binding.md 的 key/node/refresh 直塞QR示例过时，必须以实际实现为准。**

已登录人类网页 POST /api/v1/devices/onboard：

~~~json
{"label":"Kitchen panel","default_services":[]}
~~~

请求不包含 Wi-Fi 密码。返回：

~~~text
qr_payload = nyxprov://bootstrap?token=nyx_obt_...&id=<bootstrap_uuid>&url=<encoded_api_base>&exp=900
bootstrap_id, label, expires_in, expires_at
~~~

前端/CLI本地合成：

~~~text
nyxprov://full?token=nyx_obt_...&id=...&url=...&exp=900&ssid=...&psw=...
~~~

URLSearchParams 编码规则，包括 + 代表空格。Wi-Fi详情只在浏览器/CLI组成QR，不上传NyxID。设备联网后 POST 该可信 API 的 /api/v1/devices/onboard/redeem：

~~~json
{"bootstrap_token":"nyx_obt_..."}
~~~

一次性交付 api_key、node_id、refresh_token、expires_in、api_key_expires_at。没有 signing_secret、node auth_token 或可变的身份用途metadata；服务端固定 metadata.provisioning_source="device-onboard"。过期、已使用、被撤销通常410。人类可以 DELETE /devices/onboard/{bootstrap_id} 撤销未消费QR；已兑换后该endpoint不会替代运行API key/节点的撤销管理。

如果未来支持此QR，必须与WSS节点身份分开存储；识别 nyx_obt_、id/url/exp 字段后进入 bootstrap 分支，绝不当成旧 nyxprov://full?token=<node_auth>&node=<uuid> 的形式。对任意二维码url进行HTTPS与可信部署域名校验，不发送秘密到任意扫描URL。

源码：backend/src/handlers/devices.rs:60、250；backend/src/services/device_code_service/onboard.rs:43、155、273、353；frontend/src/pages/devices-onboard.tsx:119、548。

## 6. 完整功能族盘点与板子适用性

这是对最新源码/接口文档的产品能力覆盖，不意味着这些功能全部已在当前部署启用。

| 功能族 | 上游能力 | 当前板子与建议 |
| --- | --- | --- |
| 身份认证 | 第一方账号、社交登录、MFA、登录批准/设备码、账号session、OAuth/OIDC PKCE、token exchange、DPoP/客户端证书、broker binding | 立即接入第一方临时设备登录用于节点注册；一般第三方OAuth/完整网页登录不塞进P4 |
| Agent Key/API key | 服务/node/operation隔离、独立child credential、到期、轮换、撤销、scope plan | 板子今后作为云API客户端时再加；WSS token不可混用 |
| 组织/RBAC | 个人/组织所有权、角色/组、scope、共享服务/节点、转移、审计 | 服务器支持；首批个人绑定足够，组织选择后续可加 |
| 凭据代理 | HTTP任意自定义服务、catalog、认证注入、OpenAPI/AsyncAPI发现、WebSocket/SSE、服务选择/偏好 | 已有面板custom service与普通proxy；立即可继续用于远控 |
| MCP自动工具 | scoped discovery、REST/OpenAPI转MCP、工具schema/自定义端点 | 服务器可让AI调用面板API；建议后续为面板发布可信静态OpenAPI，板子不用运行MCP服务器 |
| 凭据节点 | NAT穿透WSS、注册、token轮换、HMAC、心跳、状态、负载/多节点故障切换 | 板子已支持auth/心跳/代理/HMAC；注册尚缺，需要首批打通 |
| 节点本地凭据 | 本地加密store、远端pending metadata、X25519/HKDF/XChaCha20远程密文投递、node target更新 | 通用私有凭据节点用途；板子无下游秘密注入需求，首批不实现 |
| 二进制/流式上传 | NYXM proxy_upload v1/v2、签名metadata、顺序/中止、响应流 | 板子v1 receiver存在；普通proxy仍buffer/Base64，当前流式请求选择以machine ingress为条件；不可宣称普通 /lcd/image 已二进制 |
| 私有SSH/终端 | SSH tunnel、cert/node-key、exec、Web terminal、WebSocket代理 | 主要服务器/桌面节点功能；P4无shell，不适合照搬 |
| Machine | 文件、桌面、浏览器、shell、进程、容器/VM、hosted/separated runtime、租约/授权 | 不适合ESP32-P4；勿为了二进制上传伪装Machine身份 |
| LLM与语音 | 多提供商网关、模型服务、TTS/STT/voice/dictation、计费和provider配置 | 硬件有音频但没有相应产品流；后续独立语音App，经有限权限客户端调用 |
| NyxBot/Assistant | 会话、图文附件、invocation状态/取消、specialists、共享agent/crew、后台任务 | 后续可做独立聊天App；现Nyx Panel是远端推送显示器，不等于NyxBot聊天 |
| Channel/Bot | Telegram/Discord/Lark/Feishu/email、电话/SMS、WhatsApp gateway、bot/link/channel管理 | 主要云端；可通过服务器agent把消息推给板子，面板不实现各平台SDK |
| 自定义事件 | POST /channel-events/{conversation_id}，API key绑定会话、UUID去重、每会话限流、单次callback | 后续传感器/摄像检测有价值；需带callback_url的正确agent key及归属会话，普通hardware API key并不会自动可发事件 |
| 通知/审批 | Telegram与Expo移动推送、service审批、grant、设备绑定成功/锁定通知 | 服务器已有；ESP不是Expo手机，不能直接注册为手机push接收端；面板现有图文推送仍走custom proxy |
| Webhooks/自动化 | inbound triggers、watches/jobs、事件definition与custom channels、callback delivery | 可用于云端规则触发板子显示/控制；首批不新增复杂自动化UI |
| Monitoring | chrono-probe scoped API、service monitoring配置、单次probe/最近结果 | 服务器支持；当前面板仅连接状态/简明健康信息，监控卡片可后续 |
| 服务目录/集成SDK | AI/邮件/代码/文档/日历/云服务catalog、OAuth托管/BYO app、CLI插件、SDK | 这些是云端能力；面板作为受控终端可间接受益 |
| Billing/Admin | credits/recurring allocations、usage、tariffs、权限键、feature flags、审计链、security readiness | 留在Web控制台；面板只给余额/不可用等必要用户反馈 |
| OTA/固件 | 未发现通用ESP固件发布/签名清单/设备OTA服务；CLI/容器更新属于其他执行环境 | 板子自己的URL/raw OTA已存在，Nyx只承担授权代理；Settings可显示版本、更新阶段与失败原因 |
| Wi-Fi/二维码 | Nyx onboarding构造短期bootstrap；标准WIFI:是通用协议，非Nyx OAuth | 首批直接实现标准WIFI:与旧nyxprov://wifi；Wi-Fi无需额外云身份 |

依据入口：上游 README.md:88、docs/API.md目录、docs/NODE_PROXY.md、docs/NYXID_NODE.md、docs/CHANNEL_EVENT_GATEWAY.md、docs/chat/09-owner-cli-api.md、docs/chat/11-voice.md、backend/src/routes.rs。

事件限制的关键源码：backend/src/handlers/channel_events.rs:41；backend/src/services/channel_event_service.rs:135、399。事件信封为 event_id(UUID)、source、type、timestamp(RFC3339)、payload/metadata；要求 caller key 与 conversation.agent_api_key_id 完全相同且该 key 有 callback_url。不是随便分配的device key可加入任意人的会话。

普通代理二进制选择依据：backend/src/handlers/proxy.rs:4405 明确 stream_upload = machine_ingress.is_some() && ...；5779后才走 node_dispatch.proxy_upload。不能仅看到receiver与capabilities就宣布普通HTTP上传已使用二进制。

## 7. 现有板子逐项差距

| 项目 | 状态 | 证据与处理 |
| --- | --- | --- |
| 现有在线WSS、自定义服务、LAN恢复 | 已支持且本次只读验证 | nyx_node.c、local_api.c；Nyx/LAN GET /status成功 |
| 普通代理Base64兼容、文本与JPEG显示 | 已支持；沿用原有验收 | NyxPanel.cpp:206、221；JPEG上限64 KiB |
| 真正WSS注册 | 尚未支持 | nyx_proto.c:379只构造auth；上游register/register_ok另需新增 |
| 旧device-code | 代码存在，不能启用 | Kconfig默认n；device_code.c:105只端点名称正确，公钥hex/假SHA512签名/ts和ts_sig/开机秒均错；未读verification_uri_complete、轮转code、slow_down/denied、refresh，且339写入同一WSS凭据 |
| QR扫描 | 解码/摄像路径存在，物理扫码新流程未验收 | qr_scanner.c:127仅nyxprov，wifi/nyx/full/mqtt；没有标准WIFI: |
| 最新bootstrap QR | 不兼容，当前会拒绝 | full分支必须有node；bootstrap只有id，无node。修复时不能只放宽检查然后误写token |
| QR UI反馈 | 需产品化 | qr_scanner.c:370直接存储并3秒重启，Camera.cpp:69仅静态提示；需确认SSID、连接中、失败/重试/取消、返回调用页 |
| Wi-Fi系统Settings | 已有 | Settings/ui/WlanPage.cpp；扫码可作为其动作，统一使用C6 station网络路径 |
| 系统状态页 | 代码存在但不合适 | NyxPanel.cpp:141在内容App里直接显示status JSON/UART说明，应移到Settings卡片 |
| proxy_upload v1 | 代码存在、未证明端到端 | nyx_upload.c/nyx_security.c/nyx_node.c:502/526；只有signing secret时advertise |
| OTA | 板子实现已有 | panel_ota.c，URL下载SHA256/双槽/回滚，LAN raw上传需节点token；不应说NyxID提供ESP OTA产品 |
| 摄像/WHIP | 已有板级能力 | Camera、media_mode_manager、stream_publisher，扫码须共享摄像资源并正确返回 |
| MQTT | 默认关闭 | 不应由于QR旧分支存在就作为本轮新功能 |

## 8. 推荐信息架构和交互

- Settings → WLAN：网络列表、已连接SSID/IP；“扫描 Wi-Fi 二维码”打开Camera扫码模式。扫描后只展示SSID和加密方式，用户确认连接，不显示密码；连不上时保留可返回的设置入口。
- Settings → NyxID：未绑定显示“连接 NyxID”，需要网络时直接引导WLAN。授权页展示8字符码、二维码、剩余时间、“取消”。批准后显示“注册设备 → 设置远程访问 → 连接成功”；成功页显示设备名称和在线状态，隐藏所有密钥。
- Settings → Device/About：固件版本、网络、Nyx连接、摄像状态、更新状态；对人显示“网络未连接”“时间同步中”“授权已失效”“服务尚未创建”等，不显示JSON/error dump。
- Camera：常规拍摄/预览与扫码模式明确区分；扫码时有取景框、亮度/距离提示、停止按钮；Back释放preview/QR任务并返回发起页。不要把扫描失败、过期bootstrap等错误写入消息内容App。
- Nyx Panel：消息列表、图片及时间；新的远控文本/图片可唤起这个App；Wi-Fi、账号设置、设备调试不放内容页。
- remote screen别名保留兼容：home→内容App；settings/wifi→Settings/WLAN；camera→Camera；scan→Camera扫码并在结束返回Settings；info/operator/device_bind→对应Settings状态/绑定页。

标准WIFI:建议至少支持 T:WPA/WEP/nopass、S:、P:、H:true，正确处理反斜线转义的分号/冒号/反斜线、Unicode SSID、开放网络空密码，并拒绝不支持的EAP/企业配置。保留nyxprov://wifi?ssid=...&psw=...兼容，但不要静默截断payload或允许重复关键字段/嵌入NUL。确认前不写NVS，不盲目重启；连接成功后再提交可靠配置，失败可回退旧网络。

## 9. 优先级与可执行验收

P0：先完成Settings/Camera/内容App职责拆分；标准Wi-Fi QR；新的full-account device login控制节点注册状态机；旧device_code保持关闭。无需客户端API key、bootstrap兑换或libsodium。

P1：给面板custom API发布静态OpenAPI供Nyx MCP发现；提供可读的注册中断恢复/重新授权/撤销与OTA进度页；保持现有远控兼容。

P2：只有新建云API消费、语音App或传感器事件产品需求后才增加独立client key身份及bootstrap扫码。届时先解决设备key续期与硬件poll限流冲突，再做独立身份设计。

验收顺序：

1. 主机验证真实HTTP契约与错误：pending、slow_down增加间隔、denied、expired、delivered；stub/mock只辅助状态机，不当成线上成功。
2. 构建检查固件slot余量；确认new full login不引入不必要Ed25519库；运行现有host checks和idf build。
3. 用屏幕显示真实授权QR，手机扫码并明确批准；检查板子拿到account_session后完成register_ok与custom service；读取服务确认node_id一致。绝不由agent自行批准。
4. 用第二个已有授权客户端 GET /status、POST /lcd/text、POST JPEG、打开Settings/Camera并返回；验板上的实际显示，不只看HTTP 200。
5. 重启板子，验证无需账号token即可WSS重新认证。确认NVS仅保留node运行凭据/slug/阶段，无用户session token；临时session logout后该账号Bearer应401，WSS仍up。
6. 在pending、register_ok前后、service创建前后分别Cancel/断网/重启；确保停止后台动作、可恢复，不产生无穷新节点、不覆盖原工作凭据。模拟register_ok丢包，验证token_id恢复而非重复注册。
7. 物理扫描标准WIFI QR：有转义/Unicode/开放网络/错误密码/取消；确认SSID、C6连接状态以及重启后的保存一致。扫码不泄露密码、不把错误写进消息App。
8. 已有v0.1.14身份升级保留；图文、LAN恢复、三轮Camera、WHIP持续流及OTA原有验收不退化。
9. proxy_upload单独标为尚待端到端验证；只有实际传输观测到NYXM且板子正确显示才改变状态。普通proxy继续明确为Base64。

本次没有执行会创建或批准身份的步骤，也没有修改现有板子状态；这些是后续实现完成后必须实际走通的验收。
