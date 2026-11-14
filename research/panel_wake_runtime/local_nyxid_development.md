# Local NyxID for development-board voice checks

The wake model can be evaluated offline, but the wake-to-conversation check
also needs a reachable NyxID backend. Check the board's configured API origin
after changing networks. An idle listener with voice state `error` does not
establish a working conversation path.

Use a clean NyxID worktree at the intended server revision. Preserve any
uncommitted work in the main checkout. Keep the build target, Mongo data,
environment file, signing keys and logs outside the panel repository.

```sh
CARGO_TARGET_DIR="$NYXID_RESEARCH_TARGET" CARGO_PROFILE_DEV_DEBUG=0 \
  cargo build --locked --manifest-path "$NYXID_RESEARCH_WORKTREE/backend/Cargo.toml" \
  --bin nyxid-server -j 2
```

Wait for that original build to finish. Keep its log and exit status; an
observation timeout is not a reason to start another build. Start the database
when the backend is ready, using the existing local data and replica-set name.
For the existing panel test database the port is27127 and replica set is
`nyx-panel-voice`. Limit the WiredTiger cache explicitly:

```sh
"$NYXID_RESEARCH_MONGOD" --dbpath "$NYXID_RESEARCH_DB" \
  --port 27127 --bind_ip 127.0.0.1 --replSet nyx-panel-voice \
  --wiredTigerCacheSizeGB 0.25 --logpath "$NYXID_RESEARCH_MONGO_LOG" --fork
```

Load the existing private server environment in the backend process. Preserve
its database URL, encryption key and JWT key paths so its saved local accounts
and provider connections remain readable. Set `BASE_URL` to the current LAN
API origin and `FRONTEND_URL` to the corresponding frontend origin. Update
`CORS_ALLOWED_ORIGINS` and `CSRF_TRUSTED_ORIGINS` for that frontend as needed.
These process overrides do not require rewriting the private environment file.
Keep the environment and authentication responses out of shared logs.

Check `GET /health` on the backend before changing the development board.
For a new development firmware profile, set both
`NYX_PANEL_AGENT_DEV_API_URL` and `NYX_PANEL_AGENT_DEV_LOGIN_URL`; the latter
ends in `/login/agent-key` on the same LAN host. Set
`NYX_PANEL_NODE_DEV_MODE=ON` to keep node registration and services local too.
The profile hash gives Agent Key, voice and node settings separate NVS
namespaces. A new origin therefore needs the board's own login and local
authorization. A Mac account or CLI key is not a substitute for the board key.
Provider credentials stay in NyxID.

After onboarding, verify the selected Bot, its voice grant and its independent
grant to the development board's services. Then test wake, buffered first
utterance, reply, hangup/rearm and reconnect. API health, an offline recognition
result or a successful firmware build alone does not establish those behaviors.
Retain the passive stability monitor and identify the development board before
any flash. The dining-room production board is outside this workflow.
