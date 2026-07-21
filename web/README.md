# DeckWeaver Web

A small FastAPI + React frontend for the DeckWeaver CLI. Adds user
login, file upload, a job queue with live progress and ETA, project
history with delete, mode selection (full vs. text-only), and an
auto-update poller against the GitHub remote.

The web layer is **completely additive** — nothing here changes how
the existing CLI works. `python scripts/convert.py ...` keeps behaving
exactly as before. The web app simply spawns the same CLI as a
subprocess.

```
web/
├── backend/                   FastAPI app + SQLite + worker queue
│   ├── app/
│   │   ├── main.py            Entry point, lifespan, route wiring
│   │   ├── config.py          Env-driven settings
│   │   ├── db.py, models.py   SQLAlchemy: users + jobs tables
│   │   ├── security.py        bcrypt + JWT
│   │   ├── deps.py            FastAPI deps (current_user, db)
│   │   ├── queue.py           Single-worker async queue
│   │   ├── runner.py          Subprocess wrapper around convert.py
│   │   ├── eta.py             Rolling-average ETA estimator
│   │   ├── github_sync.py     Periodic git fetch / pull / re-exec
│   │   ├── ws.py              WebSocket broker
│   │   └── routes/            auth, jobs, system
│   ├── manage.py              Admin CLI: create-user, reset-password
│   └── requirements.txt
├── frontend/                  React + Vite + TS
│   └── src/{api,pages,components}
├── data/                      Gitignored: SQLite + uploads + outputs
├── start.sh                   Dev launcher (uvicorn + vite)
├── start-prod.sh              Build frontend + uvicorn (single port)
└── .env.example
```

## Quick start (dev)

```bash
# from repo root
cp web/.env.example web/.env       # edit at minimum admin password + JWT secret
bash web/start.sh
```

Backend → http://localhost:8000, frontend → http://localhost:5173.
Vite proxies `/api` and `/ws` to the backend, so you can browse the
frontend without any extra config.

## Dependency layout

CLI-only users never need to install web-layer dependencies. The
declarations are deliberately split into two files:

| File | What it has | Who installs it |
| --- | --- | --- |
| `requirements.txt` (repo root) | PaddleOCR, opencv, python-pptx, … (CLI pipeline) | Anyone running `scripts/convert.py` |
| `web/backend/requirements.txt` | FastAPI, uvicorn, SQLAlchemy, passlib, bcrypt, … | Only users deploying the web layer |

`scripts/bootstrap.sh` only installs the CLI side. `web/start.sh` only
adds the web side (and only on first run, when `import fastapi` fails).
Both sets land in the same Python interpreter — there is no separate
virtualenv — so when the web backend spawns `scripts/convert.py` as a
subprocess it inherits the OCR / pipeline dependencies the CLI user
already has. If you want a true isolated env, create one yourself
with `python -m venv` before running either install.

## Quick start (prod-ish, single port)

```bash
bash web/start-prod.sh
# defaults to 0.0.0.0:8000, both frontend and API on the same origin
PORT=9000 bash web/start-prod.sh   # override
```

`start-prod.sh` builds the frontend (`vite build`) and starts uvicorn.
If `web/frontend/dist/` exists, FastAPI serves it automatically.

## Configuration

Everything is environment-driven (`web/.env` is auto-loaded). See
`web/.env.example` for the full list. Most important:

| Env | Default | Notes |
| --- | --- | --- |
| `DECKWEAVER_ADMIN_USERNAME` | `admin` | Seeded on first launch if no admin exists. |
| `DECKWEAVER_ADMIN_PASSWORD` | `admin` | **Change this before exposing the service.** |
| `DECKWEAVER_JWT_SECRET` | placeholder | Long random string. Tokens are invalidated when this changes. |
| `DECKWEAVER_PYTHON_BIN` | `python3` | Interpreter used to spawn `scripts/convert.py`. Use the same Python you ran `scripts/bootstrap.sh` with so the conversion has all its deps. |
| `DECKWEAVER_AUTO_UPDATE` | `true` | If true, the backend periodically `git fetch`es and pulls + restarts when behind. |
| `DECKWEAVER_UPDATE_POLL_SECONDS` | `600` | Poll interval. |
| `DECKWEAVER_GIT_BRANCH` | `main` | Branch tracked for auto-update. |
| `DECKWEAVER_USE_VLM` | `false` | If true, the runner invokes `scripts/convert_vlm.py` (cloud VLM) instead of the default local-OCR `scripts/convert.py`. Lower fidelity — text + shapes only, no extracted picture objects. Requires the `LLM_*` vars below and `httpx` (`requirements-vps.txt`). |
| `DECKWEAVER_LLM_BASE` | _(unset)_ | OpenAI-compatible base URL for the VLM profile (e.g. `https://api.example.com/v1`). Required when `DECKWEAVER_USE_VLM=true`. |
| `DECKWEAVER_LLM_KEY` | _(unset)_ | Bearer token sent as `Authorization: Bearer …` to the VLM endpoint. Required when `DECKWEAVER_USE_VLM=true`. |
| `DECKWEAVER_LLM_MODEL` / `DECKWEAVER_LLM_FALLBACK` | `gpt-5.5` / `gpt-5.4` | Primary and fallback model names. |
| `DECKWEAVER_LLM_PARALLEL` | `2` | Concurrent page requests to the VLM endpoint. |
| `DECKWEAVER_MAX_LONG_EDGE` | `1280` | Max pixel size of the page PNG sent to the VLM (longer edge). |

## User management

Admin is seeded from env on first launch. Add more users either via
the API (admin-gated) or the bundled CLI:

```bash
python3 web/backend/manage.py create-user alice s3cret
python3 web/backend/manage.py create-user bob s3cret --admin
python3 web/backend/manage.py reset-password admin newpass
python3 web/backend/manage.py list-users
```

## API surface

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/api/auth/login` | Returns JWT. |
| `GET`  | `/api/auth/me` | Current user. |
| `GET`  | `/api/users` | Admin: list users. |
| `POST` | `/api/users` | Admin: create user. |
| `DELETE` | `/api/users/{id}` | Admin: delete user. |
| `POST` | `/api/jobs` | Multipart upload (`files[]` + `mode`). |
| `GET`  | `/api/jobs` | List my jobs (or all if admin). |
| `GET`  | `/api/jobs/{id}` | Job detail + ETA. |
| `DELETE` | `/api/jobs/{id}` | Remove job + uploaded/produced files. |
| `GET`  | `/api/jobs/{id}/download` | `slides.pptx`. |
| `GET`  | `/api/jobs/{id}/logs` | Last 200 log lines. |
| `GET`  | `/api/system/version` | Commit, behind/ahead, auto-update state. |
| `POST` | `/api/system/update` | Admin: trigger pull + restart now. |
| `WS`   | `/ws/jobs?token=...` | Pushes job + system events. |

## How ETA is computed

Per `(mode, page_count)` history is recomputed from the `jobs` table on
demand: the last 20 successful runs in each mode produce an average
`seconds_per_page`. A queued job's ETA is

```
eta = Σ(remaining_pages_of_each_ahead × seconds_per_page[their_mode])
    + remaining_pages_of_self × seconds_per_page[my_mode]
```

Cold-start defaults are intentionally pessimistic (60s/page for full,
25s/page for text-only) so the first ETA isn't laughably low.

## Auto-update flow

1. Every `UPDATE_POLL_SECONDS` the backend runs `git fetch` against the
   configured remote/branch.
2. If `HEAD` is behind, queue is paused (no new jobs accepted), current
   job is allowed to finish, then `git pull --ff-only`.
3. If `requirements.txt` or `web/backend/requirements.txt` changed
   (detected via `git hash-object`), `pip install -r` runs.
4. The process re-execs itself (`os.execv`) with the same argv. Uvicorn
   rebinds and resumes from a clean state. Any jobs that were running
   at restart get marked `failed` with `error_msg="interrupted by restart"`.

Set `DECKWEAVER_AUTO_UPDATE=false` to disable the pull step (poller
still runs and surfaces the `behind` count in the UI so admins can
click "Update now" manually).

## Security model

DeckWeaver Web is designed to run in two modes:

- **Single-machine tool** — trusted local user, simple deploy. Defaults are conservative but the hardening on top is optional.
- **Public network deploy** — untrusted users hitting it from the internet. Several knobs must be set correctly, and you need a reverse proxy with TLS in front. The boot-time secret check refuses to start with placeholder values precisely so you don't ship the dev config to production by accident.

### What's built in

| Layer | What it does | Where |
| --- | --- | --- |
| **Auth** | bcrypt password hashing; JWT (HMAC-SHA256) with TTL; admin-only user creation (no public signup); login throttled per source IP (10/min default); constant-time-ish login response | [security.py](backend/app/security.py), [routes/auth.py](backend/app/routes/auth.py) |
| **Boot-time checks** | Refuse to start when `DECKWEAVER_ADMIN_PASSWORD` is a known weak default or `DECKWEAVER_JWT_SECRET` is too short / still placeholder. Override with `DECKWEAVER_REQUIRE_SECURE_SECRETS=false` for trusted local deploys | [main.py:_check_secrets](backend/app/main.py) |
| **Upload hardening** | Per-file and per-request size caps, file-count cap, file-extension whitelist, magic-byte sanity check, zip-slip / zip-bomb / symlink-in-zip protection | [routes/jobs.py](backend/app/routes/jobs.py) |
| **Per-user limits** | Max concurrent (`queued`+`running`) jobs per user (default 2), max total stored jobs per user (default 50). Returns `429`/`409` to abusers | [routes/jobs.py:create_job](backend/app/routes/jobs.py) |
| **Subprocess sandbox** | `convert.py` is spawned through a FS sandbox so it can only write to its own job dirs + model caches + `/tmp` + `/dev`. Auto-detected: `sandbox-exec` on macOS, `bwrap` or `firejail` on Linux | [sandbox.py](backend/app/sandbox.py) |
| **Resource limits** | `RLIMIT_AS` (memory), `RLIMIT_CPU` (CPU time), `RLIMIT_FSIZE` (single-file size) applied via `preexec_fn` so they propagate to every descendant (including LibreOffice) | [sandbox.py:make_preexec](backend/app/sandbox.py) |
| **Env scrubbing** | The subprocess sees only an explicit whitelist of env vars (PATH, HOME, locale, model-cache dirs, the GPU toggle). Secrets in the web server's env never reach `convert.py` | [sandbox.py:safe_env](backend/app/sandbox.py) |
| **HTTP hardening** | CSP, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Permissions-Policy`, HSTS. Outer body-size guard rejects oversized requests before route reads body | [middleware.py](backend/app/middleware.py) |
| **Auto-update is OFF by default** | A compromised upstream auto-pull = RCE. Opt in with `DECKWEAVER_AUTO_UPDATE=true` only when you trust the publisher | [config.py](backend/app/config.py), [github_sync.py](backend/app/github_sync.py) |
| **Retention sweeper** | Background task deletes finished jobs older than `DECKWEAVER_JOB_RETENTION_DAYS` (default 30). Files on disk go too | [retention.py](backend/app/retention.py) |
| **Isolation between users** | DB queries always scope by `owner_id` for non-admins; you can't read or download another user's jobs | [routes/jobs.py](backend/app/routes/jobs.py), [ws.py](backend/app/ws.py) |
| **Crash recovery** | Jobs left `running` when the process died (e.g. on auto-update restart) are marked `failed` on next boot rather than re-spawning blindly | [queue.py:boot_recover](backend/app/queue.py) |

### Public-deploy checklist

If you're putting this on the open internet, the application layer above is only half of it. Do all of these:

1. **TLS termination by a reverse proxy.** Put nginx or Caddy in front of uvicorn. Issue a Let's Encrypt cert. Forward `/api/*` and `/ws/*` to `127.0.0.1:8000`. Let the proxy serve `web/frontend/dist/` directly if you want.
2. **Bind uvicorn to localhost only.** `--host 127.0.0.1`. The proxy is the only thing that talks to it. Never expose uvicorn directly.
3. **Set strong secrets.** Long random `DECKWEAVER_JWT_SECRET` (≥ 32 chars). Real `DECKWEAVER_ADMIN_PASSWORD`. Do NOT set `DECKWEAVER_REQUIRE_SECURE_SECRETS=false`. The boot-time check exists for this reason.
4. **Run as a non-root user.** Create a dedicated `deckweaver` user. The process should never need root. Its home dir holds the model caches the sandbox allowlists.
5. **systemd hardening** (Linux). Example unit-file snippet:
   ```ini
   [Service]
   User=deckweaver
   Group=deckweaver
   NoNewPrivileges=true
   ProtectSystem=strict
   ProtectHome=read-only
   ReadWritePaths=/opt/Image2PPT/web/data /home/deckweaver
   PrivateTmp=true
   ProtectKernelTunables=true
   ProtectKernelModules=true
   ProtectControlGroups=true
   RestrictSUIDSGID=true
   LockPersonality=true
   MemoryDenyWriteExecute=false   # paddle/torch JITs need this off
   ```
6. **Install a sandbox helper.** On Linux: `apt install bubblewrap` (preferred) or `apt install firejail`. `auto` mode picks the best available. On macOS `sandbox-exec` is built in.
7. **Install CJK fonts that match Microsoft YaHei metrics (Linux).** The PPTX generator hard-codes `Microsoft YaHei` as the font name. The text-size and text-position calibration steps render the layout through LibreOffice; if YaHei is missing, LO falls back to a font with different character widths, calibration locks in wrong sizes, and the final PPTX overflows its boxes when opened on a machine that *does* have YaHei. Fix:
   ```bash
   sudo apt install fonts-wqy-microhei fonts-wqy-zenhei fonts-noto-cjk-extra
   mkdir -p ~/.config/fontconfig/conf.d
   cat > ~/.config/fontconfig/conf.d/30-yahei.conf <<'XML'
   <?xml version="1.0"?>
   <!DOCTYPE fontconfig SYSTEM "fonts.dtd">
   <fontconfig>
     <alias binding="strong">
       <family>Microsoft YaHei</family>
       <prefer>
         <family>WenQuanYi Micro Hei</family>
         <family>Noto Sans CJK SC</family>
       </prefer>
     </alias>
     <alias binding="strong">
       <family>微软雅黑</family>
       <prefer><family>WenQuanYi Micro Hei</family></prefer>
     </alias>
     <alias binding="strong">
       <family>PingFang SC</family>
       <prefer>
         <family>WenQuanYi Micro Hei</family>
         <family>Noto Sans CJK SC</family>
       </prefer>
     </alias>
   </fontconfig>
   XML
   fc-cache -fv
   fc-match "Microsoft YaHei"   # should resolve to wqy-microhei.ttc
   ```
   WenQuanYi Micro Hei is the open-source font designed to be metric-compatible with YaHei. If you need pixel-perfect fidelity and accept the EULA gray area, copy `msyh.ttc` / `msyhbd.ttc` from a Windows install into `~/.fonts/` instead.
8. **Rate limit at the proxy too.** The in-app login limiter is per-IP and in-memory; if you scale workers or want general API limits, do it in nginx (`limit_req_zone`).
9. **Restrict outbound network from the sandbox** once your model caches are warm: `DECKWEAVER_SANDBOX_ALLOW_NETWORK=false`. First-run model downloads happen during your manual smoke test, not during user traffic.
10. **Disk monitoring.** A long-running deploy will accumulate `web/data/outputs/`. The retention sweeper helps. Add a Prometheus / nagios / shell alert on free space.
11. **Backups.** Just `web/data/deckweaver.db` is enough (jobs are ephemeral, you don't need the outputs).
12. **No public signup.** The shipped API does not expose a `register` endpoint. Admin creates users via [`web/backend/manage.py`](backend/manage.py) or the admin-gated `POST /api/users`. If you need open signup, you write it; budget for email verification, captcha, and an abuse-handling story.
13. **Keep deps up to date** — `python3 -m pip list --outdated` and `npm audit` periodically. PaddleOCR, PyMuPDF, LibreOffice have had CVEs historically.
14. **Logs.** uvicorn access log captures IPs + paths. Rotate it (logrotate or journald). Don't log JWTs or upload contents.
15. **Image / PDF parsing risk.** The conversion subprocess is sandboxed, but the web process itself opens uploaded PDFs briefly to count pages for ETA (via PyMuPDF). This is a small attack surface — a malformed PDF could in theory exploit PyMuPDF in the web process. Mitigations: PyMuPDF only reads metadata for `page_count`, and the web user has no privileged access. For the truly paranoid, run the whole web service in a container or VM.

### Known limits — be honest

- **`sandbox-exec` is deprecated by Apple.** It still works on every shipping macOS as of 2026 but Apple has signaled they may remove it. For Linux production, use `bwrap`.
- **Sandboxing is process-level, not VM-level.** A kernel vulnerability defeats it. For truly hostile multi-tenant workloads, run each job in a microVM (Firecracker, Kata) — out of scope for this project.
- **First-run requires network.** The sandbox lets the subprocess talk to the internet by default so PaddleOCR can fetch its models on first run. Set `DECKWEAVER_SANDBOX_ALLOW_NETWORK=false` after caches are warm.
- **Single uvicorn worker assumed.** The in-memory rate limiter and job queue are per-process. If you scale to multiple workers, you need Redis (limiter) and a shared queue.
- **JWT is in `localStorage` on the frontend.** Vulnerable to XSS exfiltration if you ever introduce an XSS hole. The CSP is strict and React escapes by default, but if you customize the frontend, audit any `dangerouslySetInnerHTML`. Moving to `HttpOnly` cookies is a planned hardening.
- **No 2FA / OAuth out of the box.** For a small private deploy this is fine. For a public service with valuable accounts, add a SSO provider.

## Non-invasive guarantees

- All runtime state lives under `web/data/` (gitignored).
- The only change to existing CLI code is one optional flag on
  `scripts/convert.py` (`--mode`), which forwards to `build_deck.py`.
  Existing commands without the flag keep their previous behavior
  bit-for-bit.
- The web backend invokes the CLI exactly the way a human would —
  same args, same CWD (repo root). There's no shared in-process state
  between the web app and the pipeline.
- If you stop the web layer, the CLI keeps working. If you never start
  the web layer, the CLI never knows it exists.
