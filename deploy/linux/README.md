# Bare-metal Linux deployment (no Docker)

This deployment runs each API as a background Linux process. Python virtual
environments isolate dependencies; they are ordinary directories, not VMs.

On a multi-user server, each user/deployment must use a separate checkout such
as `/home/<user>/apps/model-api/models`. Do not create multiple environments in
one shared writable checkout. By default this script keeps venvs, model caches,
temporary files, PID files, and logs inside that deployment directory with
private permissions.

## 1. Host prerequisites

- Linux x86_64 and Python 3.10-3.12
- NVIDIA driver compatible with the configured GPU packages
- `curl`, `nohup`, Python `venv`, and required system libraries
- Network access on the first model load, or complete local weights/caches
- Only Nginx/managed ingress port 443 exposed publicly

Ubuntu 22.04 example (package names may differ on another distribution):

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl \
  libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 libgl1
nvidia-smi
```

## 2. Install Python environments

From the `models/` directory:

```bash
chmod +x scripts/setup-linux.sh scripts/model-stack.sh
./scripts/setup-linux.sh all
```

This creates `.venv-paddle`, `.venv-siglip`, and `.venv-api`. Keeping Paddle
and PyTorch separate avoids loading their CUDA/cuDNN userspace libraries into
the same Python process.

GPU wheels and NVIDIA runtime dependencies can total several gigabytes. Keep at
least 20-30 GB free per isolated deployment and run the setup inside `tmux` or
`screen` when the SSH connection may disconnect. The setup script uses extended
timeout/retry/resume settings. If a CDN download still fails, keep the existing
venv and retry only the failed group:

```bash
./scripts/setup-linux.sh paddle
./scripts/setup-linux.sh siglip
./scripts/setup-linux.sh api
```

Do not delete a partially installed environment solely because pip reports an
`incomplete-download`; pip can reuse completed cached artifacts and resume the
remaining download on the next attempt.

## 3. Configure secrets and runtime

```bash
cp .env.linux.example .env.runtime
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
chmod 600 .env.runtime
```

Put the two different generated values into `MODEL_GATEWAY_API_KEY` and
`INTERNAL_API_TOKEN`. Never commit `.env.runtime`. Set
`GATEWAY_ENABLED_PIPELINES` to the profile actually deployed, for example
`ocr-custom,text-det-v5,text-recognition`, or use `all` only when every
pipeline and direct leaf is running. A named stack
profile automatically overrides this value for the Gateway process it starts.

Validate the host and configuration:

```bash
./scripts/model-stack.sh check
```

Per-deployment runtime paths are:

```text
.venv-api/       API/Pipeline Python environment
.venv-paddle/    Paddle Python environment
.venv-siglip/    PyTorch/SigLIP Python environment
.cache/          PaddleX, Hugging Face, PyTorch and XDG caches
.tmp/            temporary uploaded/decoded images
.run/            PID files
logs/            service logs
weights/         optional local model exports
```

If two deployments run under the same Linux account, their separate checkout
paths still keep these files isolated. They must also use different port ranges
in each `.env.runtime` and must be assigned compatible GPU capacity/devices.

## 4. Start services

Start only Custom OCR and the Gateway:

```bash
./scripts/model-stack.sh start ocr-custom-stack
```

Other profiles:

```bash
./scripts/model-stack.sh start ocr-paddle-stack
./scripts/model-stack.sh start layout-stack
./scripts/model-stack.sh start table-stack
./scripts/model-stack.sh start table-v2-stack
./scripts/model-stack.sh start verification-stack
./scripts/model-stack.sh start all
```

The `all` profile intentionally excludes Streamlit. Start the Demo separately:

```bash
./scripts/model-stack.sh start demo
```

Processes ignore SSH hangup, write logs under `logs/`, and store validated PID
files under `.run/`.

## 5. Operate the stack

```bash
./scripts/model-stack.sh status all
./scripts/model-stack.sh readiness ocr-custom-stack
./scripts/model-stack.sh logs gateway 200
./scripts/model-stack.sh restart gateway
./scripts/model-stack.sh stop ocr-custom-stack
./scripts/model-stack.sh stop all
```

`stop` reads a service-specific PID file and verifies `/proc/<pid>/cmdline`
before sending SIGTERM. It refuses to kill a reused/unrelated PID and uses
SIGKILL only if the configured graceful timeout expires.

## 6. Public traffic

The scripts bind Gateway and every internal service to `127.0.0.1`. Backend on
the same host uses `http://127.0.0.1:8080`; public or cross-host traffic should
use Nginx/managed HTTPS ingress and the configuration in `deploy/nginx/`.

Do not expose ports 8001-8013 or 8080 to the Internet. If Backend is on another
host, use a private network/DNS or the HTTPS Gateway domain and firewall it to
the Backend source network.

## 7. Reboot and crash recovery

The script survives an SSH logout but does not automatically start processes
after a machine reboot and does not continuously restart a crashed process.
For production, wrap the same commands with systemd/Supervisor (or a managed
process service). Until that is configured, run `status` and `readiness` after
every deployment/reboot.
