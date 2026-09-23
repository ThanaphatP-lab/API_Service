# Windows model stack commands

Run these commands from the `models` directory in Command Prompt or
PowerShell. The Windows entry point is `model-stack.cmd`; `model-stack.sh` is
for Linux/WSL only.

## First-time setup

Install Python 3.12 x64 from python.org, then create the isolated Paddle,
SigLIP, and API environments:

```cmd
scripts\setup-local.cmd all
scripts\model-stack.cmd check
```

## Start

```cmd
scripts\model-stack.cmd start core-stack
scripts\model-stack.cmd start table-v2-stack
scripts\model-stack.cmd start demo
```

`start` launches hidden background processes and stores validated Windows PID
files under `.run`. Closing Command Prompt does not stop those processes.

Use either `table-stack` or `table-v2-stack`. Starting both intentionally loads
duplicate table models into GPU memory.

## Operate

```cmd
scripts\model-stack.cmd status core-stack
scripts\model-stack.cmd readiness core-stack
scripts\model-stack.cmd logs gateway 200
scripts\model-stack.cmd restart gateway
scripts\model-stack.cmd stop core-stack
scripts\model-stack.cmd stop table-v2-stack
scripts\model-stack.cmd stop all
```

Windows output and error logs are stored separately:

```text
logs\gateway.windows.log
logs\gateway.windows.error.log
```

## Individual services

```cmd
scripts\model-stack.cmd start det-v5
scripts\model-stack.cmd start rec-th
scripts\model-stack.cmd start gateway
scripts\model-stack.cmd stop gateway
```

`run-service.cmd` remains available for foreground debugging:

```cmd
scripts\run-service.cmd table-v2
```
