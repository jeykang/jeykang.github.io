# pipx-runner

A minimal container that installs [`pipx`](https://pipx.pypa.io) and, at startup, installs any Python CLI tools you specify via environment variables. Installed console scripts are added to `PATH`, so you can invoke them directly with `docker exec`.

## Quick start

```bash
docker run --rm \
  -e PIPX_PACKAGES="httpie black==24.4.2" \
  ghcr.io/jeykang/pipx-runner:latest
```

`docker exec pipx-runner http --version`

## Environment variables

- `PIPX_PACKAGES` — Packages to install, whitespace-separated (e.g. `httpie black==24.4.2 'yt-dlp[rtmp]'`).

- `PIPX_PACKAGES_FILE` — File path (inside container) with newline-separated specs.

- `PIPX_FORCE` — If 1 (default), use `pipx install --force` to upgrade to the requested version.

- `PIPX_INSTALL_ARGS` — Extra arguments to pipx install (e.g., `--python python3.12`).

- `PIPX_PIP_ARGS` — Extra pip args used by pipx (e.g., `--index-url ...`).

- `PIPX_HOME` — Where pipx stores venvs (default `/opt/pipx`). If you want your venvs to persist, you can map `PIPX_HOME` to some location on your host machine.

- `PIPX_BIN_DIR` — Where console scripts go (default `/opt/pipx/bin`). The entrypoint will fall back to `~/.local/bin` if it cannot write here (e.g., when running as non-root).