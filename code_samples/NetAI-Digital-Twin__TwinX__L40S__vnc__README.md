This project utilizes Docker Compose to quickly deploy multiple VNC desktop environments with enabled NVIDIA GPU access.

## 🚀 Project Execution and Setup

### 1\. Prepare the Environment Configuration File (`.env`)

Create and use a **`.env`** file by referencing the `example.env` file.

```bash
cp example.env .env
```

### 2\. Run Containers

Deploy the VNC servers by running Docker Compose in detached (background) mode.

```bash
docker compose up -d
```

-----

## 🌐 VNC Connection Information (Web Browser)

The VNC servers are accessible via a web browser using NoVNC. `[ServerIP]` is the IP address of the host server where the containers are running.

| User | Port | Connection Address (Web Browser) |
| :---: | :---: | :--- |
| **User 1** | `6910` | `http://[ServerIP]:6910` |
| **User 2** | `6911` | `http://[ServerIP]:6911` |
| **User 3** | `6912` | `http://[ServerIP]:6912` |
| **User 4** | `6913` | `http://[ServerIP]:6913` |

-----

## 🔑 Container Internal Login Details

Default account information for accessing the terminal (shell) inside each VNC container.

  * **VNC Container Internal Terminal Default Account:** `headless`
  * **VNC Container Internal Terminal Default Password:** `headless`

-----

## ⚙️ VNC Environment

  * **Docker Access:** Within each VNC container, the Docker-in-Docker environment is enabled
  * **GPU Access:** Allowing for **NVIDIA GPU usage and management** via the `docker` command and `nvidia-smi`.
  * **Automatic Restart:** Containers are configured with `restart: unless-stopped`, ensuring they automatically restart after a host reboot or if they crash.
