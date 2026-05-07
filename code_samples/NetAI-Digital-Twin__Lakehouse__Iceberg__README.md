
> [!NOTE]
> Replace `localhost` with your machine's IP address if necessary.

# Version Check

### 🛠 Versions

| Software | Version | Repository / Base Image |
| --- | --- | --- |
| **Nessie** | `0.106.0` | `ghcr.io/projectnessie/nessie` |
| **MinIO** | `2025-09-07T16-13-09Z` | `quay.io/minio/minio` |
| **Apache Spark** | `3.5.5` | `tabulario/spark-iceberg` |
| **Scala** | `2.12.18` | `tabulario/spark-iceberg` |
| **Apache Iceberg** | `1.8.1` | `tabulario/spark-iceberg` |

---

### 🔍 How to Verify Versions

You can verify the installed software versions using the following commands:

1. **Project Nessie**

    Check the container logs to find the version and build information.
    ```bash
    docker logs [nessie_container_name]
    ```

2. **MinIO**

    Execute the version command inside the MinIO container to check the specific release tag.
    
    ```bash
    docker exec -it minio minio --version
    ```

3. **Apache Spark & Scala**

    Run the `spark-submit` command inside your Spark container to view both Spark and Scala versions.
    
    ```bash
    # Run inside the Spark container
    spark-submit --version
    ```

4. **Apache Iceberg**

    Since Iceberg is bundled as a library, check the version by inspecting the JAR filename in the `jars` directory.
    
    ```bash
    ls jars | grep iceberg
    # Example: iceberg-spark-runtime-3.5_2.12-1.8.1.jar
    # (3.5 = Spark, 2.12 = Scala, 1.8.1 = Iceberg version)\
    ```

---

# Start Iceberg

0. **Clone the repository**
    ```bash
    git clone <REPO>
    cd <REPO>
    ```

1.  **Create a `.env` file**
    ```bash
    cp example.env .env
    ```
      - Refer to the `example.env` file for configuration.
    > [!NOTE]
    > The `MINIO_ROOT_USER` and `AWS_ACCESS_KEY_ID`, as well as the `MINIO_ROOT_PASSWORD` and `AWS_SECRET_ACCESS_KEY` in the `.env` file, must match each other. 
    >
    > The USER must be at least 5 characters long, and the PASSWORD must be at least 8 characters long.

2.  **Grant execution permissions and run `start.sh`**

    ```bash
    chmod +x start.sh
    ./start.sh
    ```

3.  **Create a `spark1` bucket in MinIO** (Required for the Spark test script)
          
    You can create the required `spark1` bucket using either the Web UI or the Container CLI.
    
    #### **Option A: Via Web UI (Recommended for GUI users)**
    
    1. **Access the Console:** Open [http://localhost:9001](https://www.google.com/search?q=http://localhost:9001) in your browser.
    2. **Login:** Use the `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` defined in your `.env` file.
    3. **Create Bucket:** Click on **'Buckets'** -> **'Create Bucket'** and name it `spark1`.
    
    #### **Option B: Via Container CLI (Recommended for terminal users)**
    
    1. **Access the MinIO container:**
        ```bash
        docker exec -it minio bash
        
        ```
    
    
    2. **Configure alias and create the bucket:**
        ```bash
        # Use the credentials from your .env file
        mc alias set local http://localhost:9000 admin password
        mc mb local/spark1
        
        ```
    3. **Check created bucket**
       ```bash
       mc ls local
       ```

---

4.  **Access the `spark-iceberg` container**

    ```bash
    docker exec -it spark-iceberg bash
    ```

5.  **Run the environment setup test**

    ```bash
    spark-submit python-scripts/spark-iceberg-nessie_test.py
    ```

6.  **Monitor Spark Jobs**

      - While the Python code is running, check the Spark Web UI.
      - Access: http://localhost:4040 (or 4041)

7.  **Verify data in MinIO Console**

      - Access: http://localhost:9001
      - Check the `spark1` bucket to see if data has been created.

8.  **Verify results in Nessie Web UI**

      - Access: http://localhost:19120
      - Check if the `db` namespace/folder has been created.

> [!NOTE]
> The MinIO and Nessie data are stored in the `minio_data` and `nessie_data` folders within the directory where the Docker Compose command is executed, so the data is preserved even if the containers are removed and restarted.
