# Build the image
docker compose build

# Run training on GPU 0
docker compose up train

# Quick test (5 epochs)
docker compose up train-quick

# Run benchmark for comparison
docker compose up benchmark

# Use specific GPU
GPU_ID=1 docker compose up train

# Use multiple GPUs
docker compose up train-multi

# Use auto batch sizing (adapts to any GPU)
docker compose up train-auto

# Or with environment variables
TARGET_MEMORY=0.75 MAX_BATCH=128 docker compose up train-auto

# Or using the helper script
./docker/run.sh train-auto