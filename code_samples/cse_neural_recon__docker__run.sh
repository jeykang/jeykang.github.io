#!/bin/bash
# Helper script for Docker-based training
# Usage: ./docker/run.sh [command] [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_help() {
    echo "Neural 3D Reconstruction - Docker Training Helper"
    echo ""
    echo "Usage: ./docker/run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  build              Build the Docker image"
    echo "  train              Run full training (100 epochs)"
    echo "  train-auto         Run training with auto batch size scaling"
    echo "  train-quick [N]    Run quick training (default: 5 epochs)"
    echo "  benchmark [N]      Run productivity benchmark (default: 2 epochs)"
    echo "  compare            Compare metrics across devices"
    echo "  shell              Start interactive shell with GPU"
    echo "  logs               Show training logs"
    echo "  stop               Stop all running containers"
    echo "  clean              Remove containers and cached images"
    echo ""
    echo "Environment Variables:"
    echo "  CUDA_VISIBLE_DEVICES=0,1   Select GPUs (default: all)"
    echo "  EPOCHS=10                   Override epoch count"
    echo "  TARGET_MEMORY=0.80         GPU memory fraction for auto scaling"
    echo "  MAX_BATCH=64               Maximum batch size for auto scaling"
    echo ""
    echo "Examples:"
    echo "  ./docker/run.sh build"
    echo "  ./docker/run.sh train"
    echo "  ./docker/run.sh train-auto    # Auto-scales batch size to GPU"
    echo "  CUDA_VISIBLE_DEVICES=0 ./docker/run.sh train"
    echo "  ./docker/run.sh train-quick 10"
    echo "  ./docker/run.sh benchmark 5"
}

check_nvidia_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime may not be configured${NC}"
        echo "Make sure nvidia-container-toolkit is installed"
    fi
}

case "${1:-help}" in
    build)
        echo -e "${GREEN}Building Docker image...${NC}"
        docker compose build
        ;;
    
    train)
        check_nvidia_docker
        echo -e "${GREEN}Starting full training...${NC}"
        docker compose up train
        ;;
    
    train-auto)
        check_nvidia_docker
        echo -e "${GREEN}Starting training with auto batch size scaling...${NC}"
        echo -e "${YELLOW}Target memory: ${TARGET_MEMORY:-80}% | Max batch: ${MAX_BATCH:-64}${NC}"
        docker compose up train-auto
        ;;
    
    train-quick)
        check_nvidia_docker
        EPOCHS="${2:-5}"
        echo -e "${GREEN}Starting quick training (${EPOCHS} epochs)...${NC}"
        EPOCHS=$EPOCHS docker compose up train-quick
        ;;
    
    benchmark)
        check_nvidia_docker
        BENCHMARK_EPOCHS="${2:-2}"
        echo -e "${GREEN}Running benchmark (${BENCHMARK_EPOCHS} epochs)...${NC}"
        BENCHMARK_EPOCHS=$BENCHMARK_EPOCHS docker compose up benchmark
        ;;
    
    compare)
        echo -e "${GREEN}Comparing device metrics...${NC}"
        docker compose run --rm compare
        ;;
    
    shell)
        check_nvidia_docker
        echo -e "${GREEN}Starting interactive shell...${NC}"
        docker compose run --rm shell
        ;;
    
    logs)
        docker compose logs -f train
        ;;
    
    stop)
        echo -e "${YELLOW}Stopping all containers...${NC}"
        docker compose down
        ;;
    
    clean)
        echo -e "${YELLOW}Cleaning up containers and cache...${NC}"
        docker compose down -v --rmi local
        ;;
    
    help|--help|-h)
        print_help
        ;;
    
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        print_help
        exit 1
        ;;
esac
