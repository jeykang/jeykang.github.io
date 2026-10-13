# Filesystem Indexer

[![CI](https://github.com/jeykang/fs-indexer/actions/workflows/ci.yml/badge.svg)](https://github.com/jeykang/fs-indexer/actions/workflows/ci.yml)
[![Release](https://github.com/jeykang/fs-indexer/actions/workflows/release.yml/badge.svg)](https://github.com/jeykang/fs-indexer/actions/workflows/release.yml)
[![Security Audit](https://github.com/jeykang/fs-indexer/actions/workflows/security.yml/badge.svg)](https://github.com/jeykang/fs-indexer/actions/workflows/security.yml)

A high-performance, dockerized filesystem indexing and search system using Meilisearch.

## Features

- 🔍 **Multiple search modes**: Substring, regex (RE2), and plain text search
- 📁 **Metadata-only indexing**: Fast, lightweight indexing without file content
- ⏰ **Automatic hourly rescans**: Keep your index up-to-date
- 🚀 **High performance**: Scales to millions of files
- 🎯 **Advanced filtering**: By extension, directory, size, and modification time
- 📊 **Web UI**: Clean, responsive interface with real-time search
- 🐳 **Fully containerized**: Easy deployment with Docker Compose
- 🔒 **Production ready**: Health checks, logging, error handling

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/jeykang/fs-indexer.git
cd fs-indexer
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env to set your HOST_PATH
```

3. Start the system:
```bash
make deploy
```

4. Access the web interface:
- Web UI: http://localhost:8081
- API: http://localhost:8080
- Meilisearch: http://localhost:7700

## Using Pre-built Images

Pull and run pre-built images from Docker Hub:

```bash
# Download the release compose file
wget https://github.com/jeykang/fs-indexer/releases/latest/download/docker-compose.release.yml

# Start with pre-built images
docker compose -f docker-compose.release.yml up -d
```

## Development

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Make

### Building

```bash
make build
```

### Testing

```bash
# Run all tests
make test

# Run specific test suites
make test-unit
make test-integration
make test-e2e
```

### Linting

```bash
make lint
make format
```

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST_PATH` | `/home` | Host filesystem path to index |
| `SCAN_ROOTS` | `/data` | Container mount points to scan |
| `BATCH_SIZE` | `2000` | Number of files to index per batch |
| `STABILITY_SEC` | `30` | Skip files modified within N seconds |
| `DEFAULT_PAGE_SIZE` | `50` | Default search results per page |

See `.env.example` for full configuration options.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Web UI    │────▶│  Search API │───▶│ Meilisearch │
└─────────────┘     └─────────────┘     └─────────────┘
                                              ▲
                    ┌─────────────┐           │
                    │  Scheduler  │───────────┤
                    └─────────────┘           │
                           │                  │
                    ┌─────────────┐           │
                    │   Indexer   │───────────┘
                    └─────────────┘
```

## Performance

- **Storage**: ~1-2 KB per file (metadata only)
- **Memory**: 4-8 GB RAM for 10M+ files
- **Speed**: Sub-second searches on millions of files
- **Throughput**: 10,000+ files/second indexing

## Security

- Mount source directories as read-only
- Add authentication to web UI (nginx basic auth)
- Use HTTPS with proper certificates
- Regular security scans with Trivy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Support

- 📖 [Documentation](https://github.com/jeykang/fs-indexer/wiki)
- 🐛 [Issue Tracker](https://github.com/jeykang/fs-indexer/issues)
- 💬 [Discussions](https://github.com/jeykang/fs-indexer/discussions)