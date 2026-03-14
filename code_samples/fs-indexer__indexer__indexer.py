#!/usr/bin/env python3
"""
Filesystem indexer for Meilisearch.
Scans directories, extracts metadata, and indexes files.
"""

import fnmatch
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterator, List, Dict, Any

import requests
import structlog
import xxhash
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@dataclass
class Config:
    """Indexer configuration from environment variables."""

    meilisearch_url: str
    master_key: str
    scan_roots: List[str]
    root_name: str
    excludes_file: str
    stability_sec: int
    batch_size: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            meilisearch_url=os.environ.get(
                "MEILISEARCH_URL", "http://meilisearch:7700"
            ),
            master_key=os.environ.get("MEILI_MASTER_KEY", ""),
            scan_roots=[
                p.strip() for p in os.environ.get("SCAN_ROOTS", "/data").split(",")
            ],
            root_name=os.environ.get("ROOT_NAME", "data"),
            excludes_file=os.environ.get("EXCLUDES_FILE", "/app/config/excludes.txt"),
            stability_sec=int(os.environ.get("STABILITY_SEC", "30")),
            batch_size=int(os.environ.get("BATCH_SIZE", "10000")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


class MeilisearchClient:
    """Simple Meilisearch client for file indexing."""

    def __init__(self, url: str, master_key: str = ""):
        self.url = url
        self.session = requests.Session()
        if master_key:
            self.session.headers["Authorization"] = f"Bearer {master_key}"
        self.session.headers["Content-Type"] = "application/json"
        self.index_name = "files"

    def wait_for_task(self, task_uid: int, timeout: int = 300) -> bool:
        """Wait for a task to complete."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = self.session.get(f"{self.url}/tasks/{task_uid}")
                response.raise_for_status()
                task = response.json()

                status = task.get("status")
                if status == "succeeded":
                    return True
                elif status == "failed":
                    logger.error(
                        "task_failed", task_uid=task_uid, error=task.get("error")
                    )
                    return False

                time.sleep(0.5)
            except Exception as e:
                logger.warning("task_check_error", task_uid=task_uid, error=str(e))
                time.sleep(1)

        logger.error("task_timeout", task_uid=task_uid, timeout=timeout)
        return False

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def add_documents(self, documents: List[Dict[str, Any]], wait: bool = False) -> int:
        """Add or update documents in the index."""
        response = self.session.post(
            f"{self.url}/indexes/{self.index_name}/documents", json=documents
        )
        response.raise_for_status()

        task_uid = response.json().get("taskUid")

        if wait:
            self.wait_for_task(task_uid)

        return task_uid

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def delete_documents(self, filter: str, wait: bool = False) -> int:
        """Delete documents matching a filter."""
        response = self.session.post(
            f"{self.url}/indexes/{self.index_name}/documents/delete",
            json={"filter": filter},
        )
        response.raise_for_status()

        task_uid = response.json().get("taskUid")

        if wait:
            self.wait_for_task(task_uid)

        return task_uid

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        response = self.session.get(f"{self.url}/indexes/{self.index_name}/stats")
        response.raise_for_status()
        return response.json()


class FileIndexer:
    """Main indexer class for scanning and indexing files."""

    def __init__(self, config: Config):
        self.config = config
        self.client = MeilisearchClient(config.meilisearch_url, config.master_key)
        self.excludes = self._load_excludes()
        self.stats = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "files_deleted": 0,
            "errors": 0,
            "start_time": time.time(),
        }
        self.pending_tasks = []

    def _load_excludes(self) -> List[str]:
        """Load exclusion patterns from file."""
        excludes = []
        try:
            if os.path.exists(self.config.excludes_file):
                with open(self.config.excludes_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            excludes.append(line)
                logger.info("loaded_excludes", count=len(excludes))
        except Exception as e:
            logger.error("failed_to_load_excludes", error=str(e))
        return excludes

    def _is_excluded(self, path: str) -> bool:
        """Check if path matches any exclusion pattern."""
        for pattern in self.excludes:
            if fnmatch.fnmatch(path, pattern):
                return True
            # Also check if any parent directory matches
            parts = path.split("/")
            for i in range(1, len(parts)):
                partial = "/".join(parts[:i]) + "/"
                if fnmatch.fnmatch(partial, pattern):
                    return True
        return False

    def _compute_file_id(self, dev: int, ino: int) -> int:
        """Compute unique file ID from device and inode."""
        # Meilisearch needs positive integers for IDs
        hash_val = xxhash.xxh64_intdigest(f"{dev}:{ino}")
        # Ensure it's positive and within JavaScript's safe integer range
        return hash_val & 0x7FFFFFFFFFFFFFFF

    def _scan_directory(self, root_dir: str, scan_id: int) -> Iterator[Dict[str, Any]]:
        """
        Recursively scan directory and yield file documents.
        Uses iterative approach with stack to avoid deep recursion.
        """
        if not os.path.exists(root_dir):
            logger.error("root_dir_not_found", path=root_dir)
            return

        stack = [root_dir]
        now = time.time()

        while stack:
            current_dir = stack.pop()

            # Check if directory is excluded
            rel_dir = os.path.relpath(current_dir, root_dir)
            if rel_dir != "." and self._is_excluded(rel_dir):
                logger.debug("excluded_dir", path=rel_dir)
                continue

            try:
                with os.scandir(current_dir) as entries:
                    for entry in entries:
                        try:
                            # Handle directories
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                                continue

                            # Handle files
                            if not entry.is_file(follow_symlinks=False):
                                continue

                            # Check exclusions
                            rel_path = os.path.relpath(entry.path, root_dir)
                            if self._is_excluded(rel_path):
                                self.stats["files_skipped"] += 1
                                continue

                            # Get file stats
                            stat = entry.stat(follow_symlinks=False)

                            # Skip recently modified files (stability window)
                            if now - stat.st_mtime < self.config.stability_sec:
                                logger.debug("skipped_recent_file", path=entry.path)
                                self.stats["files_skipped"] += 1
                                continue

                            # Extract metadata
                            doc_id = self._compute_file_id(stat.st_dev, stat.st_ino)
                            basename = entry.name
                            dirpath = os.path.dirname(entry.path)
                            ext = (
                                os.path.splitext(basename)[1][1:].lower()
                                if "." in basename
                                else ""
                            )

                            self.stats["files_scanned"] += 1

                            # Yield document for Meilisearch
                            yield {
                                "id": doc_id,
                                "root": self.config.root_name,
                                "path": entry.path,
                                "basename": basename,
                                "ext": ext,
                                "dirpath": dirpath,
                                "size": stat.st_size,
                                "mtime": int(stat.st_mtime),
                                "uid": stat.st_uid,
                                "gid": stat.st_gid,
                                "mode": stat.st_mode,
                                "seen_at": scan_id,
                            }

                        except (OSError, IOError) as e:
                            logger.warning(
                                "file_stat_error", path=entry.path, error=str(e)
                            )
                            self.stats["errors"] += 1
                            continue

            except (OSError, IOError) as e:
                logger.error("dir_scan_error", path=current_dir, error=str(e))
                self.stats["errors"] += 1
                continue

    def _index_batch(self, documents: List[Dict[str, Any]]) -> None:
        """Index a batch of documents."""
        if not documents:
            return

        try:
            task_uid = self.client.add_documents(documents, wait=False)
            self.pending_tasks.append(task_uid)
            self.stats["files_indexed"] += len(documents)
            logger.info("batch_submitted", count=len(documents), task_uid=task_uid)
        except Exception as e:
            logger.error("batch_index_failed", error=str(e), count=len(documents))
            self.stats["errors"] += 1

    def _wait_for_pending_tasks(self) -> None:
        """Wait for all pending indexing tasks to complete."""
        if not self.pending_tasks:
            return

        logger.info("waiting_for_tasks", count=len(self.pending_tasks))

        for task_uid in self.pending_tasks:
            if not self.client.wait_for_task(task_uid):
                logger.error("task_failed", task_uid=task_uid)
                self.stats["errors"] += 1

        self.pending_tasks.clear()

    def _sweep_deletions(self, scan_id: int) -> None:
        """Remove files not seen in current scan."""
        try:
            # Build filter for deletion
            # Meilisearch filter syntax: root = "value" AND seen_at < number
            filter_str = f'root = "{self.config.root_name}" AND seen_at < {scan_id}'

            logger.info("starting_deletion_sweep", filter=filter_str)

            task_uid = self.client.delete_documents(filter_str, wait=True)

            # Get stats to see how many were deleted
            # Note: Meilisearch doesn't return deletion count directly
            # You'd need to track this differently if exact count is needed

            logger.info("deletion_sweep_complete", task_uid=task_uid)

        except Exception as e:
            logger.error("deletion_sweep_failed", error=str(e))
            self.stats["errors"] += 1

    def run(self) -> None:
        """Main indexing loop."""
        scan_id = int(time.time())
        logger.info("scan_started", scan_id=scan_id, roots=self.config.scan_roots)

        batch = []

        for root_dir in self.config.scan_roots:
            logger.info("scanning_root", root=root_dir)

            for document in self._scan_directory(root_dir, scan_id):
                batch.append(document)

                if len(batch) >= self.config.batch_size:
                    self._index_batch(batch)
                    batch.clear()

        # Index remaining files
        if batch:
            self._index_batch(batch)

        # Wait for all indexing tasks to complete
        self._wait_for_pending_tasks()

        # Remove deleted files
        self._sweep_deletions(scan_id)

        # Get final stats from Meilisearch
        try:
            index_stats = self.client.get_stats()
            logger.info(
                "index_stats",
                total_documents=index_stats.get("numberOfDocuments"),
                is_indexing=index_stats.get("isIndexing"),
            )
        except Exception as e:
            logger.warning("failed_to_get_stats", error=str(e))

        # Log statistics
        elapsed = time.time() - self.stats["start_time"]
        logger.info(
            "scan_complete",
            scan_id=scan_id,
            duration_sec=round(elapsed, 2),
            files_scanned=self.stats["files_scanned"],
            files_indexed=self.stats["files_indexed"],
            files_skipped=self.stats["files_skipped"],
            errors=self.stats["errors"],
            files_per_sec=(
                round(self.stats["files_scanned"] / elapsed, 2) if elapsed > 0 else 0
            ),
        )


def main():
    """Main entry point."""
    config = Config.from_env()

    # Set log level
    import logging

    logging.basicConfig(level=getattr(logging, config.log_level))

    logger.info("indexer_starting", config=config.__dict__)

    indexer = FileIndexer(config)

    try:
        indexer.run()
    except KeyboardInterrupt:
        logger.info("indexer_interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error("indexer_failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
