#!/usr/bin/env python3
"""
Meilisearch operations helper script.
Useful commands for managing the Meilisearch index.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any

import requests


class MeilisearchOps:
    """Helper class for Meilisearch operations."""

    def __init__(self, url: str = None, master_key: str = None):
        self.url = url or os.environ.get("MEILISEARCH_URL", "http://localhost:7700")
        self.master_key = master_key or os.environ.get("MEILI_MASTER_KEY", "")
        self.session = requests.Session()
        if self.master_key:
            self.session.headers["Authorization"] = f"Bearer {self.master_key}"
        self.session.headers["Content-Type"] = "application/json"

    def _wait_for_task(self, task_uid: int, timeout: int = 60) -> bool:
        """Wait for a task to complete."""
        start = time.time()
        while time.time() - start < timeout:
            response = self.session.get(f"{self.url}/tasks/{task_uid}")
            if response.status_code == 200:
                task = response.json()
                status = task.get("status")
                if status == "succeeded":
                    return True
                elif status == "failed":
                    print(f"Task {task_uid} failed: {task.get('error')}")
                    return False
            time.sleep(0.5)
        print(f"Task {task_uid} timed out")
        return False

    def health_check(self) -> Dict[str, Any]:
        """Check Meilisearch health."""
        response = self.session.get(f"{self.url}/health")
        return response.json()

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        response = self.session.get(f"{self.url}/indexes/files/stats")
        if response.status_code == 404:
            return {"error": "Index 'files' not found"}
        return response.json()

    def get_settings(self) -> Dict[str, Any]:
        """Get index settings."""
        response = self.session.get(f"{self.url}/indexes/files/settings")
        if response.status_code == 404:
            return {"error": "Index 'files' not found"}
        return response.json()

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """Update index settings."""
        response = self.session.patch(
            f"{self.url}/indexes/files/settings", json=settings
        )
        if response.status_code in [200, 202]:
            task_uid = response.json().get("taskUid")
            return self._wait_for_task(task_uid)
        return False

    def create_dump(self) -> str:
        """Create a database dump."""
        response = self.session.post(f"{self.url}/dumps")
        if response.status_code == 202:
            task_uid = response.json().get("taskUid")
            if self._wait_for_task(task_uid, timeout=300):
                return f"Dump created successfully (task {task_uid})"
        return "Failed to create dump"

    def list_dumps(self) -> list:
        """List available dumps."""
        # Note: Meilisearch doesn't have a direct API for listing dumps
        # This would need to check the filesystem or use tasks API
        return ["Check the /dumps directory in the Meilisearch container"]

    def delete_all_documents(self) -> bool:
        """Delete all documents from the index."""
        response = self.session.delete(f"{self.url}/indexes/files/documents")
        if response.status_code == 202:
            task_uid = response.json().get("taskUid")
            return self._wait_for_task(task_uid)
        return False

    def search_sample(self, query: str = "", limit: int = 10) -> Dict[str, Any]:
        """Perform a sample search."""
        response = self.session.post(
            f"{self.url}/indexes/files/search", json={"q": query, "limit": limit}
        )
        if response.status_code == 404:
            return {"error": "Index 'files' not found"}
        return response.json()

    def get_tasks(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent tasks."""
        response = self.session.get(f"{self.url}/tasks?limit={limit}")
        return response.json()

    def optimize_index(self) -> str:
        """Optimize the index (not needed in Meilisearch but included for reference)."""
        return "Meilisearch optimizes indexes automatically. No manual optimization needed."

    def reset_index(self) -> bool:
        """Reset the index (delete and recreate)."""
        # Delete index
        delete_response = self.session.delete(f"{self.url}/indexes/files")
        if delete_response.status_code == 202:
            task_uid = delete_response.json().get("taskUid")
            if not self._wait_for_task(task_uid):
                return False

        # Recreate index
        create_response = self.session.post(
            f"{self.url}/indexes", json={"uid": "files", "primaryKey": "id"}
        )
        if create_response.status_code == 202:
            task_uid = create_response.json().get("taskUid")
            if not self._wait_for_task(task_uid):
                return False

        # Configure settings
        settings = {
            "searchableAttributes": ["basename", "path"],
            "filterableAttributes": [
                "root",
                "ext",
                "dirpath",
                "size",
                "mtime",
                "uid",
                "gid",
                "mode",
                "seen_at",
            ],
            "sortableAttributes": ["basename", "path", "size", "mtime"],
            "rankingRules": [
                "words",
                "typo",
                "proximity",
                "attribute",
                "sort",
                "exactness",
            ],
            "typoTolerance": {
                "enabled": True,
                "minWordSizeForTypos": {"oneTypo": 4, "twoTypos": 8},
            },
        }
        return self.update_settings(settings)

    def export_documents(self, limit: int = 1000, offset: int = 0) -> list:
        """Export documents from the index."""
        response = self.session.get(
            f"{self.url}/indexes/files/documents?limit={limit}&offset={offset}"
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Meilisearch operations helper")
    parser.add_argument(
        "--url",
        help="Meilisearch URL",
        default=os.environ.get("MEILISEARCH_URL", "http://localhost:7700"),
    )
    parser.add_argument(
        "--key", help="Master key", default=os.environ.get("MEILI_MASTER_KEY", "")
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add subcommands
    subparsers.add_parser("health", help="Check health")
    subparsers.add_parser("stats", help="Get index statistics")
    subparsers.add_parser("settings", help="Get index settings")
    subparsers.add_parser("dump", help="Create database dump")
    subparsers.add_parser("reset", help="Reset index")
    subparsers.add_parser("tasks", help="List recent tasks")
    subparsers.add_parser("optimize", help="Optimize index (info only)")

    search_parser = subparsers.add_parser("search", help="Sample search")
    search_parser.add_argument("query", nargs="?", default="", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Result limit")

    export_parser = subparsers.add_parser("export", help="Export documents")
    export_parser.add_argument(
        "--limit", type=int, default=1000, help="Number of documents"
    )
    export_parser.add_argument("--offset", type=int, default=0, help="Offset")
    export_parser.add_argument("--output", help="Output file (JSON)")

    delete_parser = subparsers.add_parser("delete-all", help="Delete all documents")
    delete_parser.add_argument(
        "--confirm", action="store_true", help="Confirm deletion"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    ops = MeilisearchOps(args.url, args.key)

    try:
        if args.command == "health":
            result = ops.health_check()
            print(json.dumps(result, indent=2))

        elif args.command == "stats":
            result = ops.get_stats()
            if "error" not in result:
                print(f"Documents: {result.get('numberOfDocuments', 0)}")
                print(f"Is Indexing: {result.get('isIndexing', False)}")
                if "fieldDistribution" in result:
                    print("\nField Distribution:")
                    for field, count in result["fieldDistribution"].items():
                        print(f"  {field}: {count}")
            else:
                print(result["error"])

        elif args.command == "settings":
            result = ops.get_settings()
            print(json.dumps(result, indent=2))

        elif args.command == "dump":
            result = ops.create_dump()
            print(result)

        elif args.command == "reset":
            print("Resetting index...")
            if ops.reset_index():
                print("Index reset successfully")
            else:
                print("Failed to reset index")

        elif args.command == "tasks":
            result = ops.get_tasks()
            print(f"Recent tasks ({result.get('total', 0)} total):")
            for task in result.get("results", [])[:10]:
                print(
                    f"  [{task['uid']}] {task['type']} - {task['status']} "
                    f"({task.get('duration', 'N/A')})"
                )

        elif args.command == "optimize":
            print(ops.optimize_index())

        elif args.command == "search":
            result = ops.search_sample(args.query, args.limit)
            if "error" not in result:
                print(f"Found {result.get('estimatedTotalHits', 0)} results")
                print(f"Query took {result.get('processingTimeMs', 0)}ms")
                print("\nResults:")
                for hit in result.get("hits", []):
                    print(f"  - {hit.get('path', 'N/A')}")
            else:
                print(result["error"])

        elif args.command == "export":
            documents = ops.export_documents(args.limit, args.offset)
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(documents, f, indent=2)
                print(f"Exported {len(documents)} documents to {args.output}")
            else:
                print(json.dumps(documents, indent=2))

        elif args.command == "delete-all":
            if not args.confirm:
                print("This will delete ALL documents from the index!")
                print("Use --confirm to proceed")
                sys.exit(1)
            if ops.delete_all_documents():
                print("All documents deleted successfully")
            else:
                print("Failed to delete documents")

    except requests.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
