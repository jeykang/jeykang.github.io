#!/usr/bin/env python3
"""Verbose bootstrap script to create and configure Meilisearch index.

Adds structured JSON logging so CI/e2e runs can clearly surface the failing phase.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Logging helpers (single-line JSON to simplify parsing / debugging)
# ---------------------------------------------------------------------------
VERBOSE = os.environ.get("BOOTSTRAP_VERBOSE", "1") != "0"


def _log(level: str, msg: str, **fields: Any) -> None:
    if not VERBOSE and level == "debug":
        return
    rec: Dict[str, Any] = {"level": level, "msg": msg, "ts": round(time.time(), 3)}
    if fields:
        rec.update(fields)
    print(json.dumps(rec, separators=(",", ":")), flush=True)


def info(msg: str, **fields: Any) -> None:
    _log("info", msg, **fields)


def debug(msg: str, **fields: Any) -> None:
    _log("debug", msg, **fields)


def warn(msg: str, **fields: Any) -> None:
    _log("warn", msg, **fields)


def error(msg: str, **fields: Any) -> None:
    _log("error", msg, **fields)


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------
def make_request(
    url: str, method: str = "GET", data: dict | None = None, headers: dict | None = None
) -> dict:
    if headers is None:
        headers = {"Content-Type": "application/json"}

    master_key = os.environ.get("MEILI_MASTER_KEY")
    if master_key:
        headers["Authorization"] = f"Bearer {master_key}"

    req_data = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 204:
                return {}
            body = response.read().decode("utf-8")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                error("json_decode_error", url=url, body_preview=body[:200])
                raise
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        error(
            "http_error",
            url=url,
            code=e.code,
            reason=e.reason,
            body_preview=err_body[:500],
        )
        raise
    except Exception as e:
        error("request_exception", url=url, error=str(e))
        raise


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------
def wait_for_meilisearch(base_url: str, retries: int = 30) -> bool:
    info("waiting_for_meilisearch", base_url=base_url, retries=retries)
    for attempt in range(1, retries + 1):
        try:
            resp = make_request(f"{base_url}/health")
            if resp.get("status") == "available":
                info("meilisearch_ready")
                return True
        except Exception as e:  # noqa: BLE001
            debug("meilisearch_not_ready", attempt=attempt, error=str(e))
        time.sleep(2)
    return False


def wait_for_task(base_url: str, task_uid: int, timeout: int = 120) -> bool:
    start = time.time()
    debug("wait_for_task_start", task_uid=task_uid, timeout=timeout)
    while time.time() - start < timeout:
        try:
            task = make_request(f"{base_url}/tasks/{task_uid}")
            status = task.get("status")
            if status == "succeeded":
                debug("task_succeeded", task_uid=task_uid)
                return True
            if status == "failed":
                err = task.get("error") or {}
                # Accept index_already_exists as success for idempotency
                if isinstance(err, dict) and err.get("code") == "index_already_exists":
                    warn("task_failed_index_exists", task_uid=task_uid)
                    return True
                error("task_failed", task_uid=task_uid, error=err)
                return False
        except Exception as e:  # noqa: BLE001
            warn("task_poll_error", task_uid=task_uid, error=str(e))
        time.sleep(0.5)
    error("task_timeout", task_uid=task_uid, timeout=timeout)
    return False


def create_index(base_url: str) -> bool:
    info("create_index_start")
    try:
        # Fast path: already exists
        try:
            make_request(f"{base_url}/indexes/files")
            info("index_already_exists")
            return True
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        resp = make_request(
            f"{base_url}/indexes",
            method="POST",
            data={"uid": "files", "primaryKey": "id"},
        )
        task_uid = resp.get("taskUid")
        if task_uid is None:
            warn("index_create_missing_task_uid", raw=resp)
            return False
        if wait_for_task(base_url, int(task_uid)):
            info("index_create_ok", task_uid=task_uid)
            return True
        warn("index_create_task_failed", task_uid=task_uid)
    except urllib.error.HTTPError as e:
        if e.code == 409:
            info("index_conflict_already_exists")
            return True
        error("index_create_http_error", code=e.code, reason=e.reason)
    except Exception as e:  # noqa: BLE001
        error("index_create_exception", error=str(e))
    return False


def configure_index(base_url: str) -> bool:
    info("configure_index_start")
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
        "sortableAttributes": ["basename", "path", "size", "mtime", "seen_at"],
        "displayedAttributes": [
            "id",
            "root",
            "path",
            "basename",
            "ext",
            "dirpath",
            "size",
            "mtime",
            "uid",
            "gid",
            "mode",
        ],
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
        "pagination": {"maxTotalHits": 100000},
    }
    try:
        resp = make_request(
            f"{base_url}/indexes/files/settings", method="PATCH", data=settings
        )
        task_uid = resp.get("taskUid")
        if task_uid is None:
            warn("settings_missing_task_uid", raw=resp)
            return False
        if wait_for_task(base_url, int(task_uid), timeout=180):
            info("configure_index_ok", task_uid=task_uid)
            return True
        warn("configure_index_task_failed", task_uid=task_uid)
    except Exception as e:  # noqa: BLE001
        error("configure_index_exception", error=str(e))
    return False


def test_index(base_url: str) -> bool:
    info("test_index_start")
    doc = {
        "id": 1,
        "root": "test",
        "path": "/test/bootstrap-probe.txt",
        "basename": "bootstrap-probe.txt",
        "ext": "txt",
        "dirpath": "/test",
        "size": 1,
        "mtime": int(time.time()),
        "uid": 0,
        "gid": 0,
        "mode": 33188,
        "seen_at": int(time.time()),
    }
    try:
        resp = make_request(
            f"{base_url}/indexes/files/documents", method="POST", data=[doc]
        )
        task_uid = resp.get("taskUid")
        if task_uid is None or not wait_for_task(base_url, int(task_uid)):
            warn("test_index_add_failed", task_uid=task_uid)
            return False
        time.sleep(1)
        search = make_request(
            f"{base_url}/indexes/files/search", method="POST", data={"q": "bootstrap"}
        )
        hits = search.get("hits", [])
        if hits:
            info("test_index_search_ok", hits=len(hits))
            # cleanup
            del_resp = make_request(
                f"{base_url}/indexes/files/documents/1", method="DELETE"
            )
            wait_for_task(base_url, del_resp.get("taskUid", 0))
            return True
        warn("test_index_search_no_hits")
    except Exception as e:  # noqa: BLE001
        warn("test_index_exception", error=str(e))
    return False


def main() -> None:
    base_url = os.environ.get("MEILISEARCH_URL", "http://meilisearch:7700")
    info(
        "bootstrap_start",
        base_url=base_url,
        master_key_present=bool(os.environ.get("MEILI_MASTER_KEY")),
        verbose=VERBOSE,
    )
    try:
        if not wait_for_meilisearch(base_url):
            error("meilisearch_unavailable")
            sys.exit(1)
        if not create_index(base_url):
            error("create_index_failed")
            sys.exit(1)
        if not configure_index(base_url):
            error("configure_index_failed")
            sys.exit(1)
        if not test_index(base_url):
            warn("test_index_failed_continuing")
        info("bootstrap_complete")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        error("bootstrap_unhandled_exception", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
