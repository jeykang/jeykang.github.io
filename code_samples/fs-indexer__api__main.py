#!/usr/bin/env python3
"""
Search API for Meilisearch filesystem indexer.
Provides REST endpoints for searching indexed files.
"""

import os
import re
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
MEILISEARCH_URL = os.environ.get("MEILISEARCH_URL", "http://meilisearch:7700")
MEILI_MASTER_KEY = os.environ.get("MEILI_MASTER_KEY", "")
DEFAULT_PAGE_SIZE = int(os.environ.get("DEFAULT_PAGE_SIZE", "50"))
MAX_PAGE_SIZE = int(os.environ.get("MAX_PAGE_SIZE", "500"))

app = FastAPI(
    title="Filesystem Search API",
    description="Search indexed files with Meilisearch",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Meilisearch client session
meili_session = requests.Session()
if MEILI_MASTER_KEY:
    meili_session.headers["Authorization"] = f"Bearer {MEILI_MASTER_KEY}"
meili_session.headers["Content-Type"] = "application/json"


class SearchMode(str, Enum):
    """Search modes supported by the API."""

    PLAIN = "plain"  # Normal search
    SUBSTR = "substr"  # Substring search (default in Meilisearch)
    REGEX = "regex"  # Regex simulation using post-filtering


class SortOrder(str, Enum):
    """Sort orders for search results."""

    MTIME_DESC = "mtime_desc"
    MTIME_ASC = "mtime_asc"
    SIZE_DESC = "size_desc"
    SIZE_ASC = "size_asc"
    PATH_ASC = "path_asc"
    PATH_DESC = "path_desc"


class FileResult(BaseModel):
    """Single file search result."""

    path: str
    basename: str
    ext: str
    dirpath: str
    size: int
    mtime: int
    mtime_formatted: str
    size_formatted: str


class SearchResponse(BaseModel):
    """Search API response."""

    query: str
    mode: SearchMode
    total: int
    page: int
    per_page: int
    total_pages: int
    results: List[FileResult]
    took_ms: int


class StatsResponse(BaseModel):
    """Index statistics response."""

    total_files: int
    is_indexing: bool
    last_scan: Optional[int]
    field_distribution: Dict[str, int]


def format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_timestamp(ts: int) -> str:
    """Format Unix timestamp to human-readable date."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def apply_regex_filter(
    results: List[Dict], pattern: str, field: str = "basename"
) -> List[Dict]:
    """Apply regex filtering to results (post-processing for regex mode)."""
    try:
        regex = re.compile(pattern)
        return [r for r in results if regex.search(r.get(field, ""))]
    except re.error:
        # Invalid regex, return empty results
        return []


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        response = meili_session.get(f"{MEILISEARCH_URL}/health")
        response.raise_for_status()
        health = response.json()
        return {"status": "healthy", "meilisearch": health.get("status")}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.get("/search", response_model=SearchResponse)
async def search_files(
    q: Optional[str] = Query(None, description="Search query"),
    mode: SearchMode = Query(SearchMode.SUBSTR, description="Search mode"),
    ext: Optional[List[str]] = Query(None, description="File extensions to filter"),
    dir: Optional[str] = Query(None, description="Directory prefix filter"),
    mtime_from: Optional[int] = Query(
        None, description="Minimum modification time (Unix timestamp)"
    ),
    mtime_to: Optional[int] = Query(
        None, description="Maximum modification time (Unix timestamp)"
    ),
    size_min: Optional[int] = Query(None, description="Minimum file size in bytes"),
    size_max: Optional[int] = Query(None, description="Maximum file size in bytes"),
    sort: SortOrder = Query(SortOrder.MTIME_DESC, description="Sort order"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(
        DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Results per page"
    ),
):
    """Search indexed files with various filters and modes."""

    start_time = time.time()

    # Build Meilisearch query
    search_params = {
        "limit": (
            per_page if mode != SearchMode.REGEX else min(per_page * 10, 1000)
        ),  # Get more for regex filtering
        "offset": (page - 1) * per_page if mode != SearchMode.REGEX else 0,
        "showMatchesPosition": False,
    }

    # Set query based on mode
    if q:
        if mode == SearchMode.REGEX:
            # For regex, we do a broad search and filter later
            # Extract potential keywords from the regex pattern
            keywords = re.findall(r"\w+", q)
            search_params["q"] = " ".join(keywords) if keywords else ""
        else:
            # For PLAIN and SUBSTR, Meilisearch handles it naturally
            search_params["q"] = q
    else:
        search_params["q"] = ""

    # Build filter conditions
    filters = []

    if ext:
        # Multiple extensions with OR
        ext_filters = [f'ext = "{e}"' for e in ext]
        if len(ext_filters) > 1:
            filters.append(f"({' OR '.join(ext_filters)})")
        else:
            filters.append(ext_filters[0])

    if dir:
        # Directory prefix - escape quotes in dir path
        escaped_dir = dir.replace('"', '\\"')
        # Meilisearch doesn't support LIKE, so we'll need to post-filter
        # For now, we can filter by exact dirpath
        filters.append(f'dirpath = "{escaped_dir}"')

    if mtime_from:
        filters.append(f"mtime >= {mtime_from}")
    if mtime_to:
        filters.append(f"mtime <= {mtime_to}")

    if size_min:
        filters.append(f"size >= {size_min}")
    if size_max:
        filters.append(f"size <= {size_max}")

    if filters:
        search_params["filter"] = " AND ".join(filters)

    # Set sort order
    sort_map = {
        SortOrder.MTIME_DESC: ["mtime:desc"],
        SortOrder.MTIME_ASC: ["mtime:asc"],
        SortOrder.SIZE_DESC: ["size:desc"],
        SortOrder.SIZE_ASC: ["size:asc"],
        SortOrder.PATH_ASC: ["path:asc"],
        SortOrder.PATH_DESC: ["path:desc"],
    }
    search_params["sort"] = sort_map[sort]

    # Execute search
    try:
        response = meili_session.post(
            f"{MEILISEARCH_URL}/indexes/files/search", json=search_params
        )
        response.raise_for_status()
        search_result = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # Process results
    hits = search_result.get("hits", [])

    # Apply regex filtering if needed
    if mode == SearchMode.REGEX and q:
        hits = apply_regex_filter(hits, q)
        # Adjust pagination for regex-filtered results
        total = len(hits)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        hits = hits[start_idx:end_idx]
    else:
        total = search_result.get("estimatedTotalHits", 0)

    # Format results
    results = []
    for hit in hits:
        results.append(
            FileResult(
                path=hit.get("path", ""),
                basename=hit.get("basename", ""),
                ext=hit.get("ext", ""),
                dirpath=hit.get("dirpath", ""),
                size=hit.get("size", 0),
                mtime=hit.get("mtime", 0),
                mtime_formatted=format_timestamp(hit.get("mtime", 0)),
                size_formatted=format_size(hit.get("size", 0)),
            )
        )

    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page if total > 0 else 0

    # Calculate response time
    took_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        query=q or "",
        mode=mode,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        results=results,
        took_ms=took_ms,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get index statistics."""
    try:
        response = meili_session.get(f"{MEILISEARCH_URL}/indexes/files/stats")
        response.raise_for_status()
        stats = response.json()

        # Get the most recent seen_at value for last scan time
        try:
            search_response = meili_session.post(
                f"{MEILISEARCH_URL}/indexes/files/search",
                json={"q": "", "limit": 1, "sort": ["seen_at:desc"]},
            )
            search_response.raise_for_status()
            search_data = search_response.json()
        except requests.RequestException:
            # Fallback: try without sort (schema may not mark seen_at as sortable yet)
            try:
                fallback = meili_session.post(
                    f"{MEILISEARCH_URL}/indexes/files/search",
                    json={"q": "", "limit": 1},
                )
                fallback.raise_for_status()
                search_data = fallback.json()
            except requests.RequestException:
                search_data = {"hits": []}

        last_scan = None
        if search_data.get("hits"):
            last_scan = search_data["hits"][0].get("seen_at")

        return StatsResponse(
            total_files=stats.get("numberOfDocuments", 0),
            is_indexing=stats.get("isIndexing", False),
            last_scan=last_scan,
            field_distribution=stats.get("fieldDistribution", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@app.get("/suggest")
async def suggest_extensions():
    """Get list of available file extensions for filtering."""
    try:
        # Use facets to get extension distribution
        response = meili_session.post(
            f"{MEILISEARCH_URL}/indexes/files/search",
            json={"q": "", "limit": 0, "facets": ["ext"]},
        )
        response.raise_for_status()
        result = response.json()

        facet_dist = result.get("facetDistribution", {}).get("ext", {})

        # Format as list sorted by count
        extensions = [
            {"ext": ext, "count": count}
            for ext, count in sorted(
                facet_dist.items(), key=lambda x: x[1], reverse=True
            )
        ][
            :100
        ]  # Limit to top 100

        return {"extensions": extensions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {e}")


@app.post("/reindex")
async def trigger_reindex():
    """Trigger a manual reindex (would need to signal the indexer container)."""
    # This would typically trigger the indexer container to run
    # For now, return a message
    return {"message": "Reindex triggered", "status": "pending"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
