"""Unit tests for the search API."""

import os
import sys
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app, SearchMode

client = TestClient(app)


class TestSearchAPI:
    """Test cases for Search API endpoints."""

    @patch("main.meili_session.get")
    def test_health_check_success(self, mock_get):
        """Test health check endpoint when Meilisearch is healthy."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "available"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["meilisearch"] == "available"

    @patch("main.meili_session.get")
    def test_health_check_failure(self, mock_get):
        """Test health check endpoint when Meilisearch is down."""
        mock_get.side_effect = Exception("Connection error")

        response = client.get("/health")
        assert response.status_code == 503
        assert "Service unhealthy" in response.json()["detail"]

    @patch("main.meili_session.post")
    def test_search_basic(self, mock_post):
        """Test basic search functionality."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "id": 1,
                    "path": "/test/file.txt",
                    "basename": "file.txt",
                    "ext": "txt",
                    "dirpath": "/test",
                    "size": 1024,
                    "mtime": 1700000000,
                }
            ],
            "estimatedTotalHits": 1,
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = client.get("/search?q=test")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test"
        assert data["mode"] == SearchMode.SUBSTR.value
        assert data["total"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["basename"] == "file.txt"

    @patch("main.meili_session.post")
    def test_search_with_filters(self, mock_post):
        """Test search with multiple filters."""
        mock_response = Mock()
        mock_response.json.return_value = {"hits": [], "estimatedTotalHits": 0}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = client.get(
            "/search?q=test&ext=txt&ext=pdf&dir=/docs&size_min=100&size_max=10000"
        )
        assert response.status_code == 200

        # Verify the filter was built correctly
        call_args = mock_post.call_args
        search_params = call_args[1]["json"]
        assert "filter" in search_params
        # Check that the filter contains expected elements
        filter_str = search_params["filter"]
        assert "ext" in filter_str
        assert "size >=" in filter_str

    @patch("main.meili_session.post")
    def test_search_regex_mode(self, mock_post):
        """Test regex search mode with post-filtering."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "basename": "test123.txt",
                    "path": "/test123.txt",
                    "ext": "txt",
                    "dirpath": "/",
                    "size": 100,
                    "mtime": 1700000000,
                },
                {
                    "basename": "test.txt",
                    "path": "/test.txt",
                    "ext": "txt",
                    "dirpath": "/",
                    "size": 100,
                    "mtime": 1700000000,
                },
                {
                    "basename": "file456.txt",
                    "path": "/file456.txt",
                    "ext": "txt",
                    "dirpath": "/",
                    "size": 100,
                    "mtime": 1700000000,
                },
            ],
            "estimatedTotalHits": 3,
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Search for files with numbers in name - URL encoded regex
        response = client.get("/search?q=test[0-9]%2B&mode=regex")
        assert response.status_code == 200
        data = response.json()
        # Regex filter should only match test123.txt
        assert data["total"] == 1
        assert data["results"][0]["basename"] == "test123.txt"

    @patch("main.meili_session.post")
    def test_search_pagination(self, mock_post):
        """Test search pagination."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "basename": f"file{i}.txt",
                    "path": f"/file{i}.txt",
                    "ext": "txt",
                    "dirpath": "/",
                    "size": 100,
                    "mtime": 1700000000,
                }
                for i in range(5)
            ],
            "estimatedTotalHits": 100,
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = client.get("/search?page=2&per_page=5")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["per_page"] == 5
        assert data["total_pages"] == 20  # 100 total / 5 per page

        # Check that offset was calculated correctly
        call_args = mock_post.call_args
        search_params = call_args[1]["json"]
        assert search_params["offset"] == 5  # (page 2 - 1) * 5

    @patch("main.meili_session.post")
    def test_search_sorting(self, mock_post):
        """Test different sort orders."""
        mock_response = Mock()
        mock_response.json.return_value = {"hits": [], "estimatedTotalHits": 0}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Test each sort order - use the actual enum values
        sort_tests = [
            ("mtime_desc", ["mtime:desc"]),
            ("size_asc", ["size:asc"]),
            ("path_desc", ["path:desc"]),
        ]

        for sort_order, expected_sort in sort_tests:
            response = client.get(f"/search?sort={sort_order}")
            assert response.status_code == 200, f"Failed for sort order: {sort_order}"

            call_args = mock_post.call_args
            search_params = call_args[1]["json"]
            assert search_params["sort"] == expected_sort

    @patch("main.meili_session.get")
    @patch("main.meili_session.post")
    def test_stats_endpoint(self, mock_post, mock_get):
        """Test statistics endpoint."""
        # Mock index stats
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "numberOfDocuments": 1000,
            "isIndexing": False,
            "fieldDistribution": {"ext": 100, "size": 100},
        }
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        # Mock search for last scan
        mock_post_response = Mock()
        mock_post_response.json.return_value = {"hits": [{"seen_at": 1700000000}]}
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response

        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 1000
        assert data["is_indexing"] is False
        assert data["last_scan"] == 1700000000
        assert "field_distribution" in data

    @patch("main.meili_session.post")
    def test_suggest_extensions(self, mock_post):
        """Test extension suggestions endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "facetDistribution": {"ext": {"txt": 500, "pdf": 300, "py": 200, "md": 100}}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = client.get("/suggest")
        assert response.status_code == 200
        data = response.json()
        assert "extensions" in data
        extensions = data["extensions"]
        assert len(extensions) == 4
        # Should be sorted by count descending
        assert extensions[0]["ext"] == "txt"
        assert extensions[0]["count"] == 500

    def test_search_invalid_page(self):
        """Test search with invalid page number."""
        response = client.get("/search?page=0")
        assert response.status_code == 422  # Validation error

    def test_search_excessive_page_size(self):
        """Test search with page size exceeding maximum."""
        response = client.get("/search?per_page=1000")
        assert response.status_code == 422  # Validation error

    @patch("main.meili_session.post")
    def test_search_empty_query(self, mock_post):
        """Test search without query (browse all)."""
        mock_response = Mock()
        mock_response.json.return_value = {"hits": [], "estimatedTotalHits": 0}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = client.get("/search")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == ""

        # Should still call Meilisearch with empty query
        call_args = mock_post.call_args
        search_params = call_args[1]["json"]
        assert search_params["q"] == ""

    def test_reindex_endpoint(self):
        """Test manual reindex trigger."""
        response = client.post("/reindex")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert "message" in data

    @patch("main.meili_session.post")
    def test_search_with_special_chars_in_query(self, mock_post):
        """Test search with special characters in query."""
        mock_response = Mock()
        mock_response.json.return_value = {"hits": [], "estimatedTotalHits": 0}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Test with various special characters
        special_queries = ["test's", "file-name", "path/to/file", "[test]", "(test)"]

        for query in special_queries:
            response = client.get(f"/search?q={query}")
            assert response.status_code == 200

    @patch("main.meili_session.post")
    def test_search_with_multiple_extensions(self, mock_post):
        """Test search with multiple extension filters."""
        mock_response = Mock()
        mock_response.json.return_value = {"hits": [], "estimatedTotalHits": 0}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = client.get("/search?ext=txt&ext=pdf&ext=doc")
        assert response.status_code == 200

        call_args = mock_post.call_args
        search_params = call_args[1]["json"]
        filter_str = search_params["filter"]
        # Should create an OR condition for extensions
        assert "ext = " in filter_str
        assert "OR" in filter_str

    @patch("main.meili_session.post")
    def test_search_error_handling(self, mock_post):
        """Test search error handling."""
        mock_post.side_effect = Exception("Meilisearch connection error")

        response = client.get("/search?q=test")
        assert response.status_code == 500
        assert "Search failed" in response.json()["detail"]

    @patch("main.meili_session.get")
    def test_stats_error_handling(self, mock_get):
        """Test stats endpoint error handling."""
        mock_get.side_effect = Exception("Connection error")

        response = client.get("/stats")
        assert response.status_code == 500
        assert "Failed to get stats" in response.json()["detail"]
