"""Unit tests for the search API."""

import os
import sys

# Import the API module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import SearchMode, SortOrder, app, build_search_query, escape_sql, format_size

from unittest.mock import patch

from faker import Faker
from fastapi.testclient import TestClient

fake = Faker()
client = TestClient(app)


class TestAPI:
    """Test cases for Search API."""

    def test_health_check(self):
        """Test health check endpoint."""
        with patch("main.execute_sql") as mock_sql:
            mock_sql.return_value = {"data": []}
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    def test_escape_sql(self):
        """Test SQL escaping."""
        assert escape_sql("test") == "test"
        assert escape_sql("test's") == "test''s"
        assert escape_sql("test\\path") == "test\\\\path"
        assert escape_sql("'; DROP TABLE--") == "''; DROP TABLE--"

    def test_format_size(self):
        """Test size formatting."""
        assert format_size(100) == "100.0 B"
        assert format_size(1024) == "1.0 KB"
        assert format_size(1024 * 1024) == "1.0 MB"
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_build_search_query_basic(self):
        """Test basic query building."""
        search_query, count_query = build_search_query(
            q="test",
            mode=SearchMode.SUBSTR,
            ext=None,
            dir=None,
            mtime_from=None,
            mtime_to=None,
            size_min=None,
            size_max=None,
            sort=SortOrder.MTIME_DESC,
            page=1,
            per_page=50,
        )

        assert "MATCH('@basename *test*')" in search_query
        assert "ORDER BY mtime DESC" in search_query
        assert "LIMIT 50 OFFSET 0" in search_query
        assert "COUNT(*)" in count_query

    def test_build_search_query_with_filters(self):
        """Test query building with filters."""
        search_query, count_query = build_search_query(
            q="doc",
            mode=SearchMode.REGEX,
            ext=["pdf", "docx"],
            dir="/home/user",
            mtime_from=1000000,
            mtime_to=2000000,
            size_min=1024,
            size_max=1048576,
            sort=SortOrder.SIZE_DESC,
            page=2,
            per_page=25,
        )

        assert "REGEX(basename, 'doc')" in search_query
        assert "ext IN ('pdf','docx')" in search_query
        assert "dirpath LIKE '/home/user%'" in search_query
        assert "mtime >= 1000000" in search_query
        assert "mtime <= 2000000" in search_query
        assert "size >= 1024" in search_query
        assert "size <= 1048576" in search_query
        assert "ORDER BY size DESC" in search_query
        assert "LIMIT 25 OFFSET 25" in search_query

    @patch("main.execute_sql")
    def test_search_endpoint(self, mock_sql):
        """Test search endpoint."""
        mock_sql.side_effect = [
            {"data": [{"total": 2}]},  # Count query
            {
                "data": [  # Search query
                    {
                        "path": "/test/file1.txt",
                        "basename": "file1.txt",
                        "ext": "txt",
                        "dirpath": "/test",
                        "size": 1024,
                        "mtime": 1700000000,
                    },
                    {
                        "path": "/test/file2.pdf",
                        "basename": "file2.pdf",
                        "ext": "pdf",
                        "dirpath": "/test",
                        "size": 2048,
                        "mtime": 1700000100,
                    },
                ]
            },
        ]

        response = client.get("/search?q=file&mode=substr")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["basename"] == "file1.txt"
        assert data["results"][1]["basename"] == "file2.pdf"

    @patch("main.execute_sql")
    def test_stats_endpoint(self, mock_sql):
        """Test stats endpoint."""
        mock_sql.side_effect = [
            {"data": [{"total": 1000}]},  # Total files
            {
                "data": [{"total_size": 1073741824, "last_scan": 1700000000}]
            },  # Size and scan
            {
                "data": [  # Per-root stats
                    {"root": "data", "count": 800, "size": 900000000},
                    {"root": "archive", "count": 200, "size": 173741824},
                ]
            },
        ]

        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_files"] == 1000
        assert data["total_size"] == 1073741824
        assert data["last_scan"] == 1700000000
        assert len(data["roots"]) == 2

    @patch("main.execute_sql")
    def test_suggest_endpoint(self, mock_sql):
        """Test extension suggestions endpoint."""
        mock_sql.return_value = {
            "data": [
                {"ext": "txt", "count": 500},
                {"ext": "pdf", "count": 300},
                {"ext": "py", "count": 200},
            ]
        }

        response = client.get("/suggest")
        assert response.status_code == 200

        data = response.json()
        assert len(data["extensions"]) == 3
        assert data["extensions"][0]["ext"] == "txt"
        assert data["extensions"][0]["count"] == 500
