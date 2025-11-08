"""Integration tests for the filesystem indexer system."""

import os
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8080")


class TestIntegration:
    """Integration test cases."""

    def test_api_health(self):
        """Test API health check."""
        response = requests.get(f"{API_URL}/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_search_basic(self):
        """Test basic search functionality."""
        response = requests.get(
            f"{API_URL}/search", params={"q": "test", "mode": "substr"}, timeout=5
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data

    def test_search_with_filters(self):
        """Test search with filters."""
        response = requests.get(
            f"{API_URL}/search",
            params={
                "q": "test",
                "mode": "substr",
                "ext": ["txt", "pdf"],
                "sort": "mtime_desc",
                "per_page": 10,
            },
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 10

    def test_stats(self):
        """Test statistics endpoint."""
        response = requests.get(f"{API_URL}/stats", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "total_files" in data
        assert data["total_files"] >= 0

    def test_suggestions(self):
        """Test extension suggestions."""
        response = requests.get(f"{API_URL}/suggest", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "extensions" in data
        assert isinstance(data["extensions"], list)

    def test_pagination(self):
        """Test result pagination."""
        # First page
        response1 = requests.get(
            f"{API_URL}/search", params={"page": 1, "per_page": 5}, timeout=5
        )
        assert response1.status_code == 200
        data1 = response1.json()

        # Second page
        response2 = requests.get(
            f"{API_URL}/search", params={"page": 2, "per_page": 5}, timeout=5
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # Results should be different if there are enough files
        if data1["total"] > 5:
            assert data1["results"] != data2["results"]
