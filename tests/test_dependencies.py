from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from farl.dependencies import request_endpoint


class TestRequestEndpoint:
    def test_request_endpoint_basic(self):
        """Test basic functionality with path and method"""
        # Mock request object
        request = Request(
            {
                "type": "http",
                "path": "/api/users",
                "method": "GET",
            }
        )

        result = request_endpoint(request)
        assert result == "/api/users:GET"

    def test_request_endpoint_post_method(self):
        """Test with POST method"""
        request = Request(
            {
                "type": "http",
                "path": "/api/users",
                "method": "POST",
            }
        )

        result = request_endpoint(request)
        assert result == "/api/users:POST"

    def test_request_endpoint_empty_path(self):
        """Test with empty path"""
        request = Request(
            {
                "type": "http",
                "path": "",
                "method": "GET",
            }
        )

        result = request_endpoint(request)
        assert result == ":GET"

    def test_request_endpoint_missing_path(self):
        """Test when path is missing from scope"""
        request = Request(
            {
                "type": "http",
                "method": "GET",
            }
        )

        result = request_endpoint(request)
        assert result == ":GET"

    def test_request_endpoint_with_query_params(self):
        """Test with query parameters (should be ignored)"""
        request = Request(
            {
                "type": "http",
                "path": "/api/users",
                "method": "GET",
                "query_string": b"id=123&name=test",
            }
        )

        result = request_endpoint(request)
        assert result == "/api/users:GET"

    def test_request_endpoint_nested_path(self):
        """Test with nested path"""
        request = Request(
            {
                "type": "http",
                "path": "/api/v1/users/123/posts",
                "method": "DELETE",
            }
        )

        result = request_endpoint(request)
        assert result == "/api/v1/users/123/posts:DELETE"

    def test_request_endpoint_integration_with_fastapi(self):
        """Test integration with actual FastAPI request"""
        app = FastAPI()

        @app.get("/test/{item_id}")
        def test_endpoint(request: Request, item_id: int):
            endpoint = request_endpoint(request)
            return {"endpoint": endpoint, "item_id": item_id}

        with TestClient(app) as client:
            response = client.get("/test/42")

            assert response.status_code == 200
            data = response.json()
            assert data["endpoint"] == "/test/42:GET"
            assert data["item_id"] == 42
