from pathlib import Path

from shared.upstream import get_readiness, post_images


class _Response:
    status_code = 200
    ok = True
    headers = {}

    def json(self):
        return {"data": {"status": "ok"}}

    def raise_for_status(self):
        return None


def test_internal_token_is_sent_to_model_service(monkeypatch, tmp_path: Path):
    token = "internal-token-with-more-than-thirty-two-characters"
    monkeypatch.setenv("INTERNAL_API_TOKEN", token)
    image = tmp_path / "sample.png"
    image.write_bytes(b"image")
    captured = {}

    def post(*args, **kwargs):
        captured.update(kwargs.get("headers") or {})
        return _Response()

    monkeypatch.setattr("shared.upstream.requests.post", post)
    post_images("http://model", "/api/v1/predict", [image], request_id="req_test")

    assert captured["Authorization"] == f"Bearer {token}"
    assert captured["X-Request-ID"] == "req_test"


def test_internal_token_is_sent_to_readiness(monkeypatch):
    token = "internal-token-with-more-than-thirty-two-characters"
    monkeypatch.setenv("INTERNAL_API_TOKEN", token)
    captured = {}

    def get(*args, **kwargs):
        captured.update(kwargs.get("headers") or {})
        return _Response()

    monkeypatch.setattr("shared.upstream.requests.get", get)
    get_readiness("http://model", request_id="req_test")

    assert captured["Authorization"] == f"Bearer {token}"
