import asyncio
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

from clients.model_service_client import HTTPModelClient
from pipelines.table import orchestrator
from shared import upstream
from shared.contracts import ModelAPIError


def structure(rows):
    return {"table_rows": rows, "table_structured": {
        "rows": rows, "cells": [
            {"row": r, "col": c, "text": text, "rowSpan": 1, "colSpan": 1}
            for r, row in enumerate(rows) for c, text in enumerate(row)
        ],
    }}


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def infer(self, base_url, endpoint, paths, **kwargs):
        self.calls.append((base_url, endpoint, paths, kwargs))
        result = self.responses[base_url]
        if isinstance(result, Exception):
            raise result
        return result


def run_table(client, mode="auto", include_ocr=True):
    return asyncio.run(orchestrator.recognize_table(
        Path("input.png"), mode=mode, include_ocr=include_ocr,
        wired_url="wired", wireless_url="wireless", ocr_url="ocr",
        request_id="req_test", client=client,
    ))


@pytest.fixture
def grid(monkeypatch):
    monkeypatch.setattr(orchestrator, "analyze_grid", lambda path: {
        "detected": True, "confidence": 0.9,
    })


def test_table_good_initial_candidate_does_not_call_alternate(grid):
    rows = [["Name", "Amount"], ["Alice", "100"]]
    client = FakeClient({"wired": structure(rows), "ocr": {}})
    result = run_table(client)
    assert result["table_rows"] == rows
    assert sorted(call[0] for call in client.calls) == ["ocr", "wired"]
    assert not result["table_debug"]["auto_strategy"]["alternate_called"]
    assert all(call[3]["request_id"] == "req_test" for call in client.calls)


@pytest.mark.parametrize("initial", [structure([["collapsed"]]), RuntimeError("failed")])
def test_table_fallback_reuses_single_ocr_call(grid, initial):
    client = FakeClient({"wired": initial, "ocr": {}, "wireless": structure([
        ["Name", "Amount"], ["Alice", "100"],
    ])})
    result = run_table(client)
    assert sorted(call[0] for call in client.calls) == ["ocr", "wired", "wireless"]
    assert result["table_debug"]["auto_strategy"]["alternate_called"]
    if isinstance(initial, Exception):
        assert result["table_debug"]["upstream_failures"] == [
            {"upstream": "wired", "code": "UPSTREAM_CALL_FAILED"},
        ]


def test_table_explicit_mode_never_calls_alternate(grid):
    client = FakeClient({"wired": structure([["collapsed"]])})
    run_table(client, mode="wired", include_ocr=False)
    assert [call[0] for call in client.calls] == ["wired"]


def test_table_all_upstreams_failed_preserves_error(grid):
    client = FakeClient({"wired": RuntimeError(), "wireless": RuntimeError()})
    with pytest.raises(ModelAPIError) as caught:
        run_table(client, include_ocr=False)
    assert caught.value.code == "TABLE_CANDIDATES_EMPTY"


@pytest.mark.parametrize("batch_fails", [False, True])
def test_semi_table_crops_are_cleaned_and_ocr_not_repeated(monkeypatch, tmp_path, batch_fails):
    crops = [tmp_path / "a.png", tmp_path / "b.png"]
    for crop in crops:
        crop.touch()
    monkeypatch.setattr(orchestrator, "analyze_grid", lambda path: {
        "detected": True, "semi_analysis": {"detected": True, "regions": []},
    })
    monkeypatch.setattr(orchestrator, "_semi_region_crops", lambda *args: crops)
    monkeypatch.setattr(orchestrator, "assemble_table_result", lambda **kwargs: {"good": False})
    monkeypatch.setattr(orchestrator, "assemble_semi_table_result", lambda **kwargs: {"good": True})
    monkeypatch.setattr(orchestrator, "table_result_needs_fallback", lambda result: not result["good"])

    class SemiClient(FakeClient):
        def infer(self, base_url, endpoint, paths, **kwargs):
            if endpoint == "/api/v1/table-structure-batches":
                self.calls.append((base_url, endpoint, paths, kwargs))
                assert kwargs["multiple"] is True
                if batch_fails:
                    raise RuntimeError("batch failed")
                return {"predictions": [{}, {}]}
            return super().infer(base_url, endpoint, paths, **kwargs)

    client = SemiClient({"wired": {}, "wireless": {}, "ocr": {}})
    result = run_table(client)
    assert not any(crop.exists() for crop in crops)
    assert sum(call[0] == "ocr" for call in client.calls) == 1
    assert result["table_debug"]["auto_strategy"]["alternate_called"] is batch_fails


def test_http_client_preserves_transport_fields_and_errors(monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return {"predictions": []}
    monkeypatch.setattr(upstream, "post_images", post)
    fields = {"version": "v5", "model": "thai_ft_v1"}
    result = HTTPModelClient().infer(
        "rec", "/api/v1/text-recognition-batches", [Path("crop.png")],
        request_id="req_test", fields=fields, multiple=True,
    )
    assert result == {"predictions": []}
    assert calls == [(("rec", "/api/v1/text-recognition-batches", [Path("crop.png")]),
                      {"request_id": "req_test", "fields": fields, "multiple": True})]
    error = ModelAPIError(502, "UPSTREAM_MODEL_ERROR", "upstream failed")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(upstream, "post_images", fail)
    with pytest.raises(ModelAPIError) as caught:
        HTTPModelClient().infer("rec", "/api/v1/text-recognitions", [], request_id="r")
    assert caught.value is error
