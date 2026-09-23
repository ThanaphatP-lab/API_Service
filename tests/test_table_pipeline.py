import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

from pipeline.table_pipeline import assemble_table_result, table_result_needs_fallback


def _structure(rows):
    cells = [
        {"row": row, "col": col, "text": text, "rowSpan": 1, "colSpan": 1}
        for row, values in enumerate(rows)
        for col, text in enumerate(values)
    ]
    return {"table_structured": {"rows": rows, "cells": cells}, "table_rows": rows}


def test_auto_quality_accepts_usable_initial_candidate_without_alternate():
    result = assemble_table_result(
        mode="auto",
        wired=_structure([["Name", "Amount"], ["Alice", "100"]]),
        wireless=None,
        ocr=None,
        grid={"detected": True, "confidence": 0.9},
    )
    assert result["table_rows"] == [["Name", "Amount"], ["Alice", "100"]]
    assert table_result_needs_fallback(result) is False


def test_auto_quality_requests_alternate_for_collapsed_structure():
    result = assemble_table_result(
        mode="auto",
        wired=_structure([["Name Amount Alice 100"]]),
        wireless=None,
        ocr=None,
        grid={"detected": True, "confidence": 0.2},
    )
    assert table_result_needs_fallback(result) is True


def test_ocr_geometry_recovers_empty_structure_model():
    ocr = {
        "segments": [
            {"text": "Name", "confidence": 0.9, "bbox": {"x": 10, "y": 10, "width": 40, "height": 12}},
            {"text": "Amount", "confidence": 0.9, "bbox": {"x": 110, "y": 10, "width": 50, "height": 12}},
            {"text": "Alice", "confidence": 0.8, "bbox": {"x": 10, "y": 45, "width": 40, "height": 12}},
            {"text": "100", "confidence": 0.8, "bbox": {"x": 110, "y": 45, "width": 50, "height": 12}},
        ]
    }
    result = assemble_table_result(
        mode="wireless",
        wired=None,
        wireless={"predictions": [{}]},
        ocr=ocr,
        grid={"detected": False, "confidence": 0.0},
    )
    assert result["table_rows"] == [["Name", "Amount"], ["Alice", "100"]]
    assert result["selected_method"] == "borderless_text_clustering"


def test_ocr_geometry_preserves_wide_merged_header_span():
    ocr = {
        "segments": [
            {"text": "Summary", "confidence": 0.9, "bbox": {"x": 5, "y": 5, "width": 170, "height": 14}},
            {"text": "Name", "confidence": 0.9, "bbox": {"x": 10, "y": 35, "width": 45, "height": 12}},
            {"text": "Amount", "confidence": 0.9, "bbox": {"x": 110, "y": 35, "width": 55, "height": 12}},
        ]
    }
    result = assemble_table_result(
        mode="wireless", wired=None, wireless={"predictions": [{}]}, ocr=ocr,
        grid={"detected": False, "confidence": 0.0},
    )
    owner = next(cell for cell in result["table_structured"]["cells"] if cell.get("text") == "Summary")
    assert owner["colSpan"] == 2
