from __future__ import annotations

from typing import Any
from html.parser import HTMLParser
from html import escape
from pipelines.table.common import (_cells_from_rows, _normalize_cells, _normalize_rows, _rows_from_cells, _text)


def _collect_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        result = [value]
        for child in value.values():
            result.extend(_collect_dicts(child))
        return result
    if isinstance(value, (list, tuple)):
        result: list[dict[str, Any]] = []
        for child in value:
            result.extend(_collect_dicts(child))
        return result
    return []


def _find_html(value: Any) -> str:
    if isinstance(value, str):
        return value if "<table" in value.lower() else ""
    if isinstance(value, list) and value and all(isinstance(item, (str, int, float)) for item in value):
        joined = "".join(str(item) for item in value)
        return joined if "<table" in joined.lower() else ""
    if isinstance(value, dict):
        for key in ("table_html", "html", "pred_html", "structure_html", "structure"):
            found = _find_html(value.get(key))
            if found:
                return found
        for child in value.values():
            found = _find_html(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_html(child)
            if found:
                return found
    return ""


class _TableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.cells: list[dict[str, Any]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None
        self.row = -1
        self.col = 0
        self.rowspan = 1
        self.colspan = 1
        self.occupied: set[tuple[int, int]] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.row += 1
            self.col = 0
            self.current_row = []
        elif tag in {"td", "th"} and self.current_row is not None:
            while (self.row, self.col) in self.occupied:
                self.current_row.append("")
                self.col += 1
            values = {key.lower(): value for key, value in attrs}
            try:
                self.rowspan = max(1, int(values.get("rowspan") or 1))
                self.colspan = max(1, int(values.get("colspan") or 1))
            except (TypeError, ValueError):
                self.rowspan = self.colspan = 1
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.current_row is not None and self.current_cell is not None:
            value = _text("".join(self.current_cell))
            self.cells.append(
                {
                    "row": self.row, "col": self.col, "text": value,
                    "ocrText": value, "groundTruth": value,
                    "rowSpan": self.rowspan, "colSpan": self.colspan,
                }
            )
            self.current_row.extend([value, *([""] * (self.colspan - 1))])
            for row_offset in range(self.rowspan):
                for col_offset in range(self.colspan):
                    position = (self.row + row_offset, self.col + col_offset)
                    self.occupied.add(position)
                    if row_offset or col_offset:
                        self.cells.append(
                            {
                                "row": position[0], "col": position[1], "text": "",
                                "ocrText": "", "groundTruth": "", "rowSpan": 1,
                                "colSpan": 1, "hidden": True,
                            }
                        )
            self.col += self.colspan
            self.current_cell = None
            self.rowspan = self.colspan = 1
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None


def _structured_from_html(html: str) -> dict[str, Any] | None:
    if not html:
        return None
    parser = _TableHtmlParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    rows = _normalize_rows(parser.rows)
    if not rows:
        return None
    return {"rows": rows, "cells": parser.cells, "headerRowCount": 1, "postProcessing": "stdlib-html-parser"}


def _extract_structure(data: dict[str, Any]) -> tuple[str, list[list[str]], dict[str, Any] | None]:
    html = _find_html(data)
    for record in _collect_dicts(data):
        structured = record.get("table_structured", record.get("structured"))
        if isinstance(structured, dict):
            cells = _normalize_cells(structured.get("cells"))
            rows = _normalize_rows(structured.get("rows")) or _rows_from_cells(cells)
            if rows or cells:
                return html, rows, {**structured, "rows": rows, "cells": cells or _cells_from_rows(rows)}
        cells = _normalize_cells(record.get("cells", record.get("table_cells")))
        rows = _normalize_rows(record.get("table_rows", record.get("rows"))) or _rows_from_cells(cells)
        if rows or cells:
            return html, rows, {"rows": rows, "cells": cells or _cells_from_rows(rows), "headerRowCount": 1}
    parsed = _structured_from_html(html)
    return (html, parsed["rows"], parsed) if parsed else (html, [], None)


def _rows_html(rows: list[list[str]]) -> str:
    return "<table>" + "".join(
        "<tr>" + "".join(f"<td>{escape(_text(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    ) + "</table>"

