"""Compatibility exports; table processing lives in pipelines.table."""

from pipelines.table.common import (
    _BORDERLESS_MIN_COLUMNS,
    _BORDERLESS_MIN_ROWS,
    _TABLE_BORDERLESS_FINAL_CONFIDENCE_THRESHOLD,
    _TABLE_BORDERLESS_FILL_RATIO_THRESHOLD,
    _TABLE_BORDERLESS_COLUMN_CONSISTENCY_THRESHOLD,
    _TABLE_BORDERLESS_SPARSE_ROW_RATIO_THRESHOLD,
    _TABLE_CANDIDATE_TIE_EPSILON,
    _TABLE_LOW_OCR_CONFIDENCE_THRESHOLD,
    _SEMI_TABLE_MIN_CONFIDENCE,
    _SEMI_TABLE_MIN_TOPOLOGY_CHANGE_RATIO,
    _clamp01,
    _text,
    _normalize_rows,
    _markdown_table,
    _bbox,
    _normalize_cells,
    _rows_from_cells,
    _cells_from_rows,
    _intersection_area,
)

from pipelines.table.html_parser import (
    _collect_dicts,
    _find_html,
    _TableHtmlParser,
    _structured_from_html,
    _extract_structure,
    _rows_html,
)

from pipelines.table.cell_assignment import (
    _ocr_cells,
    _cluster_ocr,
    _grid_from_ocr,
    _assign_ocr_to_structured_cells,
    _slice_ocr,
)

from pipelines.table.quality import (
    _quality,
    table_result_needs_fallback,
)

from pipelines.table.candidates import (
    _candidate,
    _summary,
    _select,
)

from pipelines.table.grid_analysis import (
    _line_groups,
    _semi_table_regions,
    analyze_grid,
    _slice_grid,
)

from pipelines.table.result_assembly import (
    assemble_table_result,
    assemble_semi_table_result,
)
