# -*- coding: utf-8 -*-
"""Table grid reconstruction for multi-column reports.

PP-OCR returns flat text blocks with boxes; on dense multi-column tables
the naive "cluster by y, sort by x" reading order scrambles columns (the
existing ``_sort_reading_order`` even falls back to detection order for
these). This module rebuilds an explicit {row, column, text} grid from the
block boxes so the downstream normalizer can extract per-cell instead of
guessing column membership from a linear text stream.

Pipeline:
1. column boundaries  - cluster every block's left/right edges into a
   sorted list of vertical bands; each block belongs to the band its
   center falls in.
2. row bands          - cluster blocks by vertical position (y-center
   proximity with row-height tolerance).
3. grid               - each (row band, column band) cell merges the
   blocks whose center lies inside it.

Every step is deterministic and fails safe: if the layout does not look
tabular (few blocks, huge spread, no shared column edges), the module
returns None and the caller keeps the plain linear text path.
"""
from __future__ import annotations

import re
import statistics
from typing import Any

from ..schemas import OCRBlock


class TableGrid:
    def __init__(
        self,
        columns: list[list[float]],
        rows: list[list[float]],
        cells: dict[tuple[int, int], list[tuple[float, float, str, Any]]],
    ) -> None:
        # columns: [[x0, x1], ...] left-to-right; rows: [[y0, y1], ...] top-down
        self.columns = columns
        self.rows = rows
        # (row_idx, col_idx) -> [(cx, cy, text, confidence), ...]; fragments
        # keep their block position so cell_text can order them and
        # low_confidence_matrix can flag cells sourced from weak OCR reads.
        self.cells = cells

    def cell_text(self, row: int, col: int) -> str:
        items = sorted(
            self.cells.get((row, col), []), key=lambda it: (it[1], it[0])
        )
        return " ".join(it[2] for it in items).strip()

    # Axis-scale tick rows: on vital-signs sheets the left column is the
    # chart's Y axis (pulse 180..20, temperature 42°..34°), one tick per
    # row. Each such row holds a single cell whose text is only numbers
    # ("180 42", "160 41°"), and the leading numbers run strictly
    # monotonic down the sheet. These are layout ticks, not recorded data.
    _AXIS_TEXT = re.compile(r"^\d{1,4}(?:\s*\d{1,4}°?)?$")

    def _axis_rows(self) -> set[int]:
        candidates: list[tuple[int, float]] = []
        for r in range(len(self.rows)):
            filled = [
                (c, self.cell_text(r, c))
                for c in range(len(self.columns))
                if self.cell_text(r, c)
            ]
            if len(filled) != 1:
                continue
            text = filled[0][1]
            if not self._AXIS_TEXT.match(text):
                continue
            first_num = re.match(r"(\d+)", text)
            if not first_num:
                continue
            candidates.append((r, float(first_num.group(1))))
        if len(candidates) < 3:
            return set()
        firsts = [v for _, v in candidates]
        monotonic = all(b < a for (a, b) in zip(firsts, firsts[1:])) or all(
            b > a for (a, b) in zip(firsts, firsts[1:])
        )
        if not monotonic:
            return set()
        return {r for r, _ in candidates}

    def _title_rows(self) -> set[int]:
        """Title rows above the header, e.g. "检验报告单" spanning the sheet.

        A title row fills only a small share of the columns with a longer
        text, unlike a header row (项目/结果/单位/参考范围) or a data row.
        Only the first row is considered; returning an empty set when the
        layout is too narrow to tell a title from data.
        """
        if len(self.rows) < 2 or len(self.columns) < 3:
            return set()
        filled = [
            self.cell_text(0, c)
            for c in range(len(self.columns))
            if self.cell_text(0, c)
        ]
        if not filled:
            return set()
        # A header/data row fills more than half of the columns.
        if len(filled) * 2 > len(self.columns):
            return set()
        if sum(len(t) for t in filled) < 4:
            return set()
        return {0}

    def to_text(self) -> str:
        """Render the grid as a readable row/column table for the model."""
        axis_rows = self._axis_rows()
        lines: list[str] = []
        for r in range(len(self.rows)):
            cells = [
                self.cell_text(r, c) or "-"
                for c in range(len(self.columns))
            ]
            marker = "【轴刻度行，非数据】" if r in axis_rows else ""
            lines.append(f"第{r + 1}行{marker}: " + " | ".join(cells))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "cells": {
                f"{r},{c}": self.cell_text(r, c)
                for r in range(len(self.rows))
                for c in range(len(self.columns))
            },
        }

    def to_matrix(self) -> list[list[str]] | None:
        """Row-major matrix for the frontend table view.

        Axis tick rows (chart Y-scale coordinates) are layout, not data:
        drop them so the table only carries recorded rows. Returns None
        when the reconstructed grid is too sparse to be useful as a table
        (e.g. handwritten sheets whose blocks scatter across a huge empty
        area) — the caller then keeps the fact-list view instead.
        """
        axis_rows = self._axis_rows()
        skip = axis_rows | self._title_rows()
        matrix = [
            [self.cell_text(r, c) or "" for c in range(len(self.columns))]
            for r in range(len(self.rows))
            if r not in skip
        ]
        if not _useful_matrix(matrix, len(self.columns)):
            return None
        return matrix

    def apply_corrections(
        self, corrected: list[list[str]] | None
    ) -> list[list[str]] | None:
        """Take the GLM per-cell corrected matrix and normalize it.

        The vision model reads the grid text and outputs a same-shape 2D
        array with garbled cells fixed (e.g. OCR "7/6" -> "g/L"). This
        validates the shape (must match the OCR grid), drops axis tick
        rows, and applies the same sparsity floor as to_matrix(). Returns
        None when the correction is malformed — the caller then keeps the
        plain OCR matrix.
        """
        if not corrected or not isinstance(corrected, list):
            return None
        if len(corrected) != len(self.rows):
            return None
        ncols = len(self.columns)
        norm: list[list[str]] = []
        for r, row in enumerate(corrected):
            if not isinstance(row, list):
                return None
            cells = [str(c).strip() for c in row[:ncols]]
            while len(cells) < ncols:
                cells.append("")
            norm.append(cells)
        axis_rows = self._axis_rows()
        skip = axis_rows | self._title_rows()
        matrix = [
            row for r, row in enumerate(norm) if r not in skip
        ]
        if not _useful_matrix(matrix, ncols):
            return None
        return matrix

    def low_confidence_matrix(
        self, floor: float | None = None
    ) -> list[list[bool]]:
        """Per-cell weak-OCR flags, row-aligned with to_matrix().

        A cell is flagged when any fragment inside it carried an OCR
        confidence below `floor`. Rows skipped by to_matrix() (axis ticks,
        title rows) are skipped here too, so the two matrices line up.
        """
        if floor is None:
            floor = _COL_CONF_FLOOR
        axis_rows = self._axis_rows()
        skip = axis_rows | self._title_rows()
        ncols = len(self.columns)
        matrix: list[list[bool]] = []
        for r in range(len(self.rows)):
            if r in skip:
                continue
            row: list[bool] = []
            for c in range(ncols):
                low = False
                for frag in self.cells.get((r, c), []):
                    conf = frag[3] if len(frag) > 3 else None
                    if conf is None:
                        continue
                    try:
                        if float(conf) < floor:
                            low = True
                            break
                    except (TypeError, ValueError):
                        continue
                row.append(low)
            matrix.append(row)
        return matrix


_SPARSITY_FLOOR = 0.25
# OCR confidence below this is usually garbled handwriting or noise; such
# blocks must not shape the column bands (they create phantom columns), but
# they still land in cells and reach the model.
_COL_CONF_FLOOR = 0.45


def _useful_matrix(matrix: list[list[str]], ncols: int) -> bool:
    """Whether a grid reads as a table worth showing.

    Two traps this guards against:
    - a filled-share floor alone accepts handwritten sheets where two long
      header texts and a garbled title pile up on a wide canvas;
    - a pure sparsity floor rejects wide reports that legitimately leave
      most columns empty per row.
    So a table must first have >=3 columns carrying content on 2+ rows
    (that is what makes a grid a grid). Above that, either the plain fill
    floor passes, or a header row plus >=3 rows with two cells each passes
    for genuinely wide layouts (4+ columns).
    """
    if not matrix:
        return False
    cols_ok = sum(
        1
        for c in range(ncols)
        if sum(1 for row in matrix if row[c]) >= 2
    )
    if cols_ok < 3:
        return False
    total = len(matrix) * ncols or 1
    filled = sum(1 for row in matrix for cell in row if cell)
    if filled / total >= _SPARSITY_FLOOR:
        return True
    if ncols < 4:
        return False
    rows_ok = sum(1 for row in matrix if sum(1 for cell in row if cell) >= 2)
    header_ok = sum(1 for cell in matrix[0] if cell) >= 2
    return header_ok and rows_ok >= 3


def _merge_edges(edges: list[float], tol: float) -> list[float]:
    """Merge sorted boundary coordinates within tolerance into band edges."""
    if not edges:
        return []
    edges = sorted(edges)
    merged: list[float] = [edges[0]]
    for e in edges[1:]:
        if e - merged[-1] <= tol:
            merged[-1] = (merged[-1] + e) / 2.0
        else:
            merged.append(e)
    return merged


def _columns_from_centers(items: list[dict[str, Any]]) -> list[list[float]] | None:
    """Cluster column centers; None when the layout is not clearly columned."""
    items.sort(key=lambda i: i["cx"])
    widths = [i["x1"] - i["x0"] for i in items]
    median_w = statistics.median(widths) or 1.0
    gap_tol = max(median_w * 0.6, 10.0)
    clusters: list[list[dict[str, Any]]] = []
    current = [items[0]]
    last_cx = items[0]["cx"]
    for i in items[1:]:
        if i["cx"] - last_cx <= gap_tol:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
        last_cx = i["cx"]
    clusters.append(current)
    if len(clusters) < 2:
        return None
    columns: list[list[float]] = [
        [min(i["x0"] for i in c), max(i["x1"] for i in c)] for c in clusters
    ]
    # Merge overlapping columns (e.g. a header that spans several data
    # columns creates a wide cluster overlapping narrower neighbors).
    merged: list[list[float]] = []
    for col in columns:
        if not merged or col[0] > merged[-1][1]:
            merged.append(col)
        else:
            merged[-1][1] = max(merged[-1][1], col[1])
    return merged if len(merged) >= 2 else None


def _columns_from_edges(items: list[dict[str, Any]]) -> list[list[float]] | None:
    """Fallback: cluster left/right block edges into column bands.

    Works better on dense narrow grids (handwritten temperature sheets)
    where column centers sit too close together to cluster reliably.
    """
    edges: list[float] = []
    for i in items:
        edges.append(i["x0"])
        edges.append(i["x1"])
    widths = [i["x1"] - i["x0"] for i in items]
    median_w = statistics.median(widths) or 1.0
    edge_tol = max(median_w * 0.25, 6.0)
    band_edges = _merge_edges(edges, edge_tol)
    columns: list[list[float]] = []
    for idx in range(len(band_edges) - 1):
        columns.append([band_edges[idx], band_edges[idx + 1]])
    columns = [
        col for col in columns if any(col[0] <= i["cx"] <= col[1] for i in items)
    ]
    return columns if len(columns) >= 2 else None


def _items_from_blocks(
    blocks: list[OCRBlock] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in blocks:
        bbox = block.bbox if isinstance(block, OCRBlock) else block.get("bbox")
        text = block.text if isinstance(block, OCRBlock) else block.get("text", "")
        if not bbox or len(bbox) != 4 or not text:
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox)
        items.append(
            {
                "text": str(text).strip(),
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
                "confidence": (
                    block.confidence
                    if isinstance(block, OCRBlock)
                    else block.get("confidence")
                ),
            }
        )
    return items


def _build_from_items(
    items: list[dict[str, Any]],
    min_columns: int = 2,
) -> TableGrid | None:
    """Build one grid from the blocks of a single table region."""
    if len(items) < min_columns * 2:
        return None

    # Low-confidence blocks are usually garbled handwriting; letting them
    # vote on column bands invents phantom columns. They still land in
    # cells below, so nothing disappears from the model's view.
    confident = [
        i
        for i in items
        if i.get("confidence") is None
        or float(i["confidence"]) >= _COL_CONF_FLOOR
    ]
    if len(confident) < min_columns * 2:
        confident = items
    columns = _columns_from_centers(confident) or _columns_from_edges(confident)
    if columns is None or len(columns) < min_columns:
        return None

    # --- row bands from y-centers ---
    items = sorted(items, key=lambda i: i["cy"])
    heights = [i["y1"] - i["y0"] for i in items]
    median_h = statistics.median(heights) or 1.0
    row_gap = max(median_h * 0.5, 8.0)
    # Merged cells (group titles) span several rows; if they anchored the
    # row bands, their covered rows would collapse into one band. Build the
    # bands from normal-height lines only and spread the tall ones after.
    row_items = [
        i for i in items if (i["y1"] - i["y0"]) <= median_h * 1.6
    ] or items
    rows: list[list[float]] = []
    current_min = row_items[0]["y0"]
    current_max = row_items[0]["y1"]
    last_cy = row_items[0]["cy"]
    for i in row_items[1:]:
        if abs(i["cy"] - last_cy) <= row_gap:
            current_min = min(current_min, i["y0"])
            current_max = max(current_max, i["y1"])
        else:
            rows.append([current_min, current_max])
            current_min, current_max = i["y0"], i["y1"]
        last_cy = i["cy"]
    rows.append([current_min, current_max])
    if len(rows) < 2:
        return None

    # --- assign blocks to (row, col); merged cells spread across rows ---
    cells: dict[tuple[int, int], list[tuple[float, float, str]]] = {}

    def _nearest_column(cx: float) -> int:
        return min(
            range(len(columns)),
            key=lambda idx: 0
            if columns[idx][0] <= cx <= columns[idx][1]
            else min(abs(cx - columns[idx][0]), abs(cx - columns[idx][1])),
        )

    for i in items:
        c = next(
            (idx for idx, (x0, x1) in enumerate(columns) if x0 <= i["cx"] <= x1),
            None,
        )
        if c is None:
            # A low-confidence block can sit outside the confirmed column
            # bands; attach it to the nearest column rather than drop it.
            c = _nearest_column(i["cx"])
        covered = [
            idx
            for idx, (y0, y1) in enumerate(rows)
            if i["y0"] <= y1 and i["y1"] >= y0
        ]
        if not covered:
            continue
        if (i["y1"] - i["y0"]) > median_h * 1.6 and len(covered) >= 2:
            # A merged cell (group title like "肝功能" spanning several
            # data rows): repeat it on every covered row so those rows do
            # not read as empty group columns.
            targets = covered
        else:
            targets = [
                idx
                for idx, (y0, y1) in enumerate(rows)
                if y0 <= i["cy"] <= y1
            ]
        for r in targets:
            cells.setdefault((r, c), []).append(
                (i["cx"], i["cy"], i["text"], i.get("confidence"))
            )

    return TableGrid(columns=columns, rows=rows, cells=cells)


def _split_regions(
    items: list[dict[str, Any]],
    min_columns: int = 2,
) -> list[list[dict[str, Any]]]:
    """Split blocks into separate table regions at the largest vertical gap.

    One photo may hold two independent tables (blood count on top, urine
    routine below). Rebuilding across both mixes their column bands and
    scrambles the grid; a gap of more than ~3 line heights separates them.
    Returns [items] when the layout is a single table.
    """
    need = max(min_columns * 2, 4)
    if len(items) < need * 2:
        return [items]
    ordered = sorted(items, key=lambda i: i["cy"])
    heights = [i["y1"] - i["y0"] for i in ordered]
    median_h = statistics.median(heights) or 1.0
    row_gap = max(median_h * 0.5, 8.0)
    bands: list[list[dict[str, Any]]] = []
    current = [ordered[0]]
    for i in ordered[1:]:
        if i["cy"] - current[-1]["cy"] <= row_gap:
            current.append(i)
        else:
            bands.append(current)
            current = [i]
    bands.append(current)
    spans = [
        (min(i["y0"] for i in band), max(i["y1"] for i in band))
        for band in bands
    ]
    if len(spans) < 2:
        return [items]
    gap, split_idx = max(
        ((spans[k + 1][0] - spans[k][1], k) for k in range(len(spans) - 1))
    )
    if gap < median_h * 3.0:
        return [items]
    cut = spans[split_idx][1]
    above = [i for i in ordered if i["cy"] <= cut]
    below = [i for i in ordered if i["cy"] > cut]
    if len(above) < need or len(below) < need:
        return [items]
    return [above, below]


def reconstruct_grid(
    blocks: list[OCRBlock] | list[dict[str, Any]],
    min_columns: int = 2,
) -> TableGrid | None:
    """Rebuild a row/column grid from OCR blocks with bboxes.

    Returns None when the layout is not confidently tabular. When the page
    holds several tables, the grid comes from the largest region so its
    columns stay accurate; the other region's text still reaches the model
    through the image itself.
    """
    items = _items_from_blocks(blocks)
    if len(items) < min_columns * 2:
        return None
    grids = [
        grid
        for region in _split_regions(items, min_columns)
        if (grid := _build_from_items(region, min_columns)) is not None
    ]
    if not grids:
        return None
    return max(grids, key=lambda g: sum(len(v) for v in g.cells.values()))
