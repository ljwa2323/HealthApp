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

import statistics
from typing import Any

from ..schemas import OCRBlock


class TableGrid:
    def __init__(
        self,
        columns: list[list[float]],
        rows: list[list[float]],
        cells: dict[tuple[int, int], list[str]],
    ) -> None:
        # columns: [[x0, x1], ...] left-to-right; rows: [[y0, y1], ...] top-down
        self.columns = columns
        self.rows = rows
        self.cells = cells  # (row_idx, col_idx) -> list of cell text fragments

    def cell_text(self, row: int, col: int) -> str:
        return " ".join(self.cells.get((row, col), [])).strip()

    def to_text(self) -> str:
        """Render the grid as a readable row/column table for the model."""
        lines: list[str] = []
        for r in range(len(self.rows)):
            cells = [
                self.cell_text(r, c) or "-"
                for c in range(len(self.columns))
            ]
            lines.append(f"第{r + 1}行: " + " | ".join(cells))
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


def reconstruct_grid(
    blocks: list[OCRBlock] | list[dict[str, Any]],
    min_columns: int = 2,
) -> TableGrid | None:
    """Rebuild a row/column grid from OCR blocks with bboxes.

    Returns None when the layout is not confidently tabular.
    """
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
    if len(items) < min_columns * 2:
        return None

    columns = _columns_from_centers(items) or _columns_from_edges(items)
    if columns is None or len(columns) < min_columns:
        return None

    # --- row bands from y-centers ---
    items.sort(key=lambda i: i["cy"])
    heights = [i["y1"] - i["y0"] for i in items]
    median_h = statistics.median(heights) or 1.0
    row_gap = max(median_h * 0.5, 8.0)
    rows: list[list[float]] = []
    current_min = items[0]["y0"]
    current_max = items[0]["y1"]
    last_cy = items[0]["cy"]
    for i in items[1:]:
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

    # --- assign every block to (row, col) ---
    cells: dict[tuple[int, int], list[str]] = {}
    for i in items:
        r = next(
            (idx for idx, (y0, y1) in enumerate(rows) if y0 <= i["cy"] <= y1),
            None,
        )
        c = next(
            (idx for idx, (x0, x1) in enumerate(columns) if x0 <= i["cx"] <= x1),
            None,
        )
        if r is None or c is None:
            continue
        cells.setdefault((r, c), []).append(i["text"])

    return TableGrid(columns=columns, rows=rows, cells=cells)
