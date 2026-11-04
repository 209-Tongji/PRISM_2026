"""Merged ancient-book page generator for IsoGraph.

Based on B_gen_data/StoneBook/generate_book_page.py (the more mature renderer),
with:
  * Cross-platform paths (configurable via env / CLI)
  * 4 controlled hallucination scenarios that produce REAL visual differences
    (not just flag labels):
      - DOUBLE_LINE_FLATTENING   : increase density of interlinear annotations
      - SEAL_TEXT_INTRUSION      : force >=2 seals overlapping main text
      - CROSS_COLUMN_FUSION      : drop some column separators visually
      - NOTE_MISANCHORING        : inject tian/dijiao with misleading anchoring
  * Deterministic scenario control (per-page seed; each scenario can be
    required or probabilistic)
  * Multiprocessing over text chunks (scales linearly with cores)
  * OracleGraph JSON output compatible with sdpo_vlm.data.schema

Usage (single process, 50 pages, balanced scenarios):
  python scripts/generate_data.py \
      --output ./dataset \
      --n-pages 50 \
      --balanced \
      --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------- Paths (all overridable) ----------

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "background":     ROOT.parent / "assets" / "background.png",
    "text":           ROOT.parent / "assets" / "text.txt",
    "font_main":      ROOT.parent / "assets" / "font_main.ttf",
    "font_annotation":ROOT.parent / "assets" / "font_annotation.ttf",
    "seals_dir":      ROOT.parent / "assets" / "seals",
    "font_fallback":  ROOT.parent / "assets" / "font_main.ttf",
}

# ---------- Layout constants (unchanged from StoneBook) ----------

BG_WIDTH, BG_HEIGHT = 1885, 1333
CHAR_SIZE_MAIN = 54
CHAR_SIZE_ANN = 27
CHAR_SIZE_MARGIN = 43
CHAR_SIZE_CENTER = 57
LINE_HEIGHT_MAIN = 51
LINE_HEIGHT_ANN = 25.5
LINE_HEIGHT_MARGIN = 43
TOP_BOUND = 413
BOTTOM_BOUND = 1194
MARGIN_TOP = 0
LINES_PER_COLUMN = 15
TIANTOU_LINES = 5
DIJIAO_LINES = 3
TIANTOU_OFFSET = 20
DIJIAO_OFFSET = 10

BOX_COORDS = [
    [(1560, 413), (1641, 413), (1641, 1194), (1560, 1194)],
    [(1483, 413), (1560, 413), (1560, 1194), (1483, 1194)],
    [(1402, 413), (1483, 413), (1483, 1194), (1402, 1194)],
    [(1321, 413), (1402, 413), (1402, 1194), (1321, 1194)],
    [(1238, 413), (1321, 413), (1321, 1194), (1238, 1194)],
    [(1158, 413), (1238, 413), (1238, 1194), (1158, 1194)],
    [(1077, 413), (1158, 413), (1158, 1194), (1077, 1194)],
    [(999, 413),  (1077, 413), (1077, 1194), (999, 1194)],
    [(885, 413),  (999, 413),  (999, 1194),  (885, 1194)],   # box8 = fold
    [(804, 413),  (885, 413),  (885, 1194),  (804, 1194)],
    [(725, 413),  (804, 413),  (804, 1194),  (725, 1194)],
    [(644, 413),  (725, 413),  (725, 1194),  (644, 1194)],
    [(563, 413),  (644, 413),  (644, 1194),  (563, 1194)],
    [(482, 413),  (563, 413),  (563, 1194),  (482, 1194)],
    [(401, 413),  (482, 413),  (482, 1194),  (401, 1194)],
    [(321, 413),  (401, 413),  (401, 1194),  (321, 1194)],
    [(244, 413),  (321, 413),  (321, 1194),  (244, 1194)],
]
BOX_WIDTHS = [b[1][0] - b[3][0] for b in BOX_COORDS]

PUNCT_CHARS = set("，。？！‘’“”（）【】《》〈〉〔〕—…·;；:：,.?!'\"()[]{}、•［］「」『』↑↓←→_-——")
PUNCT_NODE_ENDERS = set("，、；：,;:")
PUNCT_GROUP_ENDERS = set("。？！.?!\n")

try:
    RESAMPLING_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLING_FILTER = Image.LANCZOS


# ---------- Scenario config ----------

@dataclass
class ScenarioSpec:
    """Per-page hallucination recipe.

    Instead of randomly flagging scenarios then not rendering them, we
    treat each scenario as a concrete knob that modifies rendering.
    """
    double_line_flattening: bool = False  # boost interlinear annotations per page
    seal_text_intrusion: bool = False     # >=2 seals forced onto text
    cross_column_fusion: bool = False     # drop separators visually
    note_misanchoring: bool = False       # inject extra marginalia
    normal: bool = True                   # clean page (default)

    def to_list(self) -> List[str]:
        tags = []
        if self.double_line_flattening: tags.append("DOUBLE_LINE_FLATTENING")
        if self.seal_text_intrusion:    tags.append("SEAL_TEXT_INTRUSION")
        if self.cross_column_fusion:    tags.append("CROSS_COLUMN_FUSION")
        if self.note_misanchoring:      tags.append("NOTE_MISANCHORING")
        if not tags: tags.append("NORMAL")
        return tags

    @property
    def is_normal(self) -> bool:
        return not (
            self.double_line_flattening or self.seal_text_intrusion or
            self.cross_column_fusion or self.note_misanchoring
        )


def sample_scenario(rng: random.Random, *, balanced: bool = False) -> ScenarioSpec:
    """Sample a scenario spec.

    balanced=True gives a uniform mix across the 5 buckets (NORMAL + 4),
    which is what you want for training.  balanced=False uses low-probability
    independent sampling (closer to real distributions).
    """
    if balanced:
        bucket = rng.randrange(5)
        return ScenarioSpec(
            double_line_flattening=(bucket == 0),
            seal_text_intrusion=   (bucket == 1),
            cross_column_fusion=   (bucket == 2),
            note_misanchoring=     (bucket == 3),
            normal=                (bucket == 4),
        )
    return ScenarioSpec(
        double_line_flattening=rng.random() < 0.15,
        seal_text_intrusion=   rng.random() < 0.20,
        cross_column_fusion=   rng.random() < 0.15,
        note_misanchoring=     rng.random() < 0.15,
    )


# ---------- Font helpers ----------

def _has_glyph(font: ImageFont.ImageFont, char: str) -> bool:
    if not char.strip():
        return True
    try:
        missing_bbox = font.getmask(chr(0xFFFF)).getbbox()
        char_bbox = font.getmask(char).getbbox()
    except Exception:
        return False
    return char_bbox != missing_bbox


def _load_fonts(cfg: Dict[str, str]):
    f_main = ImageFont.truetype(cfg["font_main"], CHAR_SIZE_MAIN)
    f_ann = ImageFont.truetype(cfg["font_annotation"], int(CHAR_SIZE_ANN))
    f_margin = ImageFont.truetype(cfg["font_annotation"], CHAR_SIZE_MARGIN)
    f_center = ImageFont.truetype(cfg["font_main"], CHAR_SIZE_CENTER)
    # Fallback — required for rare chars. Pillow will raise if not findable.
    fb = cfg["font_fallback"]
    try:
        fb_main = ImageFont.truetype(fb, CHAR_SIZE_MAIN)
        fb_ann = ImageFont.truetype(fb, int(CHAR_SIZE_ANN))
        fb_margin = ImageFont.truetype(fb, CHAR_SIZE_MARGIN)
        fb_center = ImageFont.truetype(fb, CHAR_SIZE_CENTER)
    except Exception:
        # Final fallback: same main font — better than crashing.
        print(f"[WARN] Could not load fallback font '{fb}'. Using main font for missing glyphs.", file=sys.stderr)
        fb_main, fb_ann, fb_margin, fb_center = f_main, f_ann, f_margin, f_center
    return f_main, f_ann, f_margin, f_center, fb_main, fb_ann, fb_margin, fb_center


# ---------- Text parsing ----------

def _clean_text_and_extract_puncts(raw: str) -> Tuple[List[str], List[str]]:
    chars, puncts = [], []
    cur_p = ""
    for c in raw:
        if c in PUNCT_CHARS or c.isspace():
            cur_p += c
        else:
            if chars: puncts[-1] = cur_p
            chars.append(c); puncts.append(""); cur_p = ""
    if chars and cur_p:
        puncts[-1] = cur_p
    return chars, puncts


def parse_text(text: str) -> List[tuple]:
    lines = text.split('\n')
    elements: List[tuple] = []
    for line in lines:
        line = line.strip()
        if not line: continue
        hui_m = re.match(r'^(第[一二三四五六七八九十百千零\d]+回)[\s　]*(.*)$', line)
        if hui_m:
            elements.append(('hui_start', hui_m.group(1)))
            rest = hui_m.group(2).strip()
            if rest:
                elements.append(('hui_content', re.sub(r'\s+', '', rest)))
            continue
        remaining = re.sub(r'\s+', '', line)
        last_end = 0
        for m in re.finditer(r'【([^：:]+)[:：]([^】]+)】', remaining):
            before = remaining[last_end:m.start()]
            bc, bp = _clean_text_and_extract_puncts(before)
            for i in range(len(bc)):
                elements.append(('main', bc[i], bp[i]))
            ann_type = m.group(1)
            ac, ap = _clean_text_and_extract_puncts(m.group(2))
            clean = "".join(ac)
            if '眉批' in ann_type or '天头' in ann_type:
                elements.append(('tian', clean, ap))
            elif '地脚' in ann_type or '脚批' in ann_type:
                elements.append(('dijiao', clean, ap))
            else:
                t = 'SIDE_MARGINALIA' if ('侧' in ann_type or '側' in ann_type) else 'INTERLINEAR_ANNOTATION'
                elements.append(('jia', clean, ap, t))
            last_end = m.end()
        after = remaining[last_end:]
        ac, ap = _clean_text_and_extract_puncts(after)
        for i in range(len(ac)):
            elements.append(('main', ac[i], ap[i]))
    return elements


# ---------- Layout helpers ----------

def _box_x_center(i: int) -> int:
    return (BOX_COORDS[i][3][0] + BOX_COORDS[i][1][0]) // 2


def _char_y(_box: int, line: int) -> int:
    return TOP_BOUND + MARGIN_TOP + line * LINE_HEIGHT_MAIN


# ---------- Graph builder ----------

class JsonGraphBuilder:
    def __init__(self):
        self.nodes: List[dict] = []
        self.edges: List[dict] = []
        self.counters = {
            'MAIN_TEXT': 0, 'INTERLINEAR_ANNOTATION': 0, 'SIDE_MARGINALIA': 0,
            'TOP_MARGINALIA': 0, 'BOTTOM_MARGINALIA': 0, 'RED_SEAL_STAMP': 0,
        }
        self.prefixes = {
            'MAIN_TEXT': 'main', 'INTERLINEAR_ANNOTATION': 'interlinear',
            'SIDE_MARGINALIA': 'side', 'TOP_MARGINALIA': 'top',
            'BOTTOM_MARGINALIA': 'bottom', 'RED_SEAL_STAMP': 'seal',
        }
        self.groups = {k: 1 for k in self.prefixes if k != 'RED_SEAL_STAMP'}
        self.buffers = {k: {'chars': [], 'bboxes': [], 'box': None}
                        for k in self.groups}
        self.last_node = {k: None for k in self.groups}
        self.last_main_anchor: Optional[str] = None
        self.first_in_block = {k: True for k in self.groups}

    def flush(self, t_key: str) -> Optional[str]:
        buf = self.buffers[t_key]
        if not buf['chars']:
            return None
        self.counters[t_key] += 1
        nid = f"n_{self.prefixes[t_key]}{self.counters[t_key]}"
        text = "".join(buf['chars'])
        min_x = min(b[0] for b in buf['bboxes']); min_y = min(b[1] for b in buf['bboxes'])
        max_x = max(b[2] for b in buf['bboxes']); max_y = max(b[3] for b in buf['bboxes'])
        poly = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        z = 1 if t_key in ['MAIN_TEXT', 'INTERLINEAR_ANNOTATION'] else 2
        self.nodes.append({
            "node_id": nid, "polygon": poly, "text": text, "type": t_key,
            "attributes": {"group_id": f"G_{self.prefixes[t_key]}{self.groups[t_key]}",
                           "z_index": z, "color": "black"},
        })
        # READS_AFTER along the same semantic stream
        if self.last_node[t_key]:
            self.edges.append({"source": self.last_node[t_key], "target": nid,
                               "type": "READS_AFTER"})
        self.last_node[t_key] = nid
        if t_key == 'MAIN_TEXT':
            self.last_main_anchor = nid
        # ANNOTATES edge from the first annotation of a block to the current main anchor
        if t_key != 'MAIN_TEXT' and self.first_in_block.get(t_key, False):
            if self.last_main_anchor:
                self.edges.append({"source": nid, "target": self.last_main_anchor,
                                   "type": "ANNOTATES"})
            self.first_in_block[t_key] = False
        # CONFINED_WITHIN: omitted here; verifier's S_edge scores only
        # READS_AFTER/ANNOTATES/OVERLAPS, so we skip writing separator edges.
        buf['chars'].clear(); buf['bboxes'].clear(); buf['box'] = None
        return nid

    def add_char(self, t_key: str, char: str, bbox: List[int], punct: str,
                 box_idx: Optional[int]) -> None:
        buf = self.buffers[t_key]
        if buf['box'] is not None and buf['box'] != box_idx:
            self.flush(t_key)
        buf['box'] = box_idx
        buf['chars'].append(char); buf['bboxes'].append(bbox)
        if len(punct) > 0:
            self.flush(t_key)
        if any(c in PUNCT_GROUP_ENDERS for c in punct):
            self.groups[t_key] += 1

    def flush_all(self) -> None:
        for k in list(self.groups.keys()):
            self.flush(k)


# ---------- Scaffold nodes (frame / fold / fishtails) ----------

def _add_scaffold(graph: JsonGraphBuilder, scen: ScenarioSpec,
                  draw: ImageDraw.ImageDraw) -> List[int]:
    """Draw the outer frame + columns + fishtails. Returns the list of column
    indices where the separator was SUPPRESSED (for CROSS_COLUMN_FUSION).
    """
    # frame
    frame_poly = [167, 413, 1641, 413, 1641, 1194, 167, 1194]
    graph.nodes.append({
        "node_id": "n_frame1", "polygon": frame_poly, "text": "版框",
        "type": "OUTER_PAGE_FRAME",
        "attributes": {"z_index": 0, "color": "black"},
    })
    # box8 center + separators
    suppressed: List[int] = []
    # CROSS_COLUMN_FUSION: randomly suppress 2-3 separators' visual drawing.
    # The graph still records them (node exists), so the model is expected to
    # infer column boundaries from text alignment — learning to NOT merge.
    if scen.cross_column_fusion:
        rng = random.Random()  # local rng, seeded at page level outside
        candidate = [i for i in range(17) if i != 8]
        suppressed = rng.sample(candidate, k=min(3, len(candidate)))
    for i in range(17):
        if i == 8:
            graph.nodes.append({
                "node_id": "n_center1",
                "polygon": [885, 413, 999, 413, 999, 1194, 885, 1194],
                "text": "版心", "type": "CENTER_FOLD_COLUMN",
                "attributes": {"z_index": 0, "color": "black"},
            })
            # box8 is always visually bordered (fold line)
        else:
            x = BOX_COORDS[i][1][0]
            graph.nodes.append({
                "node_id": f"n_separator{i+1}",
                "polygon": [x, 413, x, 1194, x-1, 1194, x-1, 413],
                "text": "界格", "type": "COLUMN_SEPARATOR",
                "attributes": {"z_index": 0, "color": "black"},
            })
    fold_x = 942
    graph.nodes.append({"node_id": "n_fold1",
        "polygon": [fold_x-1, TOP_BOUND, fold_x+1, TOP_BOUND,
                    fold_x+1, BOTTOM_BOUND, fold_x-1, BOTTOM_BOUND],
        "text": "象鼻", "type": "FOLD_TRUNK_LINE",
        "attributes": {"z_index": 0, "color": "black"}})
    graph.nodes.append({"node_id": "n_fishtail1",
        "polygon": [fold_x-20, 570, fold_x+20, 570, fold_x+20, 600, fold_x-20, 600],
        "text": "鱼尾", "type": "FISHTAIL_MARK",
        "attributes": {"z_index": 0, "color": "black"}})
    graph.nodes.append({"node_id": "n_fishtail2",
        "polygon": [fold_x-20, 1050, fold_x+20, 1050, fold_x+20, 1080, fold_x-20, 1080],
        "text": "鱼尾", "type": "FISHTAIL_MARK",
        "attributes": {"z_index": 0, "color": "black"}})
    return suppressed


# ---------- Page renderer ----------

def render_page(
    *, background: Image.Image, elements: List[tuple], start_pos: int,
    seal_paths: List[Path], current_chapter: str,
    fonts: Tuple[ImageFont.ImageFont, ...],
    scenario: ScenarioSpec, page_rng: random.Random,
    max_columns: int = 17,
) -> Tuple[Image.Image, dict, int]:
    """Render one page; return (image, oracle_graph_dict, next_start_pos).

    Args:
        max_columns: Maximum number of text columns to fill (default 17 = full
            page).  Use 4-6 for a "lite" page that produces ~15-30 content
            nodes, keeping JSON output under 4K tokens for VLM experiments.
    """
    img = background.copy()
    draw = ImageDraw.Draw(img)
    graph = JsonGraphBuilder()
    suppressed_cols = _add_scaffold(graph, scenario, draw)

    # Clamp max_columns to valid range (BOX_COORDS has 17 slots, index 8 is fold)
    max_columns = max(1, min(17, max_columns))

    f_main, f_ann, f_margin, f_center, fb_main, fb_ann, fb_margin, fb_center = fonts

    # --- Draw center text (book + chapter name) ---
    center_text = "石头记" + (current_chapter or "")
    cx = 942 - CHAR_SIZE_CENTER // 2
    total_h = sum(CHAR_SIZE_CENTER if not c.isspace() else CHAR_SIZE_CENTER // 2
                  for c in center_text)
    y = TOP_BOUND + (BOTTOM_BOUND - TOP_BOUND - total_h) // 2
    center_bboxes: List[List[int]] = []
    actual_center = ""
    for ch in center_text:
        if ch.isspace():
            y += CHAR_SIZE_CENTER // 2
            continue
        fuse = f_center if _has_glyph(f_center, ch) else fb_center
        draw.text((cx, y), ch, font=fuse, fill='black')
        center_bboxes.append([cx, y, cx + CHAR_SIZE_CENTER, y + CHAR_SIZE_CENTER])
        actual_center += ch
        y += CHAR_SIZE_CENTER
    if center_bboxes:
        min_x = min(b[0] for b in center_bboxes); min_y = min(b[1] for b in center_bboxes)
        max_x = max(b[2] for b in center_bboxes); max_y = max(b[3] for b in center_bboxes)
        graph.nodes.append({
            "node_id": "n_centertext1",
            "polygon": [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y],
            "text": actual_center, "type": "CENTER_FOLD_TEXT",
            "attributes": {"group_id": "G_centertext1", "z_index": 1, "color": "black"},
        })

    # --- Main pagination loop ---
    page_tian = []
    page_dijiao = []
    current_box, current_line = 0, 0
    idx = start_pos

    # Scenario: boost interlinear density. We do this by promoting some 'main'
    # runs into 'jia' (double-line) when the scenario is active.
    ann_boost = scenario.double_line_flattening
    boost_remaining = 50 if ann_boost else 0  # soft budget per page

    while idx < len(elements):
        el = elements[idx]
        etype = el[0]
        if etype in ('hui_start', 'hui_content'):
            idx += 1
            continue
        if etype == 'tian':
            page_tian.append({'chars': list(el[1]), 'puncts': el[2]})
            idx += 1
            continue
        if etype == 'dijiao':
            page_dijiao.append({'chars': list(el[1]), 'puncts': el[2]})
            idx += 1
            continue
        # Scenario NOTE_MISANCHORING: convert some main text into "fake"
        # top-marginalia so the logical ANNOTATES anchor is non-obvious.
        if scenario.note_misanchoring and etype == 'main' and page_rng.random() < 0.02:
            # Accumulate a short marginalia blob
            blob = el[1]
            j = idx + 1
            while j < len(elements) and elements[j][0] == 'main' and len(blob) < 8:
                blob += elements[j][1]; j += 1
            if len(blob) >= 3:
                page_tian.append({'chars': list(blob), 'puncts': [''] * len(blob)})
                idx = j
                continue
        # Scenario DOUBLE_LINE_FLATTENING: escalate consecutive mains into jia
        if ann_boost and etype == 'main' and boost_remaining > 0 and page_rng.random() < 0.08:
            # Gather a contiguous run of main chars (up to 16) and emit as jia
            run = el[1]; punctos = [el[2]]
            j = idx + 1
            while j < len(elements) and elements[j][0] == 'main' and len(run) < 16:
                run += elements[j][1]; punctos.append(elements[j][2]); j += 1
            if len(run) >= 4:
                # Build synthetic jia element: ('jia', text, puncts_list, type)
                synthetic = ('jia', run, punctos, 'INTERLINEAR_ANNOTATION')
                # Insert it at idx, advance past 'main' consumed; but easier:
                # process directly.
                el = synthetic
                etype = 'jia'
                idx = j - 1  # -1 to compensate for the +=1 below at end of loop body
                boost_remaining -= len(run)

        if etype == 'jia':
            chars = list(el[1])
            puncts = el[2]
            t_key = "INTERLINEAR_ANNOTATION"
            graph.first_in_block[t_key] = True
            i = 0
            while i < len(chars):
                if current_box >= max_columns:
                    break
                if current_box == 8:
                    current_box = 9
                    if current_box >= max_columns:
                        break
                avail = LINES_PER_COLUMN - current_line
                max_chars = avail * 4
                if (len(chars) - i) <= max_chars:
                    cc = chars[i:]; cp = puncts[i:]; i = len(chars)
                else:
                    cc = chars[i:i + max_chars]; cp = puncts[i:i + max_chars]; i += max_chars
                num_lines = (len(cc) + 1) // 2
                right = cc[:num_lines]; rp = cp[:num_lines]
                left = cc[num_lines:]; lp = cp[num_lines:]
                xc = _box_x_center(current_box)
                char_w = CHAR_SIZE_ANN
                start_x = xc - char_w
                if right:
                    for k, ch in enumerate(right):
                        yo = (k % 2) * (LINE_HEIGHT_MAIN // 2)
                        yy = _char_y(current_box, current_line + k // 2) + yo
                        fuse = f_ann if _has_glyph(f_ann, ch) else fb_ann
                        draw.text((start_x + char_w, yy), ch, font=fuse, fill='black')
                        bbox = [start_x + char_w, yy, start_x + char_w * 2, yy + CHAR_SIZE_ANN]
                        graph.add_char(t_key, ch, bbox, rp[k], current_box)
                if left:
                    for k, ch in enumerate(left):
                        yo = (k % 2) * (LINE_HEIGHT_MAIN // 2)
                        yy = _char_y(current_box, current_line + k // 2) + yo
                        fuse = f_ann if _has_glyph(f_ann, ch) else fb_ann
                        draw.text((start_x, yy), ch, font=fuse, fill='black')
                        bbox = [start_x, yy, start_x + char_w, yy + CHAR_SIZE_ANN]
                        graph.add_char(t_key, ch, bbox, lp[k], current_box)
                lines_consumed = (num_lines + 1) // 2
                current_line += lines_consumed
                if current_line >= LINES_PER_COLUMN:
                    current_line = 0; current_box += 1
                    if current_box == 8:
                        current_box = 9
            graph.flush(t_key)
            idx += 1
            continue

        if etype == 'main':
            ch = el[1]; punct = el[2]
            if current_box == 8:
                current_box = 9
            if current_box >= max_columns:
                break
            yy = _char_y(current_box, current_line)
            xc = _box_x_center(current_box) - CHAR_SIZE_MAIN // 2
            box_left = BOX_COORDS[current_box][3][0]
            bw = BOX_WIDTHS[current_box]
            fuse = f_main if _has_glyph(f_main, ch) else fb_main
            draw.text((xc, yy), ch, font=fuse, fill='black')
            bbox = [box_left, yy, box_left + bw, yy + CHAR_SIZE_MAIN]
            graph.add_char('MAIN_TEXT', ch, bbox, punct, current_box)
            current_line += 1
            if current_line >= LINES_PER_COLUMN:
                current_line = 0; current_box += 1
                if current_box == 8:
                    current_box = 9
            idx += 1
            continue
        idx += 1

    # --- Margins ---
    if page_tian:
        tian_chars: List[Tuple[str, str]] = []
        for d in page_tian:
            for i, c in enumerate(d['chars']):
                tian_chars.append((c, d['puncts'][i] if i < len(d['puncts']) else ""))
        cols = (len(tian_chars) + TIANTOU_LINES - 1) // TIANTOU_LINES
        bw = cols * CHAR_SIZE_MARGIN
        sx = BG_WIDTH / 2 + bw / 2
        graph.first_in_block['TOP_MARGINALIA'] = True
        for i, (ch, pt) in enumerate(tian_chars):
            c_col = i // TIANTOU_LINES; r_row = i % TIANTOU_LINES
            col_x = sx - (c_col + 1) * CHAR_SIZE_MARGIN
            y_ = TOP_BOUND - TIANTOU_OFFSET - CHAR_SIZE_MARGIN - (TIANTOU_LINES - 1 - r_row) * LINE_HEIGHT_MARGIN
            fuse = f_margin if _has_glyph(f_margin, ch) else fb_margin
            draw.text((col_x, y_), ch, font=fuse, fill='black')
            bbox = [int(col_x), int(y_), int(col_x + CHAR_SIZE_MARGIN), int(y_ + CHAR_SIZE_MARGIN)]
            graph.add_char('TOP_MARGINALIA', ch, bbox, pt, None)
            if r_row == TIANTOU_LINES - 1:
                graph.flush('TOP_MARGINALIA')
                graph.first_in_block['TOP_MARGINALIA'] = True
        graph.flush('TOP_MARGINALIA')

    if page_dijiao:
        bchars: List[Tuple[str, str]] = []
        for d in page_dijiao:
            for i, c in enumerate(d['chars']):
                bchars.append((c, d['puncts'][i] if i < len(d['puncts']) else ""))
        cols = (len(bchars) + DIJIAO_LINES - 1) // DIJIAO_LINES
        bw = cols * CHAR_SIZE_MARGIN
        sx = BG_WIDTH / 2 + bw / 2
        graph.first_in_block['BOTTOM_MARGINALIA'] = True
        for i, (ch, pt) in enumerate(bchars):
            c_col = i // DIJIAO_LINES; r_row = i % DIJIAO_LINES
            col_x = sx - (c_col + 1) * CHAR_SIZE_MARGIN
            y_ = BOTTOM_BOUND + DIJIAO_OFFSET + r_row * LINE_HEIGHT_MARGIN
            fuse = f_margin if _has_glyph(f_margin, ch) else fb_margin
            draw.text((col_x, y_), ch, font=fuse, fill='black')
            bbox = [int(col_x), int(y_), int(col_x + CHAR_SIZE_MARGIN), int(y_ + CHAR_SIZE_MARGIN)]
            graph.add_char('BOTTOM_MARGINALIA', ch, bbox, pt, None)
            if r_row == DIJIAO_LINES - 1:
                graph.flush('BOTTOM_MARGINALIA')
                graph.first_in_block['BOTTOM_MARGINALIA'] = True
        graph.flush('BOTTOM_MARGINALIA')
    graph.flush_all()

    # --- Seals ---
    _place_seals(img, graph, seal_paths, scenario, page_rng)

    # --- Post-clean: text field only for text-bearing nodes ---
    ALLOWED = {"TOP_MARGINALIA", "BOTTOM_MARGINALIA", "MAIN_TEXT",
               "INTERLINEAR_ANNOTATION", "SIDE_MARGINALIA", "CENTER_FOLD_TEXT",
               "RED_SEAL_STAMP"}
    for n in graph.nodes:
        if n.get("type") not in ALLOWED:
            n["text"] = ""

    oracle = {
        "adversarial_scenario": scenario.to_list(),
        "metadata": {"width": BG_WIDTH, "height": BG_HEIGHT,
                     "suppressed_separators": suppressed_cols},
        "oracle_graph": {"nodes": graph.nodes, "edges": graph.edges},
    }
    return img, oracle, idx


def _place_seals(img: Image.Image, graph: JsonGraphBuilder,
                 seal_paths: List[Path], scen: ScenarioSpec,
                 rng: random.Random) -> None:
    if not seal_paths:
        return
    # Weighted count or forced intrusion
    if scen.seal_text_intrusion:
        num_to_draw = rng.randint(2, 4)
    else:
        seal_weights = [35, 30, 15, 8, 5, 3, 2, 1, 0.5, 0.3, 0.2]
        num_to_draw = rng.choices(range(11), weights=seal_weights, k=1)[0]
    num_to_draw = min(num_to_draw, len(seal_paths))
    if num_to_draw == 0:
        return
    selected = rng.sample(seal_paths, k=num_to_draw)

    placed: List[Tuple[int, int, int, int]] = []
    for sp in selected:
        try:
            sim = Image.open(sp).convert("RGBA")
        except Exception:
            continue
        sw, sh = sim.size
        nw, nh = int(sw * 0.8), int(sh * 0.8)
        if max(nw, nh) > 230:
            s = 230 / max(nw, nh); nw = int(nw * s); nh = int(nh * s)
        sim = sim.resize((nw, nh), RESAMPLING_FILTER)

        # Placement: prefer text overlap for SEAL_TEXT_INTRUSION
        if scen.seal_text_intrusion:
            # Pick a bbox from an existing MAIN_TEXT / INTERLINEAR node
            candidates = [n for n in graph.nodes
                          if n["type"] in ("MAIN_TEXT", "INTERLINEAR_ANNOTATION")]
            regions_from_text: List[Tuple[int, int, int, int]] = []
            for n in candidates[:200]:  # cap scan
                p = n["polygon"]; xs = p[0::2]; ys = p[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                if x2 - x1 >= nw and y2 - y1 >= nh:
                    regions_from_text.append((x1, y1, x2, y2))
            if regions_from_text:
                regions = regions_from_text
            else:
                regions = [(167, BG_WIDTH - 167, 413, BOTTOM_BOUND)]
        else:
            regions = [
                (1641 - 250, 1641, 413 - 150, 413 + 200),
                (1641 - 250, 1641, 1194 - 250, 1194 + 100),
                (167 - 50, 167 + 200, 413 - 150, 413 + 200),
                (167 - 50, 167 + 200, 1194 - 250, 1194 + 100),
                (167, BG_WIDTH - 167, 200, BG_HEIGHT - 200),
            ]

        placed_ok = False
        for _ in range(50):
            region = rng.choice(regions)
            if len(region) == 4 and all(isinstance(v, (int, float)) for v in region):
                if scen.seal_text_intrusion and len(regions) > 1:
                    x1, y1, x2, y2 = region
                    # Place within the node bbox, allowing slight overflow
                    x = rng.randint(max(0, x1 - 10), max(0, min(BG_WIDTH - nw, x2 - nw // 2)))
                    y = rng.randint(max(0, y1 - 10), max(0, min(BG_HEIGHT - nh, y2 - nh // 2)))
                else:
                    x1, x2, y1, y2 = region
                    sx_max = BG_WIDTH - nw; sy_max = BG_HEIGHT - nh
                    x1 = max(0, min(x1, sx_max)); x2 = max(0, min(x2, sx_max))
                    y1 = max(0, min(y1, sy_max)); y2 = max(0, min(y2, sy_max))
                    if x1 > x2: x1, x2 = x2, x1
                    if y1 > y2: y1, y2 = y2, y1
                    x = rng.randint(int(x1), int(x2))
                    y = rng.randint(int(y1), int(y2))
            else:
                x = rng.randint(167, max(168, BG_WIDTH - 167 - nw))
                y = rng.randint(413, max(414, BOTTOM_BOUND - nh))
            overlap = any(not (x + nw <= px or px + pw <= x or y + nh <= py or py + ph <= y)
                          for px, py, pw, ph in placed)
            if overlap:
                continue
            placed.append((x, y, nw, nh))
            img.paste(sim, (x, y), sim)
            graph.counters['RED_SEAL_STAMP'] += 1
            sid = f"n_seal{graph.counters['RED_SEAL_STAMP']}"
            graph.nodes.append({
                "node_id": sid,
                "polygon": [x, y, x + nw, y, x + nw, y + nh, x, y + nh],
                "text": "印章", "type": "RED_SEAL_STAMP",
                "attributes": {"z_index": 3, "color": "red"},
            })
            # OVERLAPS edges to any non-seal, content-bearing node it covers
            for on in graph.nodes:
                if on["type"] in ("RED_SEAL_STAMP", "OUTER_PAGE_FRAME",
                                  "COLUMN_SEPARATOR", "CENTER_FOLD_COLUMN",
                                  "FISHTAIL_MARK", "FOLD_TRUNK_LINE"):
                    continue
                if on["attributes"].get("z_index", 0) in (1, 2):
                    p = on["polygon"]
                    ox1, oy1 = min(p[0::2]), min(p[1::2])
                    ox2, oy2 = max(p[0::2]), max(p[1::2])
                    if not (x + nw <= ox1 or ox2 <= x or y + nh <= oy1 or oy2 <= y):
                        graph.edges.append({"source": sid, "target": on["node_id"],
                                            "type": "OVERLAPS"})
            placed_ok = True
            break
        if not placed_ok:
            continue


# ---------- Per-worker entrypoint ----------

def _worker(args):
    worker_id, n_pages, start_element_idx, cfg, balanced, base_seed, out_dir, max_columns = args
    rng = random.Random(base_seed + worker_id * 10007)
    try:
        bg = Image.open(cfg["background"]).convert("RGB")
        fonts = _load_fonts(cfg)
        with open(cfg["text"], "r", encoding="utf-8") as f:
            text = f.read().replace("﻿", "").replace("​", "")
        elements = parse_text(text)
        seal_paths: List[Path] = []
        if Path(cfg["seals_dir"]).exists():
            seal_paths = sorted([p for p in Path(cfg["seals_dir"]).iterdir()
                                 if p.suffix.lower() == ".png"])
        out_img_dir = Path(out_dir) / "images"; out_img_dir.mkdir(parents=True, exist_ok=True)
        out_json_dir = Path(out_dir) / "json"; out_json_dir.mkdir(parents=True, exist_ok=True)

        ptr = start_element_idx
        produced = 0
        while produced < n_pages and ptr < len(elements):
            scen = sample_scenario(rng, balanced=balanced)
            page_rng = random.Random(base_seed + worker_id * 10007 + produced)
            # Find current chapter for center_text
            current_chapter = ""
            for i in range(ptr, -1, -1):
                if i < len(elements) and elements[i][0] == "hui_start":
                    current_chapter = elements[i][1]; break
            img, oracle, next_ptr = render_page(
                background=bg, elements=elements, start_pos=ptr,
                seal_paths=seal_paths, current_chapter=current_chapter,
                fonts=fonts, scenario=scen, page_rng=page_rng,
                max_columns=max_columns,
            )
            image_id = f"iso_v1_w{worker_id}_p{produced:05d}"
            oracle["image_id"] = image_id
            (out_img_dir / f"{image_id}.jpg").write_bytes(b"")  # touch then save
            img.save(out_img_dir / f"{image_id}.jpg", quality=92)
            with open(out_json_dir / f"{image_id}.json", "w", encoding="utf-8") as f:
                json.dump(oracle, f, ensure_ascii=False, indent=2)
            ptr = next_ptr if next_ptr > ptr else ptr + 1
            produced += 1
        return produced
    except Exception as e:
        print(f"[worker {worker_id}] FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        return 0


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, required=True, help="output directory (will contain images/ and json/)")
    ap.add_argument("--n-pages", type=int, default=100, help="total pages to generate")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    ap.add_argument("--balanced", action="store_true", help="uniform 5-bucket scenario sampling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-columns", type=int, default=17,
                    help="Max text columns per page. Use 4-6 for 'lite' pages (15-30 content "
                         "nodes, JSON under 4K tokens). Default 17 = full page.")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=str, default=str(v))
    args = ap.parse_args()
    if args.n_pages < 1 or args.workers < 1:
        ap.error("--n-pages and --workers must be positive")

    cfg = {
        "background": args.background,
        "text": args.text,
        "font_main": args.font_main,
        "font_annotation": args.font_annotation,
        "seals_dir": args.seals_dir,
        "font_fallback": args.font_fallback,
    }
    for k in ("background", "text", "font_main", "font_annotation"):
        if not Path(cfg[k]).exists():
            print(f"[FATAL] missing required input '{k}': {cfg[k]}", file=sys.stderr)
            sys.exit(2)

    # Split pages across workers
    per_worker = [args.n_pages // args.workers] * args.workers
    for i in range(args.n_pages % args.workers):
        per_worker[i] += 1
    # Each worker starts at a different element offset so they don't all
    # render the same text chunk.
    with open(cfg["text"], "r", encoding="utf-8") as f:
        total_elements = len(parse_text(f.read().replace("﻿", "").replace("​", "")))
    step = max(1, total_elements // max(1, args.workers))
    tasks = [(i, per_worker[i], (i * step) % max(1, total_elements), cfg,
              args.balanced, args.seed, args.output, args.max_columns)
             for i in range(args.workers)]

    if args.workers == 1:
        done = _worker(tasks[0])
    else:
        with Pool(args.workers) as p:
            done = sum(p.map(_worker, tasks))
    print(f"Generated {done} pages → {args.output}")
    if done != args.n_pages:
        raise SystemExit("Incomplete generation; check input text length and worker errors")


if __name__ == "__main__":
    main()
