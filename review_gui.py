"""
라벨 검수 GUI - v3  (단일 통합 윈도우 · 다크 테마)
- 폴더 선택 → Cleanlab 품질 점수 계산, 캐시 저장
- 좌측: 파일 목록 (정렬·색상)
- 중앙: 이미지 캔버스 (박스 그리기/수정/삭제 — run.py 동일 조작)
- 우측: 품질 점수 · 박스 목록 · 클래스 분포
"""
import sys
import pickle
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QSlider, QSizePolicy, QListWidget, QListWidgetItem, QSplitter,
    QComboBox, QCheckBox, QScrollArea, QFrame, QLineEdit, QDialog,
    QDialogButtonBox
)
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPen, QBrush, QCursor, QFont
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QPointF, QRectF, QTimer

# ── 경로 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
CONFIG_DIR   = SCRIPT_DIR / "config"
REFS_DIR     = SCRIPT_DIR / "refs"
PREFS_PATH   = SCRIPT_DIR / ".review_prefs.json"
SOURCES_DIR  = SCRIPT_DIR / "sources"

# ── 기본 사용자 설정 ──────────────────────────────────────────────────────────
_DEFAULT_PREFS: dict = {
    "config_name": None,   # None → CONFIG_DIR 사용
    "auto_confirm": True,
    "dup_iou":  85,        # 슬라이더 정수값 (50-100)
    "dup_area": 75,        # 슬라이더 정수값 (30-100)
    "sort_key": "name_asc",
    "window_w": 1600,
    "window_h": 900,
    "show_minimap": True,
    "show_class_preview": True,
}

# ── 다크 테마 ─────────────────────────────────────────────────────────────────
DARK = """
QMainWindow, QWidget, QDialog {
    background: #1e1e1e;
    color: #dcdcdc;
}
QSplitter::handle { background: #3a3a3a; }
QListWidget {
    background: #252526;
    color: #dcdcdc;
    border: 1px solid #3a3a3a;
    font-size: 12px;
    outline: none;
}
QListWidget::item { padding: 3px 6px; border-bottom: 1px solid #2d2d2d; }
QListWidget::item:selected { background: #094771; color: #ffffff; }
QListWidget::item:hover    { background: #2a2d2e; }
QPushButton {
    background: #3c3c3c;
    color: #dcdcdc;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 12px;
}
QPushButton:hover   { background: #4a4a4a; }
QPushButton:pressed { background: #555; }
QPushButton:disabled { background: #2a2a2a; color: #555; border-color: #3a3a3a; }
QPushButton:checked { background: #1a7a1a; color: white; border-color: #1a9a1a; }
QLabel { color: #dcdcdc; }
QSlider::groove:horizontal { background: #3a3a3a; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #569cd6; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QComboBox {
    background: #3c3c3c; color: #dcdcdc;
    border: 1px solid #555; border-radius: 4px; padding: 2px 6px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #252526; color: #dcdcdc;
    selection-background-color: #094771;
}
QProgressBar {
    background: #2a2a2a; border: 1px solid #3a3a3a;
    color: #dcdcdc; text-align: center; border-radius: 3px;
}
QProgressBar::chunk { background: #0e639c; border-radius: 3px; }
QCheckBox { color: #dcdcdc; }
QCheckBox::indicator { width: 14px; height: 14px; }
QScrollBar:vertical   { background: #252526; width: 10px; }
QScrollBar:horizontal { background: #252526; height: 10px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #555; border-radius: 5px; min-height: 20px; min-width: 20px;
}
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #3a3a3a; }
QMessageBox { background: #1e1e1e; }
QToolTip { background: #333; color: #dcdcdc; border: 1px solid #555; }
"""


# ══════════════════════════════════════════════════════════════════════════════
# 유틸 함수 (원본 유지)
# ══════════════════════════════════════════════════════════════════════════════
def load_class_names_list(config_dir: Path = None):
    config_dir = config_dir or CONFIG_DIR
    names_path = config_dir / "obj.names"
    if names_path.exists():
        with open(names_path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    if DATASET_YAML.exists():
        with open(DATASET_YAML, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        names = d.get("names", {})
        if isinstance(names, dict):
            return [names.get(i, str(i)) for i in range(max(names.keys()) + 1)]
        return list(names)
    return []


def load_class_count(config_dir: Path = None):
    try:
        with open((config_dir or CONFIG_DIR) / "class_count.txt") as f:
            return int(f.read().strip())
    except Exception:
        return 31


def get_class_dist(txt_path: Path, class_names: list) -> str:
    if not txt_path.exists():
        return ""
    from collections import Counter
    cnt = Counter()
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) == 5:
            cnt[int(parts[0])] += 1
    if not cnt:
        return ""
    parts = []
    for cls in sorted(cnt):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        parts.append(f"{name}×{cnt[cls]}")
    return ", ".join(parts)


def find_config_dirs() -> list:
    return sorted([p for p in SCRIPT_DIR.iterdir()
                   if p.is_dir() and p.name.startswith("config")])


def get_class_ids(txt_path: Path) -> set:
    """txt 파일에서 클래스 ID 집합 반환."""
    ids = set()
    if txt_path.exists():
        for line in txt_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                try:
                    ids.add(int(parts[0]))
                except ValueError:
                    pass
    return ids


def get_color_for_class(cls: int) -> QColor:
    colors = [
        QColor(0, 255, 0),    QColor(255, 0, 0),    QColor(0, 0, 255),
        QColor(208, 177, 52), QColor(255, 0, 255),   QColor(0, 255, 255),
        QColor(255, 165, 0),  QColor(128, 0, 128),   QColor(0, 128, 128),
        QColor(128, 128, 0),
    ]
    return colors[cls % len(colors)]


def imread_unicode(path: str):
    try:
        with open(path, "rb") as f:
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_boxes_from_txt(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    if txt_path.exists():
        for line in txt_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = (cx - bw / 2) * img_w
                y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w
                y2 = (cy + bh / 2) * img_h
                boxes.append((cls, x1, y1, x2, y2))
    return boxes


def yolo_to_xyxy_norm(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


_DUP_IOU_THRESH  = 0.85
_DUP_AREA_RATIO  = 0.75


def _is_same_box_pair(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j,
                      iou_thresh=_DUP_IOU_THRESH,
                      area_ratio_thresh=_DUP_AREA_RATIO) -> bool:
    inter = (max(0.0, min(x2i, x2j) - max(x1i, x1j)) *
             max(0.0, min(y2i, y2j) - max(y1i, y1j)))
    if inter == 0:
        return False
    ai = (x2i - x1i) * (y2i - y1i)
    aj = (x2j - x1j) * (y2j - y1j)
    union = ai + aj - inter
    if union <= 0:
        return False
    iou = inter / union
    if iou < iou_thresh:
        return False
    if ai <= 0 or aj <= 0:
        return False
    return (min(ai, aj) / max(ai, aj)) >= area_ratio_thresh


def _merge_width_overlapping_boxes_once(boxes, thresh=0.5):
    """너비 50% 이상 겹치는 박스를 외곽 바운딩박스로 합침 — 1패스.
    boxes: list of (cls, x1, y1, x2, y2) pixel coords
    Returns: new list (merged where needed).
    """
    from collections import defaultdict
    n = len(boxes)
    if n < 2:
        return list(boxes)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        _, x1i, y1i, x2i, y2i = boxes[i]
        area_i = (x2i - x1i) * (y2i - y1i)
        for j in range(i + 1, n):
            _, x1j, y1j, x2j, y2j = boxes[j]
            area_j = (x2j - x1j) * (y2j - y1j)
            inter_x = max(0.0, min(x2i, x2j) - max(x1i, x1j))
            inter_y = max(0.0, min(y2i, y2j) - max(y1i, y1j))
            inter_area = inter_x * inter_y
            min_area = min(area_i, area_j)
            if min_area > 0 and inter_area / min_area >= thresh:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    result = []
    for members in groups.values():
        if len(members) == 1:
            result.append(boxes[members[0]])
        else:
            best_cls, best_area = boxes[members[0]][0], 0
            x1s, y1s, x2s, y2s = [], [], [], []
            for idx in members:
                cls, x1, y1, x2, y2 = boxes[idx]
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area, best_cls = area, cls
                x1s.append(x1); y1s.append(y1)
                x2s.append(x2); y2s.append(y2)
            result.append((best_cls, min(x1s), min(y1s), max(x2s), max(y2s)))
    return result


def _merge_width_overlapping_boxes(boxes, thresh=0.5):
    """통합 후 새로 생긴 큰 박스가 다시 겹칠 수 있으므로,
    변화가 없을 때까지 반복 적용한다."""
    current = list(boxes)
    while True:
        next_ = _merge_width_overlapping_boxes_once(current, thresh)
        if len(next_) == len(current):
            return next_
        current = next_


def _merge_width_overlapping_txt(txt_path: Path, thresh=0.5) -> int:
    """YOLO txt 파일에서 너비 겹침 박스를 직접 병합 저장.
    통합 후 새로 커진 박스가 다시 겹칠 수 있으므로 변화가 없을 때까지 반복.
    반환: 원본 대비 줄어든 박스 수."""
    if not txt_path.exists():
        return 0

    def _parse(path):
        boxes = []
        for line in path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                boxes.append((cls, cx - w/2, cy - h/2, cx + w/2, cy + h/2))
        return boxes

    def _one_pass(raw):
        from collections import defaultdict
        n = len(raw)
        if n < 2:
            return raw, 0
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            _, x1i, y1i, x2i, y2i = raw[i]
            area_i = (x2i - x1i) * (y2i - y1i)
            for j in range(i + 1, n):
                _, x1j, y1j, x2j, y2j = raw[j]
                area_j = (x2j - x1j) * (y2j - y1j)
                inter_x = max(0.0, min(x2i, x2j) - max(x1i, x1j))
                inter_y = max(0.0, min(y2i, y2j) - max(y1i, y1j))
                inter_area = inter_x * inter_y
                min_area = min(area_i, area_j)
                if min_area > 0 and inter_area / min_area >= thresh:
                    pi, pj = find(i), find(j)
                    if pi != pj:
                        parent[pi] = pj

        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        result = []
        for members in groups.values():
            if len(members) == 1:
                result.append(raw[members[0]])
            else:
                best_cls, best_area = raw[members[0]][0], 0
                x1s, y1s, x2s, y2s = [], [], [], []
                for idx in members:
                    cls, x1, y1, x2, y2 = raw[idx]
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area, best_cls = area, cls
                    x1s.append(x1); y1s.append(y1)
                    x2s.append(x2); y2s.append(y2)
                result.append((best_cls, min(x1s), min(y1s), max(x2s), max(y2s)))
        return result, n - len(result)

    raw = _parse(txt_path)
    original_count = len(raw)
    while True:
        raw, reduced = _one_pass(raw)
        if reduced == 0:
            break

    final_count = len(raw)
    if final_count == original_count:
        return 0  # 변경 없음

    lines = []
    for cls, x1, y1, x2, y2 in raw:
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        w  = x2 - x1;       h  = y2 - y1
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return original_count - final_count


def _remove_border_small_boxes(boxes, img_w, img_h):
    """이미지 테두리에 걸친 박스 중 평균 면적의 절반 미만인 것 제거.
    boxes: list of (cls, x1, y1, x2, y2) pixel coords
    Returns: (new_boxes, removed_count)
    """
    EPS = 1.0
    border_idx = set()
    inner_areas = []
    for i, (_, x1, y1, x2, y2) in enumerate(boxes):
        on_border = (x1 <= EPS or y1 <= EPS or
                     x2 >= img_w - EPS or y2 >= img_h - EPS)
        if on_border:
            border_idx.add(i)
        else:
            inner_areas.append((x2 - x1) * (y2 - y1))

    if not border_idx:
        return list(boxes), 0

    if inner_areas:
        avg_area = sum(inner_areas) / len(inner_areas)
    else:
        # 내부 박스가 없으면 전체 평균 사용
        all_areas = [(b[3]-b[1])*(b[4]-b[2]) for b in boxes]
        avg_area = sum(all_areas) / len(all_areas) if all_areas else 0

    threshold = avg_area * 0.5
    to_remove = {i for i in border_idx
                 if (boxes[i][3]-boxes[i][1])*(boxes[i][4]-boxes[i][2]) < threshold}

    if not to_remove:
        return list(boxes), 0

    new_boxes = [b for i, b in enumerate(boxes) if i not in to_remove]
    return new_boxes, len(to_remove)


def _remove_border_small_boxes_txt(txt_path: Path) -> int:
    """YOLO txt 파일에서 테두리 소형 박스 제거. 제거된 박스 수 반환."""
    if not txt_path.exists():
        return 0
    raw = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            raw.append((cls, cx - w/2, cy - h/2, cx + w/2, cy + h/2))

    merged, removed = _remove_border_small_boxes(raw, 1.0, 1.0)
    if not removed:
        return 0

    lines = []
    for cls, x1, y1, x2, y2 in merged:
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        w  = x2 - x1;       h  = y2 - y1
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return removed


def compute_has_duplicate(txt_path: Path,
                          iou_thresh=_DUP_IOU_THRESH,
                          area_thresh=_DUP_AREA_RATIO) -> bool:
    boxes = []
    if txt_path.exists():
        for line in txt_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                boxes.append((c, cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    n = len(boxes)
    for i in range(n):
        ci, x1i, y1i, x2i, y2i = boxes[i]
        for j in range(i + 1, n):
            cj, x1j, y1j, x2j, y2j = boxes[j]
            if ci == cj:
                continue
            if _is_same_box_pair(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j,
                                  iou_thresh, area_thresh):
                return True
    return False


def remove_duplicates_from_txt(txt_path: Path) -> int:
    if not txt_path.exists():
        return 0
    lines = [l for l in txt_path.read_text().strip().splitlines() if l.strip()]
    raw = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            c = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            raw.append((c, cx, cy, w, h))
    n = len(raw)
    if n == 0:
        return 0
    xyxy = [(cx - w/2, cy - h/2, cx + w/2, cy + h/2) for _, cx, cy, w, h in raw]
    to_remove = set()
    for i in range(n):
        if i in to_remove:
            continue
        ci = raw[i][0]
        x1i, y1i, x2i, y2i = xyxy[i]
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            cj = raw[j][0]
            if ci == cj:
                continue
            x1j, y1j, x2j, y2j = xyxy[j]
            if _is_same_box_pair(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j):
                to_remove.add(j)
    if not to_remove:
        return 0
    kept = [raw[i] for i in range(n) if i not in to_remove]
    with open(txt_path, "w") as f:
        for c, cx, cy, w, h in kept:
            f.write(f"{c} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n")
    return len(to_remove)


# ══════════════════════════════════════════════════════════════════════════════
# 캐시
# ══════════════════════════════════════════════════════════════════════════════
CACHE_VERSION = 2


def _cache_path(folder: Path) -> Path:
    return folder / ".review_cache.pkl"


def save_cache(folder: Path, results: list) -> tuple:
    try:
        cp = _cache_path(folder)
        data = {
            "version": CACHE_VERSION,
            "folder":  str(folder.resolve()),
            "ts":      time.time(),
            "results": results,
        }
        with open(cp, "wb") as f:
            pickle.dump(data, f)
        size_kb = cp.stat().st_size / 1024
        return True, f"캐시 저장 완료 ({size_kb:.0f} KB)"
    except Exception as e:
        return False, f"캐시 저장 실패: {e}"


def load_cache(folder: Path):
    cp = _cache_path(folder)
    if not cp.exists():
        return None, "캐시 없음"
    try:
        with open(cp, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != CACHE_VERSION:
            return None, "캐시 버전 불일치"
        if data.get("folder") != str(folder.resolve()):
            return None, "폴더 불일치"
        results = data.get("results") or []
        return results, f"{len(results)}개 로드"
    except Exception as e:
        return None, f"캐시 손상: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TXT 압축 워커
# ══════════════════════════════════════════════════════════════════════════════
class ZipWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str, int)   # (zip 경로, 파일 수)
    error    = pyqtSignal(str)

    def __init__(self, src_folder: Path, zip_path: Path):
        super().__init__()
        self.src_folder = src_folder
        self.zip_path   = zip_path

    def run(self):
        try:
            txt_files = sorted(self.src_folder.glob("*.txt"))
            if not txt_files:
                self.error.emit("폴더 내에 .txt 파일이 없습니다.")
                return
            total = len(txt_files)
            with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, f in enumerate(txt_files, 1):
                    zf.write(f, f.name)
                    self.progress.emit(i, total)
            self.finished.emit(str(self.zip_path), total)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# 검수 워커
# ══════════════════════════════════════════════════════════════════════════════
class ReviewWorker(QThread):
    progress   = pyqtSignal(int, int)
    status_msg = pyqtSignal(str)
    finished   = pyqtSignal(list)
    error      = pyqtSignal(str)

    def __init__(self, folder: Path, file_filter=None, auto_confirm: bool = True):
        super().__init__()
        self.folder       = folder
        self.file_filter  = file_filter
        self.auto_confirm = auto_confirm

    def run(self):
        try:
            img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            pairs = []
            for p in sorted(self.folder.iterdir()):
                if p.suffix.lower() in img_exts:
                    txt = p.with_suffix(".txt")
                    if txt.exists():
                        if self.file_filter is None or p.name in self.file_filter:
                            pairs.append((p, txt))
            if not pairs:
                self.error.emit("처리할 이미지+txt 쌍이 없습니다.")
                return

            total = len(pairs)
            label = "수정된" if self.file_filter else "전체"
            self.status_msg.emit(f"{label} {total}개 파일 스캔 중...")

            out = []
            for i, (img_path, txt_path) in enumerate(pairs):
                has_dup   = compute_has_duplicate(txt_path)
                box_count = 0
                if txt_path.exists():
                    box_count = sum(
                        1 for ln in txt_path.read_text().strip().splitlines()
                        if len(ln.split()) == 5
                    )
                out.append({
                    "img_path":      img_path,
                    "txt_path":      txt_path,
                    "is_issue":      has_dup,
                    "has_duplicate": has_dup,
                    "box_count":     box_count,
                    "confirmed":     self.auto_confirm and not has_dup,
                })
                self.progress.emit(i + 1, total)
            self.finished.emit(out)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# 캔버스 레이블 (마우스/휠 이벤트 → ReviewApp 으로 위임)
# ══════════════════════════════════════════════════════════════════════════════
class CanvasLabel(QLabel):
    def __init__(self, app: "ReviewApp"):
        super().__init__()
        self._app = app
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #2a2a2a;")
        self.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, e):   self._app._canvas_mouse_press(e);   e.accept()
    def mouseMoveEvent(self, e):    self._app._canvas_mouse_move(e);    e.accept()
    def mouseReleaseEvent(self, e): self._app._canvas_mouse_release(e); e.accept()
    def wheelEvent(self, e):        self._app._canvas_wheel(e);          e.accept()
    def keyPressEvent(self, e):     self._app._handle_key(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, self._app._on_canvas_resize)


# ══════════════════════════════════════════════════════════════════════════════
# 박스 수 인덱스 바 (iOS 연락처 이니셜 스타일)
# ══════════════════════════════════════════════════════════════════════════════
class _BoxIndexBar(QWidget):
    """파일 리스트 옆에 붙는 박스 수 퀵점프 인덱스 바."""
    jumpRequested = pyqtSignal(int)   # 스크롤할 row 인덱스

    _BG        = QColor(28, 28, 28)
    _FG        = QColor(140, 140, 140)
    _HOVER_BG  = QColor(9, 71, 113)
    _HOVER_FG  = QColor(255, 255, 255)
    _ACTIVE_FG = QColor(100, 180, 255)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(26)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self._labels: list[tuple[str, int]] = []   # (표시텍스트, row)
        self._hover_i  = -1
        self._active_i = -1

    # ── 외부 API ────────────────────────────────────────────────────────────
    def set_labels(self, labels: list[tuple[str, int]]):
        self._labels   = labels
        self._hover_i  = -1
        self._active_i = -1
        self.update()

    def set_active(self, row: int):
        """현재 선택된 row에 해당하는 버킷 하이라이트."""
        self._active_i = -1
        for i, (_, r) in enumerate(self._labels):
            if r <= row:
                self._active_i = i
            else:
                break
        self.update()

    # ── 내부 헬퍼 ───────────────────────────────────────────────────────────
    def _idx_at(self, y: int) -> int:
        n = len(self._labels)
        if n == 0:
            return -1
        slot = self.height() / n
        return max(0, min(n - 1, int(y / slot)))

    def _row_rect(self, i: int):
        from PyQt5.QtCore import QRect
        n = len(self._labels)
        if n == 0:
            return None
        slot = self.height() / n
        return QRect(0, int(i * slot), self.width(), int(slot) + 1)

    # ── 그리기 ──────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        from PyQt5.QtGui import QPainter, QFont
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self._BG)

        n = len(self._labels)
        if n == 0:
            p.end()
            return

        font = QFont("Arial", 7, QFont.Bold)
        p.setFont(font)

        for i, (label, _) in enumerate(self._labels):
            r = self._row_rect(i)
            if i == self._hover_i:
                p.fillRect(r, self._HOVER_BG)
                p.setPen(self._HOVER_FG)
            elif i == self._active_i:
                p.setPen(self._ACTIVE_FG)
            else:
                p.setPen(self._FG)
            p.drawText(r, Qt.AlignCenter, label)

        p.end()

    # ── 이벤트 ──────────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        i = self._idx_at(e.pos().y())
        if 0 <= i < len(self._labels):
            self.jumpRequested.emit(self._labels[i][1])
        e.accept()

    def mouseMoveEvent(self, e):
        i = self._idx_at(e.pos().y())
        if e.buttons() & Qt.LeftButton and 0 <= i < len(self._labels):
            self.jumpRequested.emit(self._labels[i][1])
        if i != self._hover_i:
            self._hover_i = i
            self.update()
        e.accept()

    def leaveEvent(self, _):
        self._hover_i = -1
        self.update()


# ══════════════════════════════════════════════════════════════════════════════
# sources 폴더 선택 다이얼로그
# ══════════════════════════════════════════════════════════════════════════════
class SourceFolderDialog(QDialog):
    """SOURCES_DIR 안의 하위 폴더를 목록으로 보여주고 선택하게 한다."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("검수할 폴더 선택")
        self.setMinimumSize(480, 400)
        self.selected_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 안내 레이블
        lbl = QLabel(f"📁  {SOURCES_DIR}")
        lbl.setStyleSheet("font-size:11px; color:#aaaaaa; padding:4px 0;")
        layout.addWidget(lbl)

        # 폴더 목록
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget{background:#1e1e1e;border:1px solid #3a3a3a;font-size:13px;}"
            "QListWidget::item{padding:5px 8px;border-bottom:1px solid #2a2a2a;}"
            "QListWidget::item:selected{background:#094771;color:#fff;}"
            "QListWidget::item:hover{background:#2a2d2e;}"
        )
        self._populate()
        self.list_widget.itemDoubleClicked.connect(self._accept)
        layout.addWidget(self.list_widget)

        # 버튼
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._accept)
        btn_box.rejected.connect(self.reject)
        btn_other = QPushButton("다른 폴더…")
        btn_other.clicked.connect(self._browse_other)
        btn_box.addButton(btn_other, QDialogButtonBox.ActionRole)
        layout.addWidget(btn_box)

    def _populate(self):
        self.list_widget.clear()
        if SOURCES_DIR.exists():
            folders = sorted(
                [p for p in SOURCES_DIR.iterdir() if p.is_dir()],
                key=lambda p: p.name
            )
            for f in folders:
                self.list_widget.addItem(f.name)
        else:
            self.list_widget.addItem("⚠ sources 폴더를 찾을 수 없습니다")

    def _accept(self):
        item = self.list_widget.currentItem()
        if item is None:
            return
        self.selected_path = SOURCES_DIR / item.text()
        self.accept()

    def _browse_other(self):
        folder = QFileDialog.getExistingDirectory(self, "폴더 직접 선택")
        if folder:
            self.selected_path = Path(folder).resolve()
            self.accept()


# ══════════════════════════════════════════════════════════════════════════════
# 메인 통합 창
# ══════════════════════════════════════════════════════════════════════════════
class ReviewApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("라벨 검수 도구 v3")
        self.resize(1600, 900)
        self.setMinimumSize(1100, 650)

        # ── MainWindow 상태 ──────────────────────────────────────────────────
        self.results           = []
        self.worker            = None
        self.selected_folder   = None
        self.current_config_dir= CONFIG_DIR
        self._num_classes      = load_class_count(CONFIG_DIR)
        self._sort_key         = "name_asc"    # 현재 정렬 기준
        self._filter_classes: set = set()      # 클래스 ID 필터 (빈 set = 전체 표시)
        self._filter_cls23_eq7: bool = False   # (cls2+cls3)==7 필터
        self._dup_iou   = _DUP_IOU_THRESH      # 중복 판단 IoU 임계값
        self._dup_area  = _DUP_AREA_RATIO      # 중복 판단 면적 비율

        # ── ImageReviewWindow 상태 ───────────────────────────────────────────
        self.cur_idx           = 0
        self.class_names       = load_class_names_list(CONFIG_DIR)
        self.class_count       = load_class_count(CONFIG_DIR)
        self.boxes             = []
        self.zoom              = 1.0
        self.offset            = QPoint(0, 0)
        self.w = self.h        = 1
        self.drawing           = False
        self.start_point       = None
        self.end_point         = None
        self.current_class     = 0
        self.selected_boxes    = set()
        self.selecting         = False
        self.select_start      = None
        self.select_end        = None
        self.dragging_box_index= None
        self.drag_start_pos    = None
        self.resizing_box_index= None
        self.resize_handle     = None
        self.panning           = False
        self.pan_start_pos     = None
        self.HANDLE_SIZE       = 8
        self.minimap_size      = 180
        self.show_minimap      = True
        self.show_class_preview= True
        self.class_preview_pixmap = None
        self.modified_boxes    = set()
        self.undo_stack        = []
        self.redo_stack        = []
        self._active_edit_snapshot = None

        self._build_ui()
        self.setStyleSheet(DARK)
        self._load_prefs()

    # ══════════════════════════════════════════════════════════════════════════
    # UI 구성
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 공통 위젯 헬퍼 ──────────────────────────────────────────────────
        def _btn(text, color=None, h=26, w=None):
            b = QPushButton(text)
            b.setFixedHeight(h)
            if w: b.setFixedWidth(w)
            if color:
                b.setStyleSheet(
                    f"QPushButton{{background:{color};color:#fff;border-radius:4px;"
                    f"font-weight:bold;border:none;padding:2px 8px;}}"
                    f"QPushButton:hover{{background:{color}cc;}}"
                    f"QPushButton:disabled{{background:#2a2a2a;color:#555;"
                    f"border:1px solid #3a3a3a;}}")
            return b

        def _vsep():
            s = QFrame(); s.setFrameShape(QFrame.VLine)
            s.setStyleSheet("color:#444;")
            return s

        def _hsep():
            s = QFrame(); s.setFrameShape(QFrame.HLine)
            s.setStyleSheet("color:#3a3a3a;")
            return s

        def _sec(text):
            l = QLabel(text)
            l.setStyleSheet(
                "background:#333;color:#dcdcdc;font-weight:bold;"
                "padding:3px 6px;border-radius:3px;font-size:11px;")
            return l

        # ── 상단 툴바 (2행) ──────────────────────────────────────────────────
        tb = QWidget()
        tb.setStyleSheet("background:#252526;border-bottom:1px solid #3a3a3a;")
        tbl = QVBoxLayout(tb)
        tbl.setContentsMargins(8, 5, 8, 5)
        tbl.setSpacing(4)

        # 행 1: 폴더 · 검수 · 확인 · 편집
        r1 = QHBoxLayout(); r1.setSpacing(4)
        self.folder_label = QLabel("폴더를 선택하세요")
        self.folder_label.setStyleSheet(
            "padding:2px 8px;border:1px solid #3a3a3a;border-radius:4px;"
            "background:#1e1e1e;color:#aaa;")
        self.folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_sel              = _btn("📁 폴더 선택")
        self.btn_review      = _btn("🔍 전부 검수",    "#0e639c")
        self.btn_partial     = _btn("🔄 재검수",        "#b85c00")
        self.btn_rev_sel     = _btn("☑ 선택 검수",     "#7d3c98")
        self.btn_confirm_all   = _btn("✅ 전체 확인",   "#218838")
        self.btn_confirm_sel   = _btn("☑ 선택 확인",   "#4cae4c")
        self.btn_unconfirm_all = _btn("✖ 확인 해제",   "#5a6268")
        self.btn_remove_dups   = _btn("🗑 중복 제거",   "#c0392b")
        self.btn_merge_all     = _btn("🔗 전체 통합",   "#1a6b5a")
        self.btn_rm_border_all = _btn("✂ 테두리 제거",  "#6b3a1a")

        for b in (self.btn_review, self.btn_partial, self.btn_rev_sel,
                  self.btn_confirm_all, self.btn_confirm_sel, self.btn_unconfirm_all,
                  self.btn_remove_dups, self.btn_merge_all, self.btn_rm_border_all):
            b.setEnabled(False)

        btn_sel.clicked.connect(self._select_folder)
        self.btn_review.clicked.connect(self._start_review)
        self.btn_partial.clicked.connect(self._start_partial_review)
        self.btn_rev_sel.clicked.connect(self._review_selected_items)
        self.btn_confirm_all.clicked.connect(self._confirm_all)
        self.btn_confirm_sel.clicked.connect(self._confirm_selected_items)
        self.btn_unconfirm_all.clicked.connect(self._unconfirm_all)
        self.btn_remove_dups.clicked.connect(self._remove_all_duplicates)
        self.btn_merge_all.clicked.connect(self._merge_all_overlapping)
        self.btn_rm_border_all.clicked.connect(self._remove_border_boxes_all)

        for w in [self.folder_label, btn_sel, _vsep(),
                  self.btn_review, self.btn_partial, self.btn_rev_sel, _vsep(),
                  self.btn_confirm_all, self.btn_confirm_sel, self.btn_unconfirm_all, _vsep(),
                  self.btn_remove_dups, self.btn_merge_all, self.btn_rm_border_all]:
            r1.addWidget(w)
        tbl.addLayout(r1)

        # 행 2: Config · 옵션 · 중복임계값 · 압축 · 상태
        r2 = QHBoxLayout(); r2.setSpacing(5)

        cfg_lbl = QLabel("Config:")
        cfg_lbl.setStyleSheet("color:#aaa;font-size:11px;")
        self.config_combo = QComboBox()
        self.config_combo.setFixedHeight(22)
        self.config_combo.setMinimumWidth(110)
        for cdir in find_config_dirs():
            self.config_combo.addItem(cdir.name, userData=cdir)
        default_idx = next((i for i in range(self.config_combo.count())
                            if self.config_combo.itemData(i) == CONFIG_DIR), 0)
        self.config_combo.setCurrentIndex(default_idx)
        self.config_combo.currentIndexChanged.connect(self._on_config_changed)

        self.chk_auto_confirm = QCheckBox("검수 후 자동 확인")
        self.chk_auto_confirm.setChecked(True)
        self.chk_auto_confirm.setStyleSheet("font-size:11px;")
        self.chk_auto_confirm.stateChanged.connect(lambda _: self._save_prefs())

        r2.addWidget(cfg_lbl)
        r2.addWidget(self.config_combo)
        r2.addWidget(_vsep())
        r2.addWidget(self.chk_auto_confirm)
        r2.addWidget(_vsep())

        dup_lbl = QLabel("🔁 중복:")
        dup_lbl.setStyleSheet("color:#aaa;font-size:11px;")
        r2.addWidget(dup_lbl)
        self._dup_sliders: dict = {}   # attr → (QSlider, val_QLabel)
        for attr, label, default, lo, hi in [
            ("_dup_iou",  "IoU",  85, 50, 100),
            ("_dup_area", "면적", 75, 30, 100),
        ]:
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet("color:#ccc;font-size:11px;")
            sl  = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi); sl.setValue(default)
            sl.setFixedWidth(76); sl.setFixedHeight(16)
            val = QLabel(f"{default/100:.2f}")
            val.setFixedWidth(32)
            val.setStyleSheet("color:#FFD700;font-size:11px;")
            self._dup_sliders[attr] = (sl, val)
            def _make_dup_handler(a=attr, vl=val):
                def _h(v):
                    setattr(self, a, v/100); vl.setText(f"{v/100:.2f}")
                    for r in self.results: r.pop("has_duplicate", None)
                    self._populate_list()
                    self._save_prefs()
                return _h
            sl.valueChanged.connect(_make_dup_handler())
            r2.addWidget(lbl); r2.addWidget(sl); r2.addWidget(val)

        r2.addWidget(_vsep())

        zip_lbl = QLabel("📦 압축:")
        zip_lbl.setStyleSheet("color:#aaa;font-size:11px;")
        self.zip_name_edit = QLineEdit()
        self.zip_name_edit.setPlaceholderText("파일명")
        self.zip_name_edit.setFixedHeight(22)
        self.zip_name_edit.setFixedWidth(110)
        self.zip_name_edit.setStyleSheet(
            "background:#1e1e1e;color:#dcdcdc;border:1px solid #3a3a3a;"
            "border-radius:3px;padding:1px 4px;font-size:11px;")
        self.btn_zip_txt = _btn("압축", "#5a3e8a", h=22, w=46)
        self.btn_zip_txt.setEnabled(False)
        self.btn_zip_txt.clicked.connect(self._zip_txt)
        r2.addWidget(zip_lbl)
        r2.addWidget(self.zip_name_edit)
        r2.addWidget(self.btn_zip_txt)
        r2.addWidget(_vsep())

        btn_reset_prefs = QPushButton("⟳ 설정 초기화")
        btn_reset_prefs.setFixedHeight(22)
        btn_reset_prefs.setStyleSheet(
            "QPushButton{background:#3a3a3a;color:#aaa;border:1px solid #555;"
            "border-radius:3px;font-size:10px;padding:0 6px;}"
            "QPushButton:hover{background:#555;color:#fff;}")
        btn_reset_prefs.clicked.connect(self._reset_prefs)
        r2.addWidget(btn_reset_prefs)
        r2.addWidget(_vsep())

        self.status_label = QLabel("준비")
        self.status_label.setStyleSheet("color:#aaa;font-size:11px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setVisible(False)
        r2.addWidget(self.status_label)
        r2.addWidget(self.progress_bar)
        r2.addStretch()
        tbl.addLayout(r2)

        root.addWidget(tb)

        # ── 3분할 스플리터 ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        root.addWidget(splitter, stretch=1)

        # ════════════════════════════════════════════════════════════════════
        # 좌측 패널: 파일 목록
        # ════════════════════════════════════════════════════════════════════
        left = QWidget()
        left.setMinimumWidth(200)
        left.setMaximumWidth(320)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.setSpacing(3)

        # 정렬
        sort_row = QHBoxLayout(); sort_row.setSpacing(4)
        sort_lbl = QLabel("정렬:")
        sort_lbl.setStyleSheet("color:#aaa;font-size:11px;")
        self.sort_combo = QComboBox()
        self.sort_combo.setFixedHeight(22)
        for k, v in [
            ("name_asc",   "파일명순"),
            ("dup_first",  "중복 우선"),
            ("unconfirmed","미확인 우선"),
            ("box_desc",   "박스 많은순"),
            ("box_asc",    "박스 적은순"),
        ]:
            self.sort_combo.addItem(v, userData=k)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        sort_row.addWidget(sort_lbl)
        sort_row.addWidget(self.sort_combo, stretch=1)
        lv.addLayout(sort_row)

        # 클래스 필터
        filter_hdr = QHBoxLayout(); filter_hdr.setSpacing(4)
        filter_ttl = QLabel("🔍 클래스 필터")
        filter_ttl.setStyleSheet("color:#bbb;font-size:11px;font-weight:bold;")
        self.btn_clear_filter = QPushButton("초기화")
        self.btn_clear_filter.setFixedSize(46, 20)
        self.btn_clear_filter.setStyleSheet(
            "QPushButton{background:#3a3a3a;color:#aaa;border:1px solid #555;"
            "border-radius:3px;font-size:10px;padding:0;}"
            "QPushButton:hover{background:#555;color:#fff;}")
        self.btn_clear_filter.clicked.connect(self._clear_filter)
        filter_hdr.addWidget(filter_ttl)
        filter_hdr.addStretch()
        filter_hdr.addWidget(self.btn_clear_filter)
        lv.addLayout(filter_hdr)

        filter_input_row = QHBoxLayout(); filter_input_row.setSpacing(3)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("클래스 ID (예: 0)")
        self.filter_input.setFixedHeight(22)
        self.filter_input.setStyleSheet(
            "QLineEdit{background:#2a2a2a;color:#dcdcdc;border:1px solid #555;"
            "border-radius:3px;padding:1px 4px;font-size:11px;}")
        self.filter_input.returnPressed.connect(self._add_filter_class)
        btn_add_filter = QPushButton("추가")
        btn_add_filter.setFixedSize(38, 22)
        btn_add_filter.setStyleSheet(
            "QPushButton{background:#0e639c;color:#fff;border:none;"
            "border-radius:3px;font-size:11px;}"
            "QPushButton:hover{background:#1177bb;}")
        btn_add_filter.clicked.connect(self._add_filter_class)
        filter_input_row.addWidget(self.filter_input)
        filter_input_row.addWidget(btn_add_filter)
        lv.addLayout(filter_input_row)

        self.filter_tag_list = QListWidget()
        self.filter_tag_list.setMaximumHeight(80)
        self.filter_tag_list.setStyleSheet(
            "QListWidget{background:#1e1e1e;border:1px solid #3a3a3a;font-size:11px;}"
            "QListWidget::item{padding:2px 4px;border-bottom:1px solid #2a2a2a;}"
            "QListWidget::item:selected{background:#094771;}"
            "QListWidget::item:hover{background:#2a2d2e;}")
        self.filter_tag_list.itemDoubleClicked.connect(self._remove_filter_item)
        self.filter_tag_list.setVisible(False)
        lv.addWidget(self.filter_tag_list)

        self.filter_count_label = QLabel("")
        self.filter_count_label.setStyleSheet("color:#569cd6;font-size:10px;padding:0 2px;")
        self.filter_count_label.setVisible(False)
        lv.addWidget(self.filter_count_label)

        self.chk_cls23_eq7 = QCheckBox("(cls2 + cls3) ≠ cls7 인 이미지만")
        self.chk_cls23_eq7.setStyleSheet(
            "QCheckBox{color:#FFD700;font-size:11px;padding:2px 0;}"
            "QCheckBox::indicator{width:13px;height:13px;}")
        self.chk_cls23_eq7.stateChanged.connect(self._on_cls23_filter_changed)
        lv.addWidget(self.chk_cls23_eq7)

        lv.addWidget(_hsep())

        # 파일 목록 + 인덱스 바
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemClicked.connect(self._on_list_item_clicked)

        self.box_index_bar = _BoxIndexBar()
        self.box_index_bar.jumpRequested.connect(self._on_index_bar_jump)

        list_row = QHBoxLayout()
        list_row.setSpacing(0); list_row.setContentsMargins(0, 0, 0, 0)
        list_row.addWidget(self.file_list, stretch=1)
        list_row.addWidget(self.box_index_bar)
        lv.addLayout(list_row, stretch=1)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color:#888;font-size:10px;")
        self.summary_label.setWordWrap(True)
        lv.addWidget(self.summary_label)
        splitter.addWidget(left)

        # ════════════════════════════════════════════════════════════════════
        # 중앙 패널: 캔버스
        # ════════════════════════════════════════════════════════════════════
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(2)

        # 캔버스 상단 툴바
        ctb = QHBoxLayout()
        ctb.setContentsMargins(6, 3, 6, 3)
        ctb.setSpacing(4)

        self.confirm_btn = QPushButton("확인 (V)")
        self.confirm_btn.setFixedHeight(24)
        self.confirm_btn.setCheckable(True)
        self.confirm_btn.clicked.connect(self._toggle_confirmed)

        self.btn_merge_cur     = _btn("🔗 통합",   "#1a6b5a", h=24)
        self.btn_rm_border_cur = _btn("✂ 테두리",  "#6b3a1a", h=24)
        self.btn_merge_cur.clicked.connect(self._merge_current_overlapping)
        self.btn_rm_border_cur.clicked.connect(self._remove_border_boxes_cur)

        self.score_label = QLabel("")
        self.score_label.setStyleSheet("padding:0 6px;font-weight:bold;font-size:12px;")

        self.minimap_btn = QPushButton("미니맵 (M)")
        self.minimap_btn.setFixedHeight(24)
        self.minimap_btn.clicked.connect(self.toggle_minimap)

        self.preview_btn = QPushButton("미리보기 (P)")
        self.preview_btn.setFixedHeight(24)
        self.preview_btn.clicked.connect(self.toggle_class_preview)

        self.help_btn = QPushButton("도움말 (H)")
        self.help_btn.setFixedHeight(24)
        self.help_btn.clicked.connect(self.show_guide)

        self.img_nav_label = QLabel("0 / 0")
        self.img_nav_label.setStyleSheet("color:#888;font-size:11px;padding:0 6px;")

        for w in [self.confirm_btn, self.btn_merge_cur, self.btn_rm_border_cur,
                  _vsep(), self.score_label, _vsep(),
                  self.minimap_btn, self.preview_btn, self.help_btn]:
            ctb.addWidget(w)
        ctb.addStretch()
        ctb.addWidget(self.img_nav_label)
        cv.addLayout(ctb)

        self.canvas = CanvasLabel(self)
        cv.addWidget(self.canvas, stretch=1)

        # 클래스 슬라이더
        cls_row = QHBoxLayout()
        cls_row.setContentsMargins(6, 2, 6, 2)
        cls_row.setSpacing(4)
        cls_lbl = QLabel("클래스:")
        cls_lbl.setStyleSheet("color:#aaa;font-size:11px;")
        cls_lbl.setFixedWidth(46)
        self.class_slider = QSlider(Qt.Horizontal)
        self.class_slider.setMinimum(0)
        self.class_slider.setMaximum(max(self.class_count - 1, 0))
        self.class_slider.setValue(0)
        self.class_slider.setMaximumHeight(18)
        self.class_slider.valueChanged.connect(self._on_class_slider)
        self.class_value_label = QLabel("0")
        self.class_value_label.setStyleSheet("color:#dcdcdc;font-size:11px;min-width:80px;")
        cls_row.addWidget(cls_lbl)
        cls_row.addWidget(self.class_slider, stretch=1)
        cls_row.addWidget(self.class_value_label)
        cv.addLayout(cls_row)
        splitter.addWidget(center)

        # ════════════════════════════════════════════════════════════════════
        # 우측 패널: 정보
        # ════════════════════════════════════════════════════════════════════
        right = QWidget()
        right.setMinimumWidth(180)
        right.setMaximumWidth(260)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)
        rv.setSpacing(4)

        rv.addWidget(_sec("📊 품질 점수"))
        self.info_score = QLabel("—")
        self.info_issue = QLabel("—")
        self.info_boxes = QLabel("—")
        self.info_dup   = QLabel("—")
        for lbl in (self.info_score, self.info_issue, self.info_boxes, self.info_dup):
            lbl.setStyleSheet("padding:2px 6px;font-size:12px;")
            rv.addWidget(lbl)

        rv.addWidget(_hsep())
        rv.addWidget(_sec("📦 박스 목록"))
        self.box_list = QListWidget()
        self.box_list.setMaximumHeight(200)
        self.box_list.setStyleSheet(
            "QListWidget{background:#252526;font-size:11px;}"
            "QListWidget::item{padding:2px 4px;}")
        self.box_list.itemClicked.connect(self._on_box_list_clicked)
        rv.addWidget(self.box_list)

        rv.addWidget(_hsep())
        rv.addWidget(_sec("📋 클래스별 박스 수"))
        self.cls_count_list = QListWidget()
        self.cls_count_list.setMaximumHeight(160)
        self.cls_count_list.setMinimumHeight(40)
        self.cls_count_list.setFont(QFont("Courier New", 10))
        self.cls_count_list.setFocusPolicy(Qt.NoFocus)
        self.cls_count_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.cls_count_list.setStyleSheet(
            "QListWidget{background:#1a1a1a;border:1px solid #333;"
            "border-radius:4px;outline:none;}"
            "QListWidget::item{padding:1px 4px;}"
            "QListWidget::item:selected{background:transparent;}")
        rv.addWidget(self.cls_count_list)

        rv.addStretch()
        splitter.addWidget(right)

        splitter.setSizes([220, 1100, 220])

    # ══════════════════════════════════════════════════════════════════════════
    # 사용자 설정 영속화
    # ══════════════════════════════════════════════════════════════════════════
    def _load_prefs(self):
        """저장된 설정을 읽어 UI에 반영. 오류 시 기본값 사용."""
        import json
        prefs = dict(_DEFAULT_PREFS)
        if PREFS_PATH.exists():
            try:
                loaded = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("설정 파일 형식 오류")
                prefs.update(loaded)
            except Exception as e:
                self.status_label.setText(f"⚠ 설정 로드 실패 (기본값 사용): {e}")

        # 윈도우 크기
        w = int(prefs.get("window_w", 1600))
        h = int(prefs.get("window_h", 900))
        self.resize(max(w, 1100), max(h, 650))

        # Config 콤보
        cfg_name = prefs.get("config_name")
        if cfg_name:
            for i in range(self.config_combo.count()):
                if self.config_combo.itemData(i) and \
                        self.config_combo.itemData(i).name == cfg_name:
                    self.config_combo.blockSignals(True)
                    self.config_combo.setCurrentIndex(i)
                    self.config_combo.blockSignals(False)
                    # 내부 상태도 갱신
                    cdir = self.config_combo.itemData(i)
                    self.current_config_dir = cdir
                    self._num_classes  = load_class_count(cdir)
                    self.class_names   = load_class_names_list(cdir)
                    self.class_count   = self._num_classes
                    self.class_slider.setMaximum(max(self.class_count - 1, 0))
                    break

        # 자동 확인 체크박스
        self.chk_auto_confirm.blockSignals(True)
        self.chk_auto_confirm.setChecked(bool(prefs.get("auto_confirm", True)))
        self.chk_auto_confirm.blockSignals(False)

        # 중복 슬라이더
        for attr, key, lo, hi in [
            ("_dup_iou",  "dup_iou",  50, 100),
            ("_dup_area", "dup_area", 30, 100),
        ]:
            if key in prefs and attr in self._dup_sliders:
                raw = int(prefs[key])
                val = max(lo, min(hi, raw))
                sl, lbl = self._dup_sliders[attr]
                sl.blockSignals(True)
                sl.setValue(val)
                lbl.setText(f"{val/100:.2f}")
                setattr(self, attr, val / 100)
                sl.blockSignals(False)

        # 정렬
        sort_key = prefs.get("sort_key", "name_asc")
        for i in range(self.sort_combo.count()):
            if self.sort_combo.itemData(i) == sort_key:
                self.sort_combo.blockSignals(True)
                self.sort_combo.setCurrentIndex(i)
                self._sort_key = sort_key
                self.sort_combo.blockSignals(False)
                break

        # 미니맵 / 클래스 미리보기 표시 여부
        self.show_minimap = bool(prefs.get("show_minimap", True))
        self.minimap_btn.setText("미니맵 ON (M)" if self.show_minimap else "미니맵 OFF (M)")
        self.show_class_preview = bool(prefs.get("show_class_preview", True))
        self.preview_btn.setText(
            "클래스 미리보기 ON (P)" if self.show_class_preview else "클래스 미리보기 OFF (P)")

    def _save_prefs(self):
        """현재 UI 상태를 설정 파일에 저장."""
        import json
        try:
            cdir = self.config_combo.currentData()
            prefs = {
                "config_name": cdir.name if cdir else None,
                "auto_confirm": self.chk_auto_confirm.isChecked(),
                "dup_iou":  self._dup_sliders["_dup_iou"][0].value()
                            if "_dup_iou" in self._dup_sliders else 85,
                "dup_area": self._dup_sliders["_dup_area"][0].value()
                            if "_dup_area" in self._dup_sliders else 75,
                "sort_key": self._sort_key,
                "window_w": self.width(),
                "window_h": self.height(),
                "show_minimap": self.show_minimap,
                "show_class_preview": self.show_class_preview,
            }
            PREFS_PATH.write_text(
                json.dumps(prefs, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as e:
            self.status_label.setText(f"⚠ 설정 저장 실패: {e}")

    def _reset_prefs(self):
        """설정을 기본값으로 초기화."""
        import json
        try:
            if PREFS_PATH.exists():
                PREFS_PATH.unlink()
        except Exception as e:
            self.status_label.setText(f"⚠ 설정 파일 삭제 실패: {e}")
            return

        # 기본값 복원
        self.chk_auto_confirm.blockSignals(True)
        self.chk_auto_confirm.setChecked(_DEFAULT_PREFS["auto_confirm"])
        self.chk_auto_confirm.blockSignals(False)

        for attr, key, lo, hi in [
            ("_dup_iou",  "dup_iou",  50, 100),
            ("_dup_area", "dup_area", 30, 100),
        ]:
            default_val = int(_DEFAULT_PREFS[key])
            if attr in self._dup_sliders:
                sl, lbl = self._dup_sliders[attr]
                sl.blockSignals(True)
                sl.setValue(default_val)
                lbl.setText(f"{default_val/100:.2f}")
                setattr(self, attr, default_val / 100)
                sl.blockSignals(False)

        default_sort = _DEFAULT_PREFS["sort_key"]
        for i in range(self.sort_combo.count()):
            if self.sort_combo.itemData(i) == default_sort:
                self.sort_combo.blockSignals(True)
                self.sort_combo.setCurrentIndex(i)
                self._sort_key = default_sort
                self.sort_combo.blockSignals(False)
                break

        # config는 초기화하지 않음 (의도치 않은 클래스 전환 방지)
        for r in self.results:
            r.pop("has_duplicate", None)
        self._populate_list()
        self.status_label.setText("⟳ 설정이 기본값으로 초기화되었습니다.")

    # ══════════════════════════════════════════════════════════════════════════
    # 설정 변경
    # ══════════════════════════════════════════════════════════════════════════
    def _on_config_changed(self, idx: int):
        cdir = self.config_combo.itemData(idx)
        if not cdir or cdir == self.current_config_dir:
            return
        self.current_config_dir = cdir
        self._num_classes = load_class_count(cdir)
        self.class_names  = load_class_names_list(cdir)
        self.class_count  = self._num_classes
        self.class_slider.setMaximum(max(self.class_count - 1, 0))
        self.status_label.setText(
            f"Config 전환: {cdir.name}  ({self.class_count}개 클래스)")
        self._save_prefs()

    def _on_sort_changed(self, _):
        self._sort_key = self.sort_combo.currentData()
        self._sort_results()
        self._populate_list()
        self._update_list_selection()
        self._save_prefs()

    def _sort_results(self):
        k = self._sort_key
        cur_img = self.results[self.cur_idx]["img_path"] if self.results else None
        if k == "name_asc":
            self.results.sort(key=lambda r: r["img_path"].name)
        elif k == "dup_first":
            self.results.sort(key=lambda r: (not r.get("has_duplicate", False), r["img_path"].name))
        elif k == "unconfirmed":
            self.results.sort(key=lambda r: (r.get("confirmed", False), r["img_path"].name))
        elif k == "box_desc":
            self.results.sort(key=lambda r: (-r.get("box_count", 0), r["img_path"].name))
        elif k == "box_asc":
            self.results.sort(key=lambda r: (r.get("box_count", 0), r["img_path"].name))
        else:
            self.results.sort(key=lambda r: r["img_path"].name)
        # cur_idx 재계산
        if cur_img:
            for i, r in enumerate(self.results):
                if r["img_path"] == cur_img:
                    self.cur_idx = i
                    break

    # ══════════════════════════════════════════════════════════════════════════
    # 파일 목록 패널
    # ══════════════════════════════════════════════════════════════════════════
    def _populate_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        visible_count = 0
        for i, r in enumerate(self.results):
            # ── 클래스 필터 적용 ──────────────────────────────────────────
            if self._filter_classes:
                if "class_ids" not in r:
                    r["class_ids"] = get_class_ids(r["txt_path"])
                if not self._filter_classes.issubset(r["class_ids"]):
                    continue

            # ── (cls2+cls3)==7 필터 ────────────────────────────────────
            if self._filter_cls23_eq7:
                if "cls_counts" not in r:
                    from collections import Counter
                    cnt = Counter()
                    if r["txt_path"].exists():
                        for line in r["txt_path"].read_text().splitlines():
                            parts = line.strip().split()
                            if len(parts) == 5:
                                try: cnt[int(parts[0])] += 1
                                except ValueError: pass
                    r["cls_counts"] = cnt
                c2 = r["cls_counts"].get(2, 0)
                c3 = r["cls_counts"].get(3, 0)
                c7 = r["cls_counts"].get(7, 0)
                if c2 + c3 == 0 and c7 == 0:
                    continue   # 관련 클래스 없는 이미지 제외
                if c2 + c3 == c7:
                    continue   # 정상 (cls2+cls3 == cls7)

            confirmed = r.get("confirmed", False)
            is_issue  = r["is_issue"]
            has_dup   = r.get("has_duplicate", False)

            icon    = "✅" if confirmed else ("⚠️" if is_issue else "  ")
            dup     = "🔁" if has_dup else "  "
            bc      = r.get("box_count", 0)
            name    = r["img_path"].name
            if len(name) > 20:
                name = name[:17] + "..."
            text = f"{icon}{dup} {name}  [{bc}]"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, i)   # results 인덱스 저장

            if confirmed:
                item.setForeground(QColor("#4ec94e"))
            elif has_dup:
                item.setForeground(QColor("#e07070"))
            else:
                item.setForeground(QColor("#dcdcdc"))

            if has_dup and not confirmed:
                item.setBackground(QColor(80, 20, 20, 80))
            elif confirmed:
                item.setBackground(QColor(20, 60, 20, 80))

            self.file_list.addItem(item)
            visible_count += 1

        self.file_list.blockSignals(False)

        # 인덱스 바 갱신
        self._refresh_index_bar()

        # 필터 카운트 라벨 갱신
        if self._filter_classes or self._filter_cls23_eq7:
            self.filter_count_label.setText(
                f"  {visible_count} / {len(self.results)}개 이미지 표시중")
            self.filter_count_label.setVisible(True)
        else:
            self.filter_count_label.setVisible(False)

    def _refresh_index_bar(self):
        """박스 수 기반 iOS 연락처 스타일 인덱스 바 라벨 생성."""
        n = self.file_list.count()
        if n == 0:
            self.box_index_bar.set_labels([])
            return

        # 각 row 의 box_count 수집
        row_counts: list[tuple[int, int]] = []
        for row in range(n):
            item = self.file_list.item(row)
            if item is None:
                continue
            idx = item.data(Qt.UserRole)
            bc  = self.results[idx].get("box_count", 0) if idx is not None else 0
            row_counts.append((row, bc))

        # 버킷 경계 계산 (최대 MAX_LABELS 개)
        MAX_LABELS = 20
        counts = [c for _, c in row_counts]
        mn, mx = min(counts), max(counts)

        if mx == mn:
            # 모두 같은 박스 수
            self.box_index_bar.set_labels([(str(mn), 0)])
            return

        # 고유 값이 MAX_LABELS 이하면 모두 표시
        unique_sorted = sorted(set(counts))
        if len(unique_sorted) <= MAX_LABELS:
            # 각 고유 count 의 첫 번째 row
            first_row: dict[int, int] = {}
            for row, c in row_counts:
                if c not in first_row:
                    first_row[c] = row
            labels = [(str(c), first_row[c]) for c in unique_sorted]
        else:
            # 균등 간격 버킷
            step = (mx - mn) / (MAX_LABELS - 1)
            boundaries = [int(mn + step * k) for k in range(MAX_LABELS)]
            boundaries[-1] = mx  # 마지막은 정확히 max

            labels: list[tuple[str, int]] = []
            prev_label = None
            # row_counts 를 count 기준으로 정렬해 버킷 매핑
            sorted_rc = sorted(row_counts, key=lambda x: x[1])
            b_idx = 0
            for row, c in sorted_rc:
                while b_idx < len(boundaries) - 1 and c >= boundaries[b_idx + 1]:
                    b_idx += 1
                label = str(boundaries[b_idx])
                if label != prev_label:
                    labels.append((label, row))
                    prev_label = label
            # row 기준으로 재정렬 (리스트 순서와 맞추기)
            labels.sort(key=lambda x: x[1])

        self.box_index_bar.set_labels(labels)

    def _on_index_bar_jump(self, row: int):
        item = self.file_list.item(row)
        if item:
            self.file_list.scrollToItem(item, QListWidget.PositionAtTop)
            self.box_index_bar.set_active(row)

    def _update_list_selection(self):
        if not self.results:
            return
        self.file_list.blockSignals(True)
        # UserRole로 현재 cur_idx에 해당하는 행 검색
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item and item.data(Qt.UserRole) == self.cur_idx:
                self.file_list.setCurrentRow(row)
                self.file_list.scrollToItem(item, QListWidget.PositionAtCenter)
                break
        self.file_list.blockSignals(False)

    def _on_list_item_clicked(self, item):
        # 다중 선택 중이면 이미지 전환 없이 선택만 유지
        if len(self.file_list.selectedItems()) > 1:
            return
        idx = item.data(Qt.UserRole)
        if idx is not None and idx != self.cur_idx:
            self.save_labels()
            self._refresh_list_item(self.cur_idx)
            self._load_image_at(idx)

    # ══════════════════════════════════════════════════════════════════════════
    # 이미지 로드 / 네비게이션
    # ══════════════════════════════════════════════════════════════════════════
    def _load_image_at(self, idx: int):
        if not self.results or idx < 0 or idx >= len(self.results):
            return
        self.cur_idx = idx
        r = self.results[idx]

        is_issue  = r["is_issue"]
        has_dup   = r.get("has_duplicate", False)
        confirmed = r.get("confirmed", False)
        box_count = r.get("box_count", 0)
        issue_str = "⚠️ 중복" if has_dup else "✅ 정상"
        conf_str  = "  ✅ 확인됨" if confirmed else ""
        self.setWindowTitle(
            f"검수 - {r['img_path'].name}  [{idx+1}/{len(self.results)}]"
            f"  박스: {box_count}  {issue_str}{conf_str}"
        )
        self.img_nav_label.setText(f"{idx+1} / {len(self.results)}")

        dup_color = "#e07070" if has_dup else "#4ec94e"
        self.score_label.setText(
            f"<span style='color:{dup_color}'>{issue_str}</span>  박스: {box_count}")

        self.confirm_btn.blockSignals(True)
        self.confirm_btn.setChecked(confirmed)
        self._apply_confirm_style(confirmed)
        self.confirm_btn.blockSignals(False)

        img = imread_unicode(str(r["img_path"]))
        if img is None:
            return
        self.h, self.w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(rgb.data, self.w, self.h, self.w * 3, QImage.Format_RGB888)

        self.boxes = load_boxes_from_txt(r["txt_path"], self.w, self.h)
        self.modified_boxes.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.selected_boxes.clear()
        self.selecting = False
        self._load_class_preview_pixmap(self.current_class)
        self.reset_zoom()

        if "has_duplicate" not in r:
            r["has_duplicate"] = compute_has_duplicate(r["txt_path"], self._dup_iou, self._dup_area)

        self._update_right_panel()
        self._update_list_selection()
        # 목록 항목 아이콘 갱신 (로드 후 dup 계산 완료)
        self._refresh_list_item(idx)
        # 인덱스 바 active 갱신
        row = self._current_list_row()
        if row >= 0:
            self.box_index_bar.set_active(row)

    def _current_list_row(self) -> int:
        """현재 cur_idx 에 해당하는 file_list 행 번호 반환 (-1 = 없음)."""
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item and item.data(Qt.UserRole) == self.cur_idx:
                return row
        return -1

    def _navigate_prev(self):
        row = self._current_list_row()
        if row > 0:
            self.save_labels()
            self._refresh_list_item(self.cur_idx)
            prev_idx = self.file_list.item(row - 1).data(Qt.UserRole)
            self._load_image_at(prev_idx)
        elif self.results:
            self.save_labels()
            self._refresh_list_item(self.cur_idx)

    def _navigate_next(self):
        row = self._current_list_row()
        if row >= 0 and row < self.file_list.count() - 1:
            self.save_labels()
            self._refresh_list_item(self.cur_idx)
            next_idx = self.file_list.item(row + 1).data(Qt.UserRole)
            self._load_image_at(next_idx)
        elif self.results:
            self.save_labels()
            self._refresh_list_item(self.cur_idx)

    # ── 클래스 필터 메서드 ──────────────────────────────────────────────────
    def _add_filter_class(self):
        text = self.filter_input.text().strip()
        if not text:
            return
        try:
            cls_id = int(text)
        except ValueError:
            return
        self._filter_classes.add(cls_id)
        self.filter_input.clear()
        self._refresh_filter_ui()
        self._apply_filter_and_reload()

    def _remove_filter_item(self, item):
        cls_id = item.data(Qt.UserRole)
        self._filter_classes.discard(cls_id)
        self._refresh_filter_ui()
        self._apply_filter_and_reload()

    def _clear_filter(self):
        self._filter_classes.clear()
        self._refresh_filter_ui()
        self._populate_list()
        self._update_list_selection()

    def _refresh_filter_ui(self):
        self.filter_tag_list.clear()
        for cls_id in sorted(self._filter_classes):
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else ""
            label = f"[{cls_id}] {name}  ✕ 더블클릭 제거" if name else f"[{cls_id}]  ✕ 더블클릭 제거"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, cls_id)
            item.setForeground(get_color_for_class(cls_id))
            self.filter_tag_list.addItem(item)
        has = bool(self._filter_classes)
        self.filter_tag_list.setVisible(has)
        self.filter_count_label.setVisible(has)

    def _on_cls23_filter_changed(self, state: int):
        self._filter_cls23_eq7 = bool(state)
        # 캐시 무효화
        for r in self.results:
            r.pop("cls_counts", None)
        self._apply_filter_and_reload()

    def _apply_filter_and_reload(self):
        self._populate_list()
        if self.file_list.count() == 0:
            return
        cur_visible = self._current_list_row()
        if cur_visible < 0:
            first_idx = self.file_list.item(0).data(Qt.UserRole)
            self._load_image_at(first_idx)
        else:
            self._update_list_selection()

    # ── 목록 항목 갱신 ──────────────────────────────────────────────────────
    def _refresh_list_item(self, idx: int):
        """목록의 단일 항목 아이콘/색상 재갱신 (UserRole로 행 검색)."""
        if idx < 0 or idx >= len(self.results):
            return
        r = self.results[idx]
        confirmed = r.get("confirmed", False)
        is_issue  = r["is_issue"]
        has_dup   = r.get("has_duplicate", False)
        icon = "✅" if confirmed else ("⚠️" if is_issue else "  ")
        dup  = "🔁" if has_dup else "  "
        bc   = r.get("box_count", 0)
        name = r["img_path"].name
        if len(name) > 20:
            name = name[:17] + "..."
        text = f"{icon}{dup} {name}  [{bc}]"
        # 필터 적용 시 list widget 행 번호 ≠ results 인덱스 → UserRole로 검색
        target_item = None
        for row in range(self.file_list.count()):
            it = self.file_list.item(row)
            if it and it.data(Qt.UserRole) == idx:
                target_item = it
                break
        if target_item:
            self.file_list.blockSignals(True)
            target_item.setText(text)
            if confirmed:
                target_item.setForeground(QColor("#4ec94e"))
                target_item.setBackground(QColor(20, 60, 20, 80))
            elif has_dup:
                target_item.setForeground(QColor("#e07070"))
                target_item.setBackground(QColor(80, 20, 20, 80))
            else:
                target_item.setForeground(QColor("#dcdcdc"))
                target_item.setBackground(QColor())
            self.file_list.blockSignals(False)

    # ══════════════════════════════════════════════════════════════════════════
    # 우측 정보 패널
    # ══════════════════════════════════════════════════════════════════════════
    def _update_right_panel(self):
        if not self.results or self.cur_idx >= len(self.results):
            return
        r = self.results[self.cur_idx]
        confirmed = r.get("confirmed", False)
        has_dup   = r.get("has_duplicate", False)
        box_count = r.get("box_count", 0)

        bc_color = "#4ec94e" if not has_dup else "#e07070"
        self.info_score.setText(
            f"<span style='color:{bc_color}; font-size:14px;'><b>박스 {box_count}개</b></span>")
        self.info_issue.setText("🔁 <span style='color:#e07070;'>중복 있음</span>" if has_dup
                                else "✅ <span style='color:#4ec94e;'>중복 없음</span>")
        self.info_boxes.setText(f"박스 수: <b>{box_count}</b>")
        self.info_dup.setText("🔁 <span style='color:#e09000;'>중복 있음</span>" if has_dup
                              else "<span style='color:#888;'>중복 없음</span>")

        # 박스 목록
        self.box_list.clear()
        dup_indices = self._compute_duplicate_boxes()
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
            icon = "⚠️" if i in dup_indices else ("★" if i in self.modified_boxes else "●")
            item = QListWidgetItem(f"{icon} [{cls}] {name}")
            item.setData(Qt.UserRole, i)
            color = get_color_for_class(cls)
            item.setForeground(color)
            if i in self.selected_boxes:
                item.setBackground(QColor(9, 71, 113, 180))
            self.box_list.addItem(item)

        # 클래스별 박스 수 (오름차순, 중복 없이)
        self.cls_count_list.clear()
        from collections import Counter
        cnt = Counter(cls for cls, *_ in self.boxes)
        for cls in sorted(cnt):
            name = self.class_names[cls] if cls < len(self.class_names) else "?"
            name_s = name[:12] if len(name) > 12 else name
            item = QListWidgetItem(f"[{cls:2d}] {name_s:<12} ×{cnt[cls]}")
            item.setForeground(get_color_for_class(cls))
            self.cls_count_list.addItem(item)

    def _on_box_list_clicked(self, item):
        idx = item.data(Qt.UserRole)
        if idx is not None:
            self.selected_boxes = {idx}
            self.update_display()
            self._update_right_panel()

    # ══════════════════════════════════════════════════════════════════════════
    # 저장 / Undo / Redo
    # ══════════════════════════════════════════════════════════════════════════
    def save_labels(self):
        if not hasattr(self, "qimage") or not self.results:
            return
        r    = self.results[self.cur_idx]
        path = r["txt_path"]
        with open(path, "w") as f:
            for cls, x1, y1, x2, y2 in self.boxes:
                rx1, rx2 = min(x1, x2), max(x1, x2)
                ry1, ry2 = min(y1, y2), max(y1, y2)
                xc = ((rx1 + rx2) / 2) / self.w
                yc = ((ry1 + ry2) / 2) / self.h
                bw = (rx2 - rx1) / self.w
                bh = (ry2 - ry1) / self.h
                f.write(f"{cls} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n")
        r["box_count"] = len(self.boxes)
        r["has_duplicate"] = compute_has_duplicate(path, self._dup_iou, self._dup_area)
        r.pop("class_ids", None)   # 클래스 변경 가능 → 캐시 무효화
        print(f"✅ 저장: {path.name}")

    def _serialize_state(self):
        return {"boxes": [list(b) for b in self.boxes],
                "modified_boxes": sorted(self.modified_boxes)}

    def _apply_state(self, state):
        self.boxes = [tuple(b) for b in state.get("boxes", [])]
        self.modified_boxes = set(state.get("modified_boxes", []))
        self.update_display()
        self._update_right_panel()

    def _push_undo_state(self):
        self.undo_stack.append(self._serialize_state())
        self.redo_stack.clear()

    def undo_action(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self._serialize_state())
        self._apply_state(self.undo_stack.pop())

    def redo_action(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self._serialize_state())
        self._apply_state(self.redo_stack.pop())

    # ══════════════════════════════════════════════════════════════════════════
    # 확인 토글
    # ══════════════════════════════════════════════════════════════════════════
    def _toggle_confirmed(self):
        if not self.results:
            return
        confirmed = self.confirm_btn.isChecked()
        self.results[self.cur_idx]["confirmed"] = confirmed
        self._apply_confirm_style(confirmed)
        r = self.results[self.cur_idx]
        has_dup   = r.get("has_duplicate", False)
        box_count = r.get("box_count", 0)
        issue_str = "⚠️ 중복" if has_dup else "✅ 정상"
        conf_str  = "  ✅ 확인됨" if confirmed else ""
        self.setWindowTitle(
            f"검수 - {r['img_path'].name}  [{self.cur_idx+1}/{len(self.results)}]"
            f"  박스: {box_count}  {issue_str}{conf_str}"
        )
        self._update_right_panel()
        self._refresh_list_item(self.cur_idx)
        save_cache(self.selected_folder, self.results)
        self._update_summary()

    def _apply_confirm_style(self, confirmed: bool):
        if confirmed:
            self.confirm_btn.setText("✅ 확인됨 (V)")
            self.confirm_btn.setStyleSheet(
                "QPushButton{background:#1a7a1a;color:white;border-radius:4px;"
                "font-weight:bold;border:none;padding:3px 10px;}"
                "QPushButton:hover{background:#218838;}")
        else:
            self.confirm_btn.setText("확인 (V)")
            self.confirm_btn.setStyleSheet("")

    # ══════════════════════════════════════════════════════════════════════════
    # 줌 / 피트
    # ══════════════════════════════════════════════════════════════════════════
    def reset_zoom(self):
        cw, ch = self.canvas.width(), self.canvas.height()
        if cw <= 0 or ch <= 0 or self.w <= 0 or self.h <= 0:
            return
        self._fit_zoom = min(cw / self.w, ch / self.h)
        self.zoom = self._fit_zoom
        self.offset = QPoint(
            (cw - int(self.w * self.zoom)) // 2,
            (ch - int(self.h * self.zoom)) // 2,
        )
        self.update_display()

    def _on_canvas_resize(self):
        if hasattr(self, "qimage"):
            self.reset_zoom()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "qimage"):
            self.reset_zoom()

    # ══════════════════════════════════════════════════════════════════════════
    # 박스 유틸
    # ══════════════════════════════════════════════════════════════════════════
    def get_box_index_at(self, pos: QPoint):
        x = (pos.x() - self.offset.x()) / self.zoom
        y = (pos.y() - self.offset.y()) / self.zoom
        for i in reversed(range(len(self.boxes))):
            _, x1, y1, x2, y2 = self.boxes[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    def get_resize_handle_at(self, pos: QPoint):
        ix = (pos.x() - self.offset.x()) / self.zoom
        iy = (pos.y() - self.offset.y()) / self.zoom
        radius = self.HANDLE_SIZE / self.zoom
        for i in reversed(range(len(self.boxes))):
            _, x1, y1, x2, y2 = self.boxes[i]
            if abs(ix - x1) < radius and abs(iy - y1) < radius: return i, "TL"
            if abs(ix - x2) < radius and abs(iy - y1) < radius: return i, "TR"
            if abs(ix - x1) < radius and abs(iy - y2) < radius: return i, "BL"
            if abs(ix - x2) < radius and abs(iy - y2) < radius: return i, "BR"
        return None, None

    def set_box_class(self, box_index, new_class):
        if box_index is None or not (0 <= new_class < self.class_count):
            return
        cls, x1, y1, x2, y2 = self.boxes[box_index]
        if cls == new_class:
            return
        self._push_undo_state()
        self.boxes[box_index] = (new_class, x1, y1, x2, y2)
        self.modified_boxes.add(box_index)
        self.update_display()
        self._update_right_panel()

    def _rebuild_after_delete(self, deleted: list):
        deleted_set = set(deleted)
        def shifted(idx):
            return idx - sum(1 for d in deleted if d < idx)
        self.modified_boxes = {shifted(i) for i in self.modified_boxes if i not in deleted_set}
        self.selected_boxes = {shifted(i) for i in self.selected_boxes if i not in deleted_set}

    def _compute_duplicate_boxes(self) -> set:
        dups = set()
        n = len(self.boxes)
        for i in range(n):
            ci, x1i, y1i, x2i, y2i = self.boxes[i]
            for j in range(i + 1, n):
                cj, x1j, y1j, x2j, y2j = self.boxes[j]
                if ci == cj:
                    continue
                if _is_same_box_pair(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j,
                                     self._dup_iou, self._dup_area):
                    dups.add(i)
                    dups.add(j)
        return dups

    # ══════════════════════════════════════════════════════════════════════════
    # 마우스 이벤트 (캔버스 좌표계 - label.mapFrom 불필요)
    # ══════════════════════════════════════════════════════════════════════════
    def _canvas_mouse_press(self, event):
        pos = event.pos()

        if ((event.modifiers() == Qt.ControlModifier and event.button() == Qt.LeftButton)
                or event.button() == Qt.MiddleButton):
            self.panning = True
            self.pan_start_pos = pos
            self.canvas.setCursor(Qt.ClosedHandCursor)
            return

        x = (pos.x() - self.offset.x()) / self.zoom
        y = (pos.y() - self.offset.y()) / self.zoom

        if event.modifiers() & Qt.ShiftModifier and event.button() == Qt.LeftButton:
            hit = self.get_box_index_at(pos)
            if hit is not None:
                if hit in self.selected_boxes:
                    self.selected_boxes.discard(hit)
                else:
                    self.selected_boxes.add(hit)
                self.update_display()
                self._update_right_panel()
            else:
                self.selecting    = True
                self.select_start = (x, y)
                self.select_end   = (x, y)
            return

        if event.button() == Qt.LeftButton:
            if self.selected_boxes:
                self.selected_boxes.clear()
            idx, handle = self.get_resize_handle_at(pos)
            if idx is not None:
                self.resizing_box_index = idx
                self.resize_handle = handle
                self._active_edit_snapshot = self._serialize_state()
                self.drawing = False
                return
            if 0 <= x < self.w and 0 <= y < self.h:
                self.start_point = (x, y)
                self.end_point   = None
                self.drawing     = True

        elif event.button() == Qt.RightButton:
            for i in reversed(range(len(self.boxes))):
                _, x1, y1, x2, y2 = self.boxes[i]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.dragging_box_index = i
                    self.drag_start_pos     = (x, y)
                    self._active_edit_snapshot = self._serialize_state()
                    break

    def _canvas_mouse_move(self, event):
        pos = event.pos()

        if self.panning:
            delta = pos - self.pan_start_pos
            self.offset += delta
            self.pan_start_pos = pos
            self.update_display()
            return

        x = max(0.0, min(float(self.w), (pos.x() - self.offset.x()) / self.zoom))
        y = max(0.0, min(float(self.h), (pos.y() - self.offset.y()) / self.zoom))

        if self.selecting:
            self.select_end = (x, y)
            self.update_display()
            return

        if self.resizing_box_index is not None:
            cls, x1, y1, x2, y2 = self.boxes[self.resizing_box_index]
            if   self.resize_handle == "TL": x1, y1 = x, y
            elif self.resize_handle == "TR": x2, y1 = x, y
            elif self.resize_handle == "BL": x1, y2 = x, y
            elif self.resize_handle == "BR": x2, y2 = x, y
            self.boxes[self.resizing_box_index] = (cls, x1, y1, x2, y2)
            self.modified_boxes.add(self.resizing_box_index)
            self.update_display()
            return

        if self.drawing:
            self.end_point = (x, y)
            self.update_display()
            return

        if self.dragging_box_index is not None and self.drag_start_pos:
            dx = x - self.drag_start_pos[0]
            dy = y - self.drag_start_pos[1]
            targets = (self.selected_boxes
                       if self.dragging_box_index in self.selected_boxes
                       else {self.dragging_box_index})
            for i in targets:
                cls, bx1, by1, bx2, by2 = self.boxes[i]
                self.boxes[i] = (cls, bx1+dx, by1+dy, bx2+dx, by2+dy)
                self.modified_boxes.add(i)
            self.drag_start_pos = (x, y)
            self.update_display()
            return

        idx, handle = self.get_resize_handle_at(pos)
        if idx is not None:
            self.canvas.setCursor(Qt.SizeFDiagCursor if handle in ("TL","BR")
                                  else Qt.SizeBDiagCursor)
        else:
            self.canvas.setCursor(Qt.ArrowCursor)

    def _canvas_mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.selecting:
            self.selecting = False
            if self.select_start and self.select_end:
                rx1 = min(self.select_start[0], self.select_end[0])
                rx2 = max(self.select_start[0], self.select_end[0])
                ry1 = min(self.select_start[1], self.select_end[1])
                ry2 = max(self.select_start[1], self.select_end[1])
                for i, (_, bx1, by1, bx2, by2) in enumerate(self.boxes):
                    if bx2 >= rx1 and bx1 <= rx2 and by2 >= ry1 and by1 <= ry2:
                        self.selected_boxes.add(i)
            self.update_display()
            self._update_right_panel()
            return

        if event.button() in (Qt.LeftButton, Qt.MiddleButton) and self.panning:
            self.panning = False
            self.canvas.setCursor(Qt.ArrowCursor)
            return

        if self.resizing_box_index is not None:
            prev = self._active_edit_snapshot
            cls, x1, y1, x2, y2 = self.boxes[self.resizing_box_index]
            self.boxes[self.resizing_box_index] = (
                cls, min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            if prev and prev != self._serialize_state():
                self.undo_stack.append(prev)
                self.redo_stack.clear()
            self.resizing_box_index = None
            self.resize_handle = None
            self._active_edit_snapshot = None
            self.update_display()
            self._update_right_panel()
            return

        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            pos = event.pos()
            cx = max(0.0, min(float(self.w),
                              (pos.x() - self.offset.x()) / self.zoom))
            cy = max(0.0, min(float(self.h),
                              (pos.y() - self.offset.y()) / self.zoom))
            x1, y1 = self.start_point
            rx1, rx2 = min(x1, cx), max(x1, cx)
            ry1, ry2 = min(y1, cy), max(y1, cy)
            if abs(rx2 - rx1) > 3 and abs(ry2 - ry1) > 3:
                self._push_undo_state()
                ni = len(self.boxes)
                self.boxes.append((self.current_class, rx1, ry1, rx2, ry2))
                self.modified_boxes.add(ni)
            self.update_display()
            self._update_right_panel()

        elif event.button() == Qt.RightButton:
            if (self._active_edit_snapshot
                    and self._active_edit_snapshot != self._serialize_state()):
                self.undo_stack.append(self._active_edit_snapshot)
                self.redo_stack.clear()
            self.dragging_box_index = None
            self.drag_start_pos     = None
            self._active_edit_snapshot = None

    def _canvas_wheel(self, event):
        dy = event.angleDelta().y()
        dx = event.angleDelta().x()
        pos = event.pos()

        if dx != 0:
            box_index = self.get_box_index_at(pos)
            if box_index is not None:
                cls = self.boxes[box_index][0]
                nc  = min(cls+1, self.class_count-1) if dx > 0 else max(cls-1, 0)
                self.set_box_class(box_index, nc)
                self.class_slider.setValue(nc)
            else:
                nc = (min(self.current_class+1, self.class_count-1) if dx > 0
                      else max(self.current_class-1, 0))
                self.class_slider.setValue(nc)
        elif dy != 0:
            fit = getattr(self, "_fit_zoom", self.zoom)
            new_zoom = min(self.zoom * 1.1, fit * 2.5) if dy > 0 else max(self.zoom * 0.9, fit)
            old_pos  = pos - self.offset
            rx = old_pos.x() / (self.w * self.zoom)
            ry = old_pos.y() / (self.h * self.zoom)
            self.zoom = new_zoom
            self.offset = QPoint(
                int(pos.x() - self.w * self.zoom * rx),
                int(pos.y() - self.h * self.zoom * ry),
            )
            self.update_display()

    # ══════════════════════════════════════════════════════════════════════════
    # 키보드
    # ══════════════════════════════════════════════════════════════════════════
    def keyPressEvent(self, event):
        self._handle_key(event)

    # macOS 가상 키코드 → Qt 키 매핑 (한글 IME 활성 시 key() 값이 깨짐)
    _MAC_VK_MAP = {
        0x00: Qt.Key_A, 0x01: Qt.Key_S, 0x02: Qt.Key_D, 0x03: Qt.Key_F,
        0x04: Qt.Key_H, 0x06: Qt.Key_Z, 0x08: Qt.Key_C, 0x09: Qt.Key_V,
        0x0F: Qt.Key_R, 0x20: Qt.Key_U, 0x23: Qt.Key_P, 0x2E: Qt.Key_M,
        0x24: Qt.Key_Return, 0x33: Qt.Key_Backspace,
        0x7B: Qt.Key_Left, 0x7C: Qt.Key_Right,
    }

    def _handle_key(self, event):
        key = event.key()
        # 한글 IME 활성 시 key()가 Qt.Key_unknown(0x1000000)으로 오는 경우 보정
        if key == Qt.Key_unknown or key == 0:
            key = self._MAC_VK_MAP.get(event.nativeVirtualKey(), Qt.Key_unknown)

        if key == Qt.Key_S:
            self.save_labels()

        elif key in (Qt.Key_Right, Qt.Key_D, Qt.Key_Return):
            self._navigate_next()

        elif key in (Qt.Key_Left, Qt.Key_A):
            self._navigate_prev()

        elif key == Qt.Key_F:
            self.reset_zoom()

        elif key == Qt.Key_M:
            self.toggle_minimap()

        elif key == Qt.Key_P:
            self.toggle_class_preview()

        elif key == Qt.Key_H:
            self.show_guide()

        elif key == Qt.Key_Z:
            self.undo_action()

        elif key == Qt.Key_U:
            self.redo_action()

        elif key == Qt.Key_V:
            if len(self.file_list.selectedItems()) > 1:
                self._confirm_selected_items()
            else:
                self.confirm_btn.setChecked(not self.confirm_btn.isChecked())
                self._toggle_confirmed()

        elif key == Qt.Key_Backspace:
            if self.selected_boxes:
                self._push_undo_state()
                for i in sorted(self.selected_boxes, reverse=True):
                    del self.boxes[i]
                self._rebuild_after_delete(sorted(self.selected_boxes))
                self.selected_boxes.clear()
                self.update_display()
                self._update_right_panel()
            elif self.boxes:
                self._push_undo_state()
                self.boxes.pop()
                self.update_display()
                self._update_right_panel()

        elif key == Qt.Key_C:
            if self.boxes:
                self._push_undo_state()
                self.boxes.clear()
                self.modified_boxes.clear()
                self.update_display()
                self._update_right_panel()

        elif key == Qt.Key_R:
            if self.selected_boxes:
                self._push_undo_state()
                for i in sorted(self.selected_boxes, reverse=True):
                    del self.boxes[i]
                self._rebuild_after_delete(sorted(self.selected_boxes))
                self.selected_boxes.clear()
                self.update_display()
                self._update_right_panel()
            else:
                cursor_pos = self.canvas.mapFromGlobal(QCursor.pos())
                x = (cursor_pos.x() - self.offset.x()) / self.zoom
                y = (cursor_pos.y() - self.offset.y()) / self.zoom
                for i in reversed(range(len(self.boxes))):
                    _, x1, y1, x2, y2 = self.boxes[i]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self._push_undo_state()
                        del self.boxes[i]
                        self._rebuild_after_delete([i])
                        self.update_display()
                        self._update_right_panel()
                        break

        elif key == 0x60:   # backtick
            self.current_class = 0
            self.class_slider.setValue(0)

        elif Qt.Key_0 <= key <= Qt.Key_9:
            tc = key - Qt.Key_0
            if tc < self.class_count:
                self.current_class = tc
                self.class_slider.setValue(tc)
                if self.selected_boxes:
                    self._push_undo_state()
                    for i in self.selected_boxes:
                        cls, x1, y1, x2, y2 = self.boxes[i]
                        self.boxes[i] = (tc, x1, y1, x2, y2)
                        self.modified_boxes.add(i)
                    self.update_display()
                    self._update_right_panel()
                else:
                    cursor_pos = self.canvas.mapFromGlobal(QCursor.pos())
                    self.set_box_class(self.get_box_index_at(cursor_pos), tc)

        elif key == Qt.Key_Escape:
            if self.selected_boxes:
                self.selected_boxes.clear()
                self.update_display()
                self._update_right_panel()

        elif key == Qt.Key_Q:
            self.close()

    # ══════════════════════════════════════════════════════════════════════════
    # 클래스 슬라이더
    # ══════════════════════════════════════════════════════════════════════════
    def _on_class_slider(self, value):
        self.current_class = value
        if value < len(self.class_names):
            self.class_value_label.setText(f"{value}: {self.class_names[value]}")
        else:
            self.class_value_label.setText(str(value))
        self._load_class_preview_pixmap(value)
        self.update_display()

    def _load_class_preview_pixmap(self, cls):
        refs = self.current_config_dir / "refs"
        if not refs.is_dir():
            refs = REFS_DIR
        img_path = refs / f"{cls}.png"
        self.class_preview_pixmap = QPixmap(str(img_path)) if img_path.exists() else None

    # ══════════════════════════════════════════════════════════════════════════
    # 미니맵 / 클래스 프리뷰 토글
    # ══════════════════════════════════════════════════════════════════════════
    def toggle_minimap(self):
        self.show_minimap = not self.show_minimap
        self.minimap_btn.setText("미니맵 ON (M)" if self.show_minimap else "미니맵 OFF (M)")
        self._save_prefs()
        self.update_display()

    def toggle_class_preview(self):
        self.show_class_preview = not self.show_class_preview
        self.preview_btn.setText(
            "클래스 미리보기 ON (P)" if self.show_class_preview else "클래스 미리보기 OFF (P)")
        self._save_prefs()
        self.update_display()

    # ══════════════════════════════════════════════════════════════════════════
    # 그리기 (update_display + 오버레이)
    # ══════════════════════════════════════════════════════════════════════════
    def update_display(self):
        if not hasattr(self, "qimage"):
            return
        pix = QPixmap(self.canvas.size())
        pix.fill(QColor("#2a2a2a"))
        painter = QPainter(pix)

        sw = int(self.w * self.zoom)
        sh = int(self.h * self.zoom)
        scaled = self.qimage.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(self.offset, QPixmap.fromImage(scaled))

        dup_indices = self._compute_duplicate_boxes()

        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            color = get_color_for_class(cls)
            sx1 = x1 * self.zoom + self.offset.x()
            sy1 = y1 * self.zoom + self.offset.y()
            sw_ = (x2 - x1) * self.zoom
            sh_ = (y2 - y1) * self.zoom

            painter.setPen(QPen(color, 2))
            painter.drawRect(QRectF(sx1, sy1, sw_, sh_))

            lbl = (f"{cls} {self.class_names[cls]}"
                   if cls < len(self.class_names) else str(cls))
            painter.drawText(QPointF(sx1, sy1 - 4), lbl)

            if i in self.modified_boxes:
                painter.setPen(QPen(QColor(255, 255, 0), 3))
                bs = 14
                self._draw_corner_bracket(painter, sx1,       sy1,       bs,  0)
                self._draw_corner_bracket(painter, sx1+sw_,   sy1,       bs,  90)
                self._draw_corner_bracket(painter, sx1,       sy1+sh_,   bs, -90)
                self._draw_corner_bracket(painter, sx1+sw_,   sy1+sh_,   bs,  180)
                painter.setPen(QPen(color, 2))

            if i in self.selected_boxes:
                painter.setPen(QPen(QColor(0, 200, 255), 2, Qt.DashLine))
                painter.setBrush(QBrush(QColor(0, 200, 255, 40)))
                painter.drawRect(QRectF(sx1-3, sy1-3, sw_+6, sh_+6))
                painter.setBrush(Qt.NoBrush)

            painter.setBrush(QBrush(color))
            for cx, cy in [(sx1,sy1),(sx1+sw_,sy1),(sx1,sy1+sh_),(sx1+sw_,sy1+sh_)]:
                painter.drawEllipse(QPointF(cx, cy), 3.0, 3.0)
            painter.setBrush(Qt.NoBrush)

        # 중복 박스 오버레이
        if dup_indices:
            fnt = painter.font(); fnt.setPointSize(8); fnt.setBold(True)
            painter.setFont(fnt)
            for i in dup_indices:
                _, x1, y1, x2, y2 = self.boxes[i]
                sx1 = x1 * self.zoom + self.offset.x()
                sy1 = y1 * self.zoom + self.offset.y()
                sw_ = (x2 - x1) * self.zoom
                sh_ = (y2 - y1) * self.zoom
                m = 4
                painter.setPen(QPen(QColor(255, 120, 0), 3, Qt.DashLine))
                painter.setBrush(QBrush(QColor(255, 120, 0, 45)))
                painter.drawRect(QRectF(sx1-m, sy1-m, sw_+m*2, sh_+m*2))
                badge = QRectF(sx1, max(2, sy1-17), 38, 15)
                painter.setBrush(QBrush(QColor(255, 120, 0)))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(badge, 2, 2)
                painter.setPen(QPen(Qt.white))
                painter.drawText(badge, Qt.AlignCenter, "중복!")

        # 고무줄 선택
        if self.selecting and self.select_start and self.select_end:
            rx1 = min(self.select_start[0], self.select_end[0]) * self.zoom + self.offset.x()
            ry1 = min(self.select_start[1], self.select_end[1]) * self.zoom + self.offset.y()
            rw  = abs(self.select_end[0] - self.select_start[0]) * self.zoom
            rh  = abs(self.select_end[1] - self.select_start[1]) * self.zoom
            painter.setPen(QPen(QColor(0, 120, 255), 1, Qt.DashLine))
            painter.setBrush(QBrush(QColor(0, 120, 255, 30)))
            painter.drawRect(QRectF(rx1, ry1, rw, rh))
            painter.setBrush(Qt.NoBrush)

        # 그리기 중인 박스
        if self.drawing and self.start_point and self.end_point:
            painter.setPen(QPen(Qt.red, 1, Qt.DashLine))
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            painter.drawRect(QRectF(
                min(x1,x2)*self.zoom+self.offset.x(),
                min(y1,y2)*self.zoom+self.offset.y(),
                abs(x2-x1)*self.zoom, abs(y2-y1)*self.zoom))

        self._draw_minimap(painter)
        self._draw_class_preview(painter)

        painter.end()
        self.canvas.setPixmap(pix)

    def _draw_corner_bracket(self, painter, x, y, size, angle):
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        painter.drawLine(QPointF(0, 0), QPointF(size, 0))
        painter.drawLine(QPointF(0, 0), QPointF(0, size))
        painter.restore()

    def _draw_minimap(self, painter):
        if not self.show_minimap or not hasattr(self, "qimage"):
            return
        cw, ch = self.canvas.width(), self.canvas.height()
        margin = 10
        mx = cw  - self.minimap_size - margin
        my = ch - self.minimap_size - margin
        painter.fillRect(mx-5, my-5, self.minimap_size+10, self.minimap_size+10,
                         QColor(0, 0, 0, 180))
        ar = self.w / self.h
        if ar > 1:
            mw, mh = self.minimap_size, int(self.minimap_size / ar)
        else:
            mh, mw = self.minimap_size, int(self.minimap_size * ar)
        mox = mx + (self.minimap_size - mw) // 2
        moy = my + (self.minimap_size - mh) // 2
        scaled = self.qimage.scaled(mw, mh, Qt.KeepAspectRatio, Qt.FastTransformation)
        painter.drawPixmap(mox, moy, QPixmap.fromImage(scaled))
        sx, sy = mw / self.w, mh / self.h
        vx = -self.offset.x() / self.zoom
        vy = -self.offset.y() / self.zoom
        vw =  cw / self.zoom
        vh =  ch / self.zoom
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.setBrush(QBrush(QColor(255, 255, 0, 50)))
        painter.drawRect(QRectF(mox + vx*sx, moy + vy*sy, vw*sx, vh*sy))
        for cls, x1, y1, x2, y2 in self.boxes:
            painter.setPen(QPen(get_color_for_class(cls), 1))
            painter.drawRect(QRectF(mox+x1*sx, moy+y1*sy, (x2-x1)*sx, (y2-y1)*sy))

    def _draw_class_preview(self, painter):
        if not self.show_class_preview:
            return
        cw = self.canvas.width()
        margin, size = 10, self.minimap_size
        px = cw - size - margin
        py = margin
        painter.fillRect(px-5, py-5, size+10, size+10, QColor(0, 0, 0, 180))
        if self.class_preview_pixmap and not self.class_preview_pixmap.isNull():
            sc = self.class_preview_pixmap.scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(px + (size-sc.width())//2,
                               py + (size-sc.height())//2, sc)
        else:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(QRectF(px, py, size, size), Qt.AlignCenter,
                             f"No image\n({self.current_class}.png)")

    # ══════════════════════════════════════════════════════════════════════════
    # 폴더 선택 / 검수
    # ══════════════════════════════════════════════════════════════════════════
    def _select_folder(self):
        dlg = SourceFolderDialog(self)
        if dlg.exec_() != QDialog.Accepted or dlg.selected_path is None:
            return
        self._load_folder(dlg.selected_path)

    def _load_folder(self, folder: Path):
        self.selected_folder = Path(folder).resolve()
        self.folder_label.setText(str(self.selected_folder))
        self.folder_label.setStyleSheet(
            "padding:3px 8px; border:1px solid #3a3a3a; border-radius:4px;"
            "background:#1e1e1e; color:#dcdcdc;")
        self.btn_review.setEnabled(True)
        self.btn_zip_txt.setEnabled(True)
        self.zip_name_edit.setText(self.selected_folder.name)
        self.results = []
        self.cur_idx = 0
        self.boxes = []
        self.modified_boxes.clear()
        self.selected_boxes.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.file_list.clear()
        self.summary_label.setText("")

        cached, msg = load_cache(self.selected_folder)
        if cached:
            self.results = cached
            modified = self._get_modified_files()
            mod_str  = f"  수정됨: {len(modified)}개" if modified else ""
            self.status_label.setText(f"✅ 캐시 로드{mod_str}")
            self.btn_partial.setEnabled(bool(modified))
            self._on_finished(self.results, from_cache=True)
        else:
            self.btn_partial.setEnabled(False)
            self.btn_rev_sel.setEnabled(False)
            self.btn_confirm_all.setEnabled(False)
            self.btn_confirm_sel.setEnabled(False)
            self.btn_unconfirm_all.setEnabled(False)
            self.btn_remove_dups.setEnabled(False)
            self.status_label.setText(f"캐시 없음 ({msg}). '전부 검수' 클릭")

    def _zip_txt(self):
        if not self.selected_folder:
            return
        name = self.zip_name_edit.text().strip() or self.selected_folder.name
        if not name.endswith(".zip"):
            name += ".zip"
        zip_path = self.selected_folder.parent / name
        self.btn_zip_txt.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("txt 압축 중...")
        self._zip_worker = ZipWorker(self.selected_folder, zip_path)
        self._zip_worker.progress.connect(lambda c, t: (
            self.progress_bar.setMaximum(t),
            self.progress_bar.setValue(c),
            self.progress_bar.setFormat(f"{c}/{t}"),
        ))
        self._zip_worker.finished.connect(self._zip_on_finished)
        self._zip_worker.error.connect(self._zip_on_error)
        self._zip_worker.start()

    def _zip_on_finished(self, zip_path: str, count: int):
        self.progress_bar.setVisible(False)
        self.btn_zip_txt.setEnabled(True)
        self.status_label.setText(f"압축 완료: {Path(zip_path).name}  ({count}개)")

    def _zip_on_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.btn_zip_txt.setEnabled(True)
        self.status_label.setText(f"압축 오류: {msg.splitlines()[-1]}")

    def _get_modified_files(self) -> set:
        if not self.selected_folder:
            return set()
        cp = _cache_path(self.selected_folder)
        if not cp.exists():
            return set()
        try:
            with open(cp, "rb") as f:
                data = pickle.load(f)
            ts = data.get("ts", 0)
            cached_names = {r["img_path"].name for r in (data.get("results") or [])}
            modified = set()
            img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            for p in self.selected_folder.iterdir():
                if p.suffix.lower() not in img_exts:
                    continue
                txt = p.with_suffix(".txt")
                if not txt.exists():
                    continue
                if txt.stat().st_mtime > ts or p.name not in cached_names:
                    modified.add(p.name)
            return modified
        except Exception:
            return set()

    def _start_review(self):
        if not self.selected_folder:
            return
        self.btn_review.setEnabled(False)
        self.results = []
        self.file_list.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = ReviewWorker(self.selected_folder,
                                   auto_confirm=self.chk_auto_confirm.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.status_msg.connect(lambda m: self.status_label.setText(m))
        self.worker.finished.connect(lambda r: self._on_finished(r, from_cache=False))
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _start_partial_review(self):
        if not self.selected_folder or not self.results:
            return
        modified = self._get_modified_files()
        if not modified:
            QMessageBox.information(self, "재검수", "수정된 파일이 없습니다.")
            return
        reply = QMessageBox.question(
            self, "재검수 확인",
            f"수정된 파일 {len(modified)}개만 재검수합니다. 계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.btn_review.setEnabled(False)
        self.btn_partial.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.worker = ReviewWorker(self.selected_folder,
                                   file_filter=modified,
                                   auto_confirm=self.chk_auto_confirm.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.status_msg.connect(lambda m: self.status_label.setText(m))
        self.worker.finished.connect(self._on_partial_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _review_selected_items(self):
        sel = [r for r in self.results if r.get("selected", False)]
        if not sel:
            QMessageBox.information(self, "선택 검수", "선택된 항목이 없습니다.")
            return
        file_filter = {r["img_path"].name for r in sel}
        self.btn_review.setEnabled(False)
        self.btn_rev_sel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.worker = ReviewWorker(self.selected_folder,
                                   file_filter=file_filter,
                                   auto_confirm=self.chk_auto_confirm.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_sel_review_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, cur, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(cur)
        self.status_label.setText(f"스캔 중... {cur} / {total}")

    def _on_finished(self, results, from_cache=False):
        self.results = results
        self.cur_idx = 0
        self.boxes = []
        self.modified_boxes.clear()
        self.selected_boxes.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.progress_bar.setVisible(False)
        self.btn_review.setEnabled(True)

        if not from_cache:
            save_cache(self.selected_folder, results)

        # dup 미리 계산
        for r in self.results:
            if "has_duplicate" not in r:
                r["has_duplicate"] = compute_has_duplicate(r["txt_path"], self._dup_iou, self._dup_area)

        self._sort_results()
        self._populate_list()

        has = bool(self.results)
        self.btn_confirm_all.setEnabled(has)
        self.btn_confirm_sel.setEnabled(has)
        self.btn_unconfirm_all.setEnabled(has)
        self.btn_merge_all.setEnabled(has)
        self.btn_rm_border_all.setEnabled(has)
        self._refresh_dup_btn()
        self._refresh_partial_btn()
        self._update_summary()
        self.status_label.setText("완료. 목록 클릭 → 이미지 편집")

        if self.results:
            if self.file_list.count() > 0:
                first_idx = self.file_list.item(0).data(Qt.UserRole)
                self._load_image_at(first_idx)

        # 신규 검수일 때만 자동 전처리 (캐시 로드 시에는 건너뜀)
        if not from_cache:
            self._auto_preprocess()

    def _auto_preprocess(self):
        """신규 검수 직후 자동 전처리:
        1) 재귀적 겹침 박스 통합 (변화 없을 때까지)
        2) 테두리 소형 박스 제거 (1회)
        """
        if not self.results:
            return

        merge_total = 0
        border_total = 0

        for r in self.results:
            txt_path = r.get("txt_path")
            if not txt_path:
                continue
            # 1) 재귀적 통합 (함수 내부에서 수렴할 때까지 반복)
            removed_merge = _merge_width_overlapping_txt(txt_path, thresh=0.5)
            if removed_merge:
                merge_total += removed_merge
                r["box_count"] = max(0, r.get("box_count", 0) - removed_merge)
                r.pop("has_duplicate", None)
                r.pop("cls_counts", None)

            # 2) 테두리 소형 박스 제거 (1회)
            removed_border = _remove_border_small_boxes_txt(txt_path)
            if removed_border:
                border_total += removed_border
                r["box_count"] = max(0, r.get("box_count", 0) - removed_border)
                r.pop("has_duplicate", None)
                r.pop("cls_counts", None)

        # 박스 수 오름차순 정렬로 전환 후 목록 갱신
        self._sort_key = "box_asc"
        for i in range(self.sort_combo.count()):
            if self.sort_combo.itemData(i) == "box_asc":
                self.sort_combo.blockSignals(True)
                self.sort_combo.setCurrentIndex(i)
                self.sort_combo.blockSignals(False)
                break
        self._sort_results()
        self._populate_list()
        # 정렬 후 첫 번째 이미지 로드
        if self.results and self.file_list.count() > 0:
            first_idx = self.file_list.item(0).data(Qt.UserRole)
            self._load_image_at(first_idx)

        save_cache(self.selected_folder, self.results)
        self.status_label.setText(
            f"자동 전처리 완료 — 통합 -{merge_total}개, 테두리 제거 -{border_total}개")

    def _on_partial_finished(self, new_results: list):
        self.progress_bar.setVisible(False)
        self.btn_review.setEnabled(True)
        self.btn_partial.setEnabled(False)
        new_map = {r["img_path"].name: r for r in new_results}
        merged, updated, added = [], 0, 0
        for r in self.results:
            name = r["img_path"].name
            if name in new_map:
                nr = new_map.pop(name)
                if r.get("confirmed") and not nr.get("confirmed"):
                    nr["confirmed"] = True
                merged.append(nr); updated += 1
            else:
                merged.append(r)
        for r in new_map.values():
            merged.append(r); added += 1
        self.results = merged
        for r in self.results:
            if "has_duplicate" not in r:
                r["has_duplicate"] = compute_has_duplicate(r["txt_path"], self._dup_iou, self._dup_area)
        save_cache(self.selected_folder, self.results)
        self._sort_results()
        self._populate_list()
        self._update_summary()
        self.status_label.setText(f"재검수 완료. 갱신 {updated}+신규 {added}")
        self._refresh_dup_btn()
        self._refresh_partial_btn()

    def _on_sel_review_finished(self, new_results):
        self.progress_bar.setVisible(False)
        self.btn_review.setEnabled(True)
        new_map = {r["img_path"].name: r for r in new_results}
        for i, r in enumerate(self.results):
            name = r["img_path"].name
            if name in new_map:
                was_conf = r.get("confirmed", False)
                was_sel  = r.get("selected", False)
                self.results[i] = new_map[name]
                if was_conf and not new_map[name].get("confirmed", False):
                    self.results[i]["confirmed"] = True
                self.results[i]["selected"] = was_sel
        for r in self.results:
            if "has_duplicate" not in r:
                r["has_duplicate"] = compute_has_duplicate(r["txt_path"], self._dup_iou, self._dup_area)
        save_cache(self.selected_folder, self.results)
        self._sort_results()
        self._populate_list()
        self._update_summary()
        self.status_label.setText("선택 검수 완료.")
        self._refresh_dup_btn()
        self._refresh_partial_btn()
        any_sel = any(r.get("selected", False) for r in self.results)
        self.btn_rev_sel.setEnabled(any_sel)

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_review.setEnabled(True)
        self.status_label.setText("오류 발생")
        QMessageBox.critical(self, "오류", msg)

    # ══════════════════════════════════════════════════════════════════════════
    # 일괄 확인 / 해제 / 선택 확인
    # ══════════════════════════════════════════════════════════════════════════
    def _confirm_all(self):
        unconfirmed = [i for i, r in enumerate(self.results)
                       if not r.get("confirmed", False)]
        if not unconfirmed:
            QMessageBox.information(self, "일괄 확인", "모두 이미 확인됨 상태입니다.")
            return
        reply = QMessageBox.question(
            self, "일괄 확인", f"미확인 {len(unconfirmed)}개를 모두 확인 처리할까요?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        for i in unconfirmed:
            self.results[i]["confirmed"] = True
        save_cache(self.selected_folder, self.results)
        self._populate_list()
        self._update_summary()
        if self.results:
            self._apply_confirm_style(self.results[self.cur_idx].get("confirmed", False))

    def _unconfirm_all(self):
        confirmed_list = [i for i, r in enumerate(self.results) if r.get("confirmed", False)]
        if not confirmed_list:
            QMessageBox.information(self, "전체 해제", "확인된 항목이 없습니다.")
            return
        reply = QMessageBox.question(
            self, "전체 확인 해제", f"확인된 {len(confirmed_list)}개를 모두 해제할까요?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        for i in confirmed_list:
            self.results[i]["confirmed"] = False
        save_cache(self.selected_folder, self.results)
        self._populate_list()
        self._update_summary()
        if self.results:
            self._apply_confirm_style(False)
            self.confirm_btn.blockSignals(True)
            self.confirm_btn.setChecked(False)
            self.confirm_btn.blockSignals(False)

    def _confirm_selected_items(self):
        """좌측 목록에서 선택된 항목 전부를 확인 처리."""
        if not self.results:
            return
        selected = self.file_list.selectedItems()
        if not selected:
            return
        for item in selected:
            idx = item.data(Qt.UserRole)
            if idx is not None:
                self.results[idx]["confirmed"] = True
                self._refresh_list_item(idx)
        # 현재 이미지가 포함된 경우 우측 패널 갱신
        cur_item_idxs = {i.data(Qt.UserRole) for i in selected}
        if self.cur_idx in cur_item_idxs:
            self.confirm_btn.blockSignals(True)
            self.confirm_btn.setChecked(True)
            self._apply_confirm_style(True)
            self.confirm_btn.blockSignals(False)
        save_cache(self.selected_folder, self.results)
        self._update_summary()

    # ══════════════════════════════════════════════════════════════════════════
    # 중복 박스 일괄 제거
    # ══════════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════════
    # 너비 겹침 박스 통합
    # ══════════════════════════════════════════════════════════════════════════
    def _merge_current_overlapping(self):
        """현재 이미지의 너비 50% 이상 겹치는 박스를 외곽 바운딩박스로 합침."""
        if not self.results:
            return
        before = len(self.boxes)
        merged = _merge_width_overlapping_boxes(self.boxes)
        if len(merged) == before:
            self.status_label.setText("통합할 겹치는 박스가 없습니다.")
            return
        self._push_undo_state()
        self.boxes = merged
        self.modified_boxes.clear()
        self.selected_boxes.clear()
        self.save_labels()
        self.update_display()
        self._update_right_panel()
        self.status_label.setText(
            f"통합 완료: {before}개 → {len(merged)}개 (-{before - len(merged)}개)")

    def _merge_all_overlapping(self):
        """전체 이미지에 너비 겹침 박스 통합 일괄 적용."""
        if not self.results:
            return
        reply = QMessageBox.question(
            self, "전체 중복 통합",
            f"전체 {len(self.results)}개 이미지에 너비 50% 이상 겹치는 박스를 "
            f"외곽 바운딩박스로 통합합니다.\n계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        total_merged = 0
        affected = 0
        cur_r = self.results[self.cur_idx] if self.results else None

        for i, r in enumerate(self.results):
            if i == self.cur_idx:
                # 현재 이미지는 메모리에서 바로 merge → 한 번만 저장
                before = len(self.boxes)
                merged = _merge_width_overlapping_boxes(self.boxes)
                removed = before - len(merged)
                if removed:
                    self.boxes = merged
                    self.modified_boxes.clear()
                    self.selected_boxes.clear()
                    self.save_labels()   # 통합 결과 저장 1회
                    total_merged += removed
                    affected += 1
                    r["box_count"] = len(merged)
                    r.pop("has_duplicate", None)
                    r.pop("cls_counts", None)
            else:
                removed = _merge_width_overlapping_txt(r["txt_path"])
                if removed:
                    total_merged += removed
                    affected += 1
                    r["box_count"] = max(0, r.get("box_count", 0) - removed)
                    r.pop("has_duplicate", None)
                    r.pop("cls_counts", None)

        self.update_display()
        self._update_right_panel()

        save_cache(self.selected_folder, self.results)
        self._populate_list()
        self.status_label.setText(
            f"전체 통합 완료: {affected}개 이미지, 총 {total_merged}개 박스 감소")

    def _remove_border_boxes_cur(self):
        """현재 이미지의 테두리 소형 박스 제거."""
        if not self.results:
            return
        before = len(self.boxes)
        new_boxes, removed = _remove_border_small_boxes(self.boxes, self.w, self.h)
        if not removed:
            self.status_label.setText("제거할 테두리 박스가 없습니다.")
            return
        self._push_undo_state()
        self.boxes = new_boxes
        self.modified_boxes.clear()
        self.selected_boxes.clear()
        self.save_labels()
        self.update_display()
        self._update_right_panel()
        self.status_label.setText(
            f"테두리 박스 제거: {before}개 → {len(self.boxes)}개 (-{removed}개)")

    def _remove_border_boxes_all(self):
        """전체 이미지에서 테두리 소형 박스 일괄 제거."""
        if not self.results:
            return
        reply = QMessageBox.question(
            self, "테두리 박스 전체 제거",
            f"전체 {len(self.results)}개 이미지에서 테두리에 걸친 소형 박스를 제거합니다.\n"
            f"(기준: 내부 박스 평균 면적의 50% 미만)\n계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        total_removed = 0
        affected = 0
        for i, r in enumerate(self.results):
            if i == self.cur_idx:
                before = len(self.boxes)
                new_boxes, removed = _remove_border_small_boxes(
                    self.boxes, self.w, self.h)
                if removed:
                    self.boxes = new_boxes
                    self.modified_boxes.clear()
                    self.selected_boxes.clear()
                    self.save_labels()
                    total_removed += removed
                    affected += 1
                    r["box_count"] = len(self.boxes)
                    r.pop("cls_counts", None)
                    r.pop("has_duplicate", None)
            else:
                removed = _remove_border_small_boxes_txt(r["txt_path"])
                if removed:
                    total_removed += removed
                    affected += 1
                    r["box_count"] = max(0, r.get("box_count", 0) - removed)
                    r.pop("cls_counts", None)
                    r.pop("has_duplicate", None)

        self.update_display()
        self._update_right_panel()
        save_cache(self.selected_folder, self.results)
        self._populate_list()
        self.status_label.setText(
            f"테두리 박스 제거 완료: {affected}개 이미지, 총 {total_removed}개 박스 제거")

    def _remove_all_duplicates(self):
        dup_results = [r for r in self.results if r.get("has_duplicate", False)]
        if not dup_results:
            QMessageBox.information(self, "중복 박스 제거", "중복 박스가 없습니다.")
            return
        reply = QMessageBox.question(
            self, "중복 박스 일괄 제거",
            f"중복 박스가 있는 이미지 {len(dup_results)}개를 처리합니다.\n"
            f"낮은 인덱스(먼저 그린 박스)를 유지합니다.\n계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        total_removed = 0
        for r in dup_results:
            n = remove_duplicates_from_txt(r["txt_path"])
            total_removed += n
            r["has_duplicate"] = compute_has_duplicate(r["txt_path"], self._dup_iou, self._dup_area)
            if r["txt_path"].exists():
                r["box_count"] = sum(
                    1 for line in r["txt_path"].read_text().strip().splitlines()
                    if len(line.split()) == 5)
        self._populate_list()
        save_cache(self.selected_folder, self.results)
        self._update_summary()
        self._refresh_dup_btn()
        # 현재 이미지 재로드
        if self.results:
            self._load_image_at(self.cur_idx)
        QMessageBox.information(
            self, "완료", f"처리 파일: {len(dup_results)}개  |  제거 박스: {total_removed}개")

    # ══════════════════════════════════════════════════════════════════════════
    # 버튼 상태 / 요약
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_dup_btn(self):
        has_dup = any(r.get("has_duplicate", False) for r in self.results)
        self.btn_remove_dups.setEnabled(has_dup)

    def _refresh_partial_btn(self):
        if not self.selected_folder:
            return
        modified = self._get_modified_files()
        self.btn_partial.setEnabled(bool(modified))

    def _update_summary(self):
        if not self.results:
            self.summary_label.setText("")
            return
        n  = len(self.results)
        nc = sum(r.get("confirmed", False)     for r in self.results)
        nd = sum(r.get("has_duplicate", False) for r in self.results)
        self.summary_label.setText(
            f"총 {n}개  |  중복 {nd}\n"
            f"확인 {nc}/{n}")

    # ══════════════════════════════════════════════════════════════════════════
    # 도움말
    # ══════════════════════════════════════════════════════════════════════════
    def show_guide(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("조작 가이드")
        msg.setTextFormat(Qt.RichText)
        msg.setText("""
        <h3>🖱️ 마우스</h3>
        <ul>
          <li><b>좌드래그</b> : 박스 그리기</li>
          <li><b>Ctrl+좌드래그</b> : 이미지 이동 (Panning)</li>
          <li><b>우드래그</b> : 박스 이동</li>
          <li><b>박스 모서리 드래그</b> : 박스 크기 조절</li>
          <li><b>Shift+좌클릭/드래그</b> : 다중 선택</li>
          <li><b>휠↑↓</b> : 줌 인/아웃</li>
          <li><b>휠←→ (박스 위)</b> : 클래스 변경</li>
        </ul>
        <h3>⌨️ 키보드</h3>
        <table cellpadding="2">
          <tr><td><b>A / ←</b></td><td>이전 이미지 (저장 후)</td></tr>
          <tr><td><b>D / → / Enter</b></td><td>다음 이미지 (저장 후)</td></tr>
          <tr><td><b>S</b></td><td>저장</td></tr>
          <tr><td><b>F</b></td><td>화면 맞춤</td></tr>
          <tr><td><b>V</b></td><td>확인 토글</td></tr>
          <tr><td><b>Z</b></td><td>실행 취소</td></tr>
          <tr><td><b>U</b></td><td>다시 실행</td></tr>
          <tr><td><b>R / BackSpace</b></td><td>박스 삭제</td></tr>
          <tr><td><b>C</b></td><td>전체 박스 삭제</td></tr>
          <tr><td><b>0~9</b></td><td>클래스 변경</td></tr>
          <tr><td><b>M</b></td><td>미니맵 ON/OFF</td></tr>
          <tr><td><b>P</b></td><td>클래스 미리보기 ON/OFF</td></tr>
          <tr><td><b>H</b></td><td>도움말</td></tr>
          <tr><td><b>Q / Esc</b></td><td>닫기</td></tr>
        </table>""")
        msg.exec_()

    # ══════════════════════════════════════════════════════════════════════════
    # 닫기
    # ══════════════════════════════════════════════════════════════════════════
    def closeEvent(self, event):
        self.save_labels()
        self._save_prefs()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK)

    # ── 시작 시 sources 폴더 선택 팝업 ──────────────────────────────────────
    dlg = SourceFolderDialog()
    if dlg.exec_() != QDialog.Accepted or dlg.selected_path is None:
        sys.exit(0)

    win = ReviewApp()
    win.show()
    # 선택된 폴더를 자동으로 로드
    win._load_folder(dlg.selected_path)
    sys.exit(app.exec_())
