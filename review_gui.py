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
import torch
import yaml


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QSlider, QSizePolicy, QListWidget, QListWidgetItem, QSplitter,
    QComboBox, QCheckBox, QScrollArea, QFrame, QLineEdit
)
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPen, QBrush, QCursor, QFont
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QPointF, QRectF, QTimer

# ── 경로 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
MODEL_PATH   = PROJECT_ROOT / "auto/runs/detect/train/weights/best.pt"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
CONFIG_DIR   = SCRIPT_DIR / "config"
REFS_DIR     = SCRIPT_DIR / "refs"

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


def compute_has_duplicate(txt_path: Path) -> bool:
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
            if _is_same_box_pair(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j):
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
            "model":   str(MODEL_PATH),
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
# Cleanlab 포맷 변환
# ══════════════════════════════════════════════════════════════════════════════
def _load_gt_cleanlab(txt_path: Path):
    bboxes, cls_ids = [], []
    if txt_path.exists():
        for line in txt_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                bboxes.append(yolo_to_xyxy_norm(cx, cy, w, h))
                cls_ids.append(c)
    return {
        "bboxes": np.array(bboxes,  dtype=float).reshape(-1, 4),
        "labels": np.array(cls_ids, dtype=int),
    }


def _to_cleanlab_preds(raw_preds: list, num_classes: int):
    cl = []
    for raw in raw_preds:
        per = []
        for k in range(num_classes):
            if len(raw) == 0:
                per.append(np.zeros((0, 5), dtype=float))
            else:
                mask = raw[:, 5].astype(int) == k
                per.append(raw[mask][:, :5] if mask.any()
                            else np.zeros((0, 5), dtype=float))
        cl.append(per)
    return cl


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
# 검수 워커 (원본 유지)
# ══════════════════════════════════════════════════════════════════════════════
class ReviewWorker(QThread):
    progress   = pyqtSignal(int, int)
    status_msg = pyqtSignal(str)
    finished   = pyqtSignal(list)
    error      = pyqtSignal(str)

    def __init__(self, folder: Path, num_classes: int = 31, conf: float = 0.01,
                 file_filter=None, auto_confirm: bool = True):
        super().__init__()
        self.folder       = folder
        self.num_classes  = num_classes
        self.conf         = conf
        self.file_filter  = file_filter
        self.auto_confirm = auto_confirm

    def run(self):
        try:
            from ultralytics import YOLO
            from cleanlab.object_detection.filter import find_label_issues
            from cleanlab.object_detection.rank   import get_label_quality_scores

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
            self.status_msg.emit(f"{label} {total}개 파일. 모델 로딩 중...")
            gt_labels = [_load_gt_cleanlab(txt) for _, txt in pairs]

            model = YOLO(str(MODEL_PATH))
            raw_preds, batch = [], 32
            img_paths = [str(p) for p, _ in pairs]

            for i in range(0, total, batch):
                results = model.predict(img_paths[i:i+batch],
                                        conf=self.conf, device=get_device(), verbose=False)
                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        raw_preds.append(np.zeros((0, 6), dtype=float))
                        continue
                    xyxy  = boxes.xyxyn.cpu().numpy()
                    confs = boxes.conf.cpu().numpy().reshape(-1, 1)
                    cls   = boxes.cls.cpu().numpy().reshape(-1, 1)
                    raw_preds.append(np.hstack([xyxy, confs, cls]))
                self.progress.emit(min(i + batch, total), total)

            self.status_msg.emit("Cleanlab 점수 계산 중...")
            cl_preds = _to_cleanlab_preds(raw_preds, self.num_classes)
            scores   = get_label_quality_scores(gt_labels, cl_preds, verbose=False)
            issues   = find_label_issues(gt_labels, cl_preds)

            out = []
            for i, (img_path, txt_path) in enumerate(pairs):
                s = float(scores[i])
                out.append({
                    "img_path":  img_path,
                    "txt_path":  txt_path,
                    "score":     s,
                    "is_issue":  bool(issues[i]),
                    "box_count": int(len(gt_labels[i]["labels"])),
                    "confirmed": self.auto_confirm and s >= 1.0,
                })
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

    def mousePressEvent(self, e):   self._app._canvas_mouse_press(e)
    def mouseMoveEvent(self, e):    self._app._canvas_mouse_move(e)
    def mouseReleaseEvent(self, e): self._app._canvas_mouse_release(e)
    def wheelEvent(self, e):        self._app._canvas_wheel(e)
    def keyPressEvent(self, e):     self._app._handle_key(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, self._app._on_canvas_resize)


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
        self._sort_key         = "score_asc"   # 현재 정렬 기준
        self._filter_classes: set = set()      # 클래스 ID 필터 (빈 set = 전체 표시)
        self._filter_cls23_eq7: bool = False   # (cls2+cls3)==7 필터

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

    # ══════════════════════════════════════════════════════════════════════════
    # UI 구성
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 상단 툴바 ────────────────────────────────────────────────────────
        tb = QWidget()
        tb.setFixedHeight(100)
        tb.setStyleSheet("background:#252526; border-bottom:1px solid #3a3a3a;")
        tb_layout = QVBoxLayout(tb)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(3)

        row1 = QHBoxLayout(); row1.setSpacing(6)
        self.folder_label = QLabel("폴더를 선택하세요")
        self.folder_label.setStyleSheet(
            "padding:3px 8px; border:1px solid #3a3a3a; border-radius:4px;"
            "background:#1e1e1e; color:#aaa;")
        self.folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        def _btn(text, color=None, height=26):
            b = QPushButton(text)
            b.setFixedHeight(height)
            if color:
                b.setStyleSheet(
                    f"QPushButton{{background:{color};color:white;border-radius:4px;"
                    f"font-weight:bold;border:none;padding:3px 10px;}}"
                    f"QPushButton:hover{{background:{color}cc;}}"
                    f"QPushButton:disabled{{background:#2a2a2a;color:#555;border:1px solid #3a3a3a;}}")
            return b

        btn_sel           = _btn("📁 폴더 선택")
        self.btn_review   = _btn("🔍 전부 검수",  "#0e639c")
        self.btn_partial  = _btn("🔄 재검수",      "#b85c00")
        self.btn_rev_sel  = _btn("☑ 선택 검수",   "#7d3c98")
        self.btn_confirm_all   = _btn("✅ 전체 확인",  "#218838")
        self.btn_confirm_sel   = _btn("☑ 선택 확인",  "#4cae4c")
        self.btn_unconfirm_all = _btn("✖ 확인 해제",  "#5a6268")
        self.btn_remove_dups   = _btn("🗑 중복 제거",  "#c0392b")

        for b in (self.btn_review, self.btn_partial, self.btn_rev_sel,
                  self.btn_confirm_all, self.btn_confirm_sel,
                  self.btn_unconfirm_all, self.btn_remove_dups):
            b.setEnabled(False)

        btn_sel.clicked.connect(self._select_folder)
        self.btn_review.clicked.connect(self._start_review)
        self.btn_partial.clicked.connect(self._start_partial_review)
        self.btn_rev_sel.clicked.connect(self._review_selected_items)
        self.btn_confirm_all.clicked.connect(self._confirm_all)
        self.btn_confirm_sel.clicked.connect(self._confirm_selected_items)
        self.btn_unconfirm_all.clicked.connect(self._unconfirm_all)
        self.btn_remove_dups.clicked.connect(self._remove_all_duplicates)

        row1.addWidget(self.folder_label)
        row1.addWidget(btn_sel)
        row1.addWidget(self.btn_review)
        row1.addWidget(self.btn_partial)
        row1.addWidget(self.btn_rev_sel)
        row1.addWidget(self.btn_confirm_all)
        row1.addWidget(self.btn_confirm_sel)
        row1.addWidget(self.btn_unconfirm_all)
        row1.addWidget(self.btn_remove_dups)
        tb_layout.addLayout(row1)

        row2 = QHBoxLayout(); row2.setSpacing(6)
        self.chk_auto_confirm = QCheckBox("검수 후 자동 확인 (점수=1.0)")
        self.chk_auto_confirm.setChecked(True)

        config_lbl = QLabel("Config:")
        config_lbl.setStyleSheet("color:#aaa;")
        self.config_combo = QComboBox()
        self.config_combo.setFixedHeight(24)
        self.config_combo.setMinimumWidth(120)
        for cdir in find_config_dirs():
            self.config_combo.addItem(cdir.name, userData=cdir)
        default_idx = next((i for i in range(self.config_combo.count())
                            if self.config_combo.itemData(i) == CONFIG_DIR), 0)
        self.config_combo.setCurrentIndex(default_idx)
        self.config_combo.currentIndexChanged.connect(self._on_config_changed)

        self.status_label = QLabel("준비")
        self.status_label.setStyleSheet("color:#aaa; font-size:11px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(200)

        row2.addWidget(self.chk_auto_confirm)
        row2.addSpacing(8)
        row2.addWidget(config_lbl)
        row2.addWidget(self.config_combo)
        row2.addSpacing(8)
        row2.addWidget(self.status_label)
        row2.addWidget(self.progress_bar)
        row2.addStretch()
        tb_layout.addLayout(row2)

        # ── row3: TXT 압축 ───────────────────────────────────────────────────
        sep = QWidget(); sep.setFixedHeight(1)
        sep.setStyleSheet("background:#3a3a3a;")
        tb_layout.addWidget(sep)

        row3 = QHBoxLayout(); row3.setSpacing(6)
        zip_lbl = QLabel("📦 TXT 압축:")
        zip_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        self.zip_name_edit = QLineEdit()
        self.zip_name_edit.setPlaceholderText("파일명 (폴더 선택 시 자동 입력)")
        self.zip_name_edit.setFixedHeight(22)
        self.zip_name_edit.setStyleSheet(
            "background:#1e1e1e; color:#dcdcdc; border:1px solid #3a3a3a;"
            "border-radius:3px; padding:1px 6px; font-size:11px;")
        self.btn_zip_txt = _btn("압축 시작", "#5a3e8a", height=22)
        self.btn_zip_txt.setEnabled(False)
        self.btn_zip_txt.clicked.connect(self._zip_txt)
        row3.addWidget(zip_lbl)
        row3.addWidget(self.zip_name_edit)
        row3.addWidget(self.btn_zip_txt)
        tb_layout.addLayout(row3)

        root.addWidget(tb)

        # ── 3분할 스플리터 ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        root.addWidget(splitter)

        # ── 좌측 패널: 파일 목록 ─────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(200)
        left.setMaximumWidth(320)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.setSpacing(4)

        sort_row = QHBoxLayout()
        sort_lbl = QLabel("정렬:")
        sort_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        self.sort_combo = QComboBox()
        self.sort_combo.setFixedHeight(22)
        for k, v in [("score_asc","점수 낮은순"), ("score_desc","점수 높은순"),
                     ("name_asc","파일명순"), ("issue_first","의심 우선"),
                     ("dup_first","중복 우선"), ("unconfirmed","미확인 우선")]:
            self.sort_combo.addItem(v, userData=k)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        sort_row.addWidget(sort_lbl)
        sort_row.addWidget(self.sort_combo)
        lv.addLayout(sort_row)

        # ── 클래스 필터 ──────────────────────────────────────────────────────
        filter_hdr = QHBoxLayout()
        filter_ttl = QLabel("🔍 클래스 필터")
        filter_ttl.setStyleSheet("color:#bbb; font-size:11px; font-weight:bold;")
        self.btn_clear_filter = QPushButton("초기화")
        self.btn_clear_filter.setFixedHeight(20)
        self.btn_clear_filter.setFixedWidth(48)
        self.btn_clear_filter.setStyleSheet(
            "QPushButton{background:#3a3a3a;color:#aaa;border:1px solid #555;"
            "border-radius:3px;font-size:10px;padding:0;}"
            "QPushButton:hover{background:#555;color:#fff;}")
        self.btn_clear_filter.clicked.connect(self._clear_filter)
        filter_hdr.addWidget(filter_ttl)
        filter_hdr.addStretch()
        filter_hdr.addWidget(self.btn_clear_filter)
        lv.addLayout(filter_hdr)

        filter_input_row = QHBoxLayout()
        filter_input_row.setSpacing(3)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("클래스 ID (예: 0)")
        self.filter_input.setFixedHeight(22)
        self.filter_input.setStyleSheet(
            "QLineEdit{background:#2a2a2a;color:#dcdcdc;border:1px solid #555;"
            "border-radius:3px;padding:1px 4px;font-size:11px;}")
        self.filter_input.returnPressed.connect(self._add_filter_class)
        btn_add_filter = QPushButton("추가")
        btn_add_filter.setFixedHeight(22)
        btn_add_filter.setFixedWidth(38)
        btn_add_filter.setStyleSheet(
            "QPushButton{background:#0e639c;color:#fff;border:none;"
            "border-radius:3px;font-size:11px;}"
            "QPushButton:hover{background:#1177bb;}")
        btn_add_filter.clicked.connect(self._add_filter_class)
        filter_input_row.addWidget(self.filter_input)
        filter_input_row.addWidget(btn_add_filter)
        lv.addLayout(filter_input_row)

        # 활성 필터 태그 목록 (더블클릭 → 제거)
        self.filter_tag_list = QListWidget()
        self.filter_tag_list.setMaximumHeight(88)
        self.filter_tag_list.setStyleSheet(
            "QListWidget{background:#1e1e1e;border:1px solid #3a3a3a;"
            "font-size:11px;}"
            "QListWidget::item{padding:2px 4px;border-bottom:1px solid #2a2a2a;}"
            "QListWidget::item:selected{background:#094771;}"
            "QListWidget::item:hover{background:#2a2d2e;}")
        self.filter_tag_list.itemDoubleClicked.connect(self._remove_filter_item)
        self.filter_tag_list.setVisible(False)
        lv.addWidget(self.filter_tag_list)

        self.filter_count_label = QLabel("")
        self.filter_count_label.setStyleSheet(
            "color:#569cd6; font-size:10px; padding:0 2px;")
        self.filter_count_label.setVisible(False)
        lv.addWidget(self.filter_count_label)

        self.chk_cls23_eq7 = QCheckBox("(cls2 + cls3) = 7 인 이미지만")
        self.chk_cls23_eq7.setStyleSheet(
            "QCheckBox{color:#FFD700; font-size:11px; padding:2px 0;}"
            "QCheckBox::indicator{width:13px;height:13px;}")
        self.chk_cls23_eq7.stateChanged.connect(self._on_cls23_filter_changed)
        lv.addWidget(self.chk_cls23_eq7)

        # 구분선
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#3a3a3a;")
        lv.addWidget(sep)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.itemClicked.connect(self._on_list_item_clicked)
        lv.addWidget(self.file_list)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color:#888; font-size:10px;")
        self.summary_label.setWordWrap(True)
        lv.addWidget(self.summary_label)
        splitter.addWidget(left)

        # ── 중앙 패널: 캔버스 + 슬라이더 ─────────────────────────────────────
        center = QWidget()
        cv_layout = QVBoxLayout(center)
        cv_layout.setContentsMargins(0, 0, 0, 0)
        cv_layout.setSpacing(2)

        # 캔버스 상단 미니 툴바
        ctb = QHBoxLayout()
        ctb.setContentsMargins(4, 2, 4, 2)
        ctb.setSpacing(4)

        self.confirm_btn = QPushButton("확인 (V)")
        self.confirm_btn.setFixedHeight(24)
        self.confirm_btn.setCheckable(True)
        self.confirm_btn.clicked.connect(self._toggle_confirmed)

        self.minimap_btn = QPushButton("미니맵 ON (M)")
        self.minimap_btn.setFixedHeight(24)
        self.minimap_btn.clicked.connect(self.toggle_minimap)

        self.preview_btn = QPushButton("클래스 미리보기 (P)")
        self.preview_btn.setFixedHeight(24)
        self.preview_btn.clicked.connect(self.toggle_class_preview)

        self.help_btn = QPushButton("도움말 (H)")
        self.help_btn.setFixedHeight(24)
        self.help_btn.clicked.connect(self.show_guide)

        self.score_label = QLabel("")
        self.score_label.setFixedHeight(24)
        self.score_label.setStyleSheet("padding:0 8px; font-weight:bold;")
        self.img_nav_label = QLabel("0 / 0")
        self.img_nav_label.setStyleSheet("color:#888; font-size:11px; padding:0 8px;")

        ctb.addWidget(self.confirm_btn)
        ctb.addWidget(self.minimap_btn)
        ctb.addWidget(self.preview_btn)
        ctb.addWidget(self.help_btn)
        ctb.addWidget(self.score_label)
        ctb.addStretch()
        ctb.addWidget(self.img_nav_label)
        cv_layout.addLayout(ctb)

        # 캔버스
        self.canvas = CanvasLabel(self)
        cv_layout.addWidget(self.canvas, stretch=1)

        # 클래스 슬라이더
        cls_row = QHBoxLayout()
        cls_row.setContentsMargins(4, 0, 4, 0)
        cls_row.setSpacing(4)
        cls_lbl = QLabel("클래스:")
        cls_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        cls_lbl.setFixedWidth(48)
        self.class_slider = QSlider(Qt.Horizontal)
        self.class_slider.setMinimum(0)
        self.class_slider.setMaximum(max(self.class_count - 1, 0))
        self.class_slider.setValue(0)
        self.class_slider.setMaximumHeight(18)
        self.class_slider.valueChanged.connect(self._on_class_slider)
        self.class_value_label = QLabel("0")
        self.class_value_label.setStyleSheet("color:#dcdcdc; font-size:11px; min-width:80px;")
        cls_row.addWidget(cls_lbl)
        cls_row.addWidget(self.class_slider)
        cls_row.addWidget(self.class_value_label)
        cv_layout.addLayout(cls_row)

        splitter.addWidget(center)

        # ── 우측 패널: 정보 ───────────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(180)
        right.setMaximumWidth(280)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)
        rv.setSpacing(6)

        def _section(title):
            lbl = QLabel(title)
            lbl.setStyleSheet(
                "background:#333; color:#dcdcdc; font-weight:bold;"
                "padding:3px 6px; border-radius:3px; font-size:11px;")
            return lbl

        rv.addWidget(_section("📊 품질 점수"))
        self.info_score   = QLabel("—")
        self.info_issue   = QLabel("—")
        self.info_boxes   = QLabel("—")
        self.info_dup     = QLabel("—")
        for lbl in (self.info_score, self.info_issue, self.info_boxes, self.info_dup):
            lbl.setStyleSheet("padding:2px 6px; font-size:12px;")
        rv.addWidget(self.info_score)
        rv.addWidget(self.info_issue)
        rv.addWidget(self.info_boxes)
        rv.addWidget(self.info_dup)

        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#3a3a3a;")
        rv.addWidget(line)

        rv.addWidget(_section("📦 박스 목록"))
        self.box_list = QListWidget()
        self.box_list.setMaximumHeight(220)
        self.box_list.setStyleSheet(
            "QListWidget{background:#252526;font-size:11px;}"
            "QListWidget::item{padding:2px 4px;}")
        self.box_list.itemClicked.connect(self._on_box_list_clicked)
        rv.addWidget(self.box_list)

        line2 = QFrame(); line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("color:#3a3a3a;")
        rv.addWidget(line2)

        rv.addWidget(_section("📋 클래스별 박스 수"))
        self.cls_count_list = QListWidget()
        self.cls_count_list.setMaximumHeight(160)
        self.cls_count_list.setMinimumHeight(40)
        self.cls_count_list.setFont(QFont("Courier New", 11))
        self.cls_count_list.setFocusPolicy(Qt.NoFocus)
        self.cls_count_list.setStyleSheet(
            "QListWidget{background:#1a1a1a;border:1px solid #333;"
            "border-radius:4px;outline:none;}"
            "QListWidget::item{padding:2px 8px;}"
            "QListWidget::item:selected{background:transparent;}")
        rv.addWidget(self.cls_count_list)

        rv.addStretch()
        splitter.addWidget(right)

        splitter.setSizes([220, 1100, 220])

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

    def _on_sort_changed(self, _):
        self._sort_key = self.sort_combo.currentData()
        self._sort_results()
        self._populate_list()
        self._update_list_selection()

    def _sort_results(self):
        k = self._sort_key
        cur_img = self.results[self.cur_idx]["img_path"] if self.results else None
        if k == "score_asc":
            self.results.sort(key=lambda r: r["score"])
        elif k == "score_desc":
            self.results.sort(key=lambda r: r["score"], reverse=True)
        elif k == "name_asc":
            self.results.sort(key=lambda r: r["img_path"].name)
        elif k == "issue_first":
            self.results.sort(key=lambda r: (not r["is_issue"], r["score"]))
        elif k == "dup_first":
            self.results.sort(key=lambda r: (not r.get("has_duplicate", False), r["score"]))
        elif k == "unconfirmed":
            self.results.sort(key=lambda r: (r.get("confirmed", False), r["score"]))
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
                if r["cls_counts"].get(2, 0) + r["cls_counts"].get(3, 0) != 7:
                    continue

            score     = r["score"]
            confirmed = r.get("confirmed", False)
            is_issue  = r["is_issue"]
            has_dup   = r.get("has_duplicate", False)

            icon = "✅" if confirmed else ("⚠️" if is_issue else "  ")
            dup  = "🔁" if has_dup else "  "
            sc   = f"{score:.3f}"
            name = r["img_path"].name
            if len(name) > 22:
                name = name[:19] + "..."
            text = f"{icon}{dup} {name}\n      점수: {sc}"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, i)   # results 인덱스 저장

            if confirmed:
                item.setForeground(QColor("#4ec94e"))
            elif is_issue:
                item.setForeground(QColor("#e07070") if score < 0.5
                                   else QColor("#e0a070"))
            else:
                item.setForeground(QColor("#dcdcdc"))

            if score < 0.5 and not confirmed:
                item.setBackground(QColor(80, 20, 20, 80))
            elif confirmed:
                item.setBackground(QColor(20, 60, 20, 80))

            self.file_list.addItem(item)
            visible_count += 1

        self.file_list.blockSignals(False)

        # 필터 카운트 라벨 갱신
        if self._filter_classes or self._filter_cls23_eq7:
            self.filter_count_label.setText(
                f"  {visible_count} / {len(self.results)}개 이미지 표시중")
            self.filter_count_label.setVisible(True)
        else:
            self.filter_count_label.setVisible(False)

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
        idx = item.data(Qt.UserRole)
        if idx is not None and idx != self.cur_idx:
            self.save_labels()
            self._load_image_at(idx)

    # ══════════════════════════════════════════════════════════════════════════
    # 이미지 로드 / 네비게이션
    # ══════════════════════════════════════════════════════════════════════════
    def _load_image_at(self, idx: int):
        if not self.results or idx < 0 or idx >= len(self.results):
            return
        self.cur_idx = idx
        r = self.results[idx]

        score     = r["score"]
        is_issue  = r["is_issue"]
        confirmed = r.get("confirmed", False)
        issue_str = "⚠️ 의심" if is_issue else "✅ 정상"
        conf_str  = "  ✅ 확인됨" if confirmed else ""
        self.setWindowTitle(
            f"검수 - {r['img_path'].name}  [{idx+1}/{len(self.results)}]"
            f"  점수: {score:.4f}  {issue_str}{conf_str}"
        )
        self.img_nav_label.setText(f"{idx+1} / {len(self.results)}")

        sc_color = "#e07070" if score < 0.5 else ("#e0a070" if score < 0.8 else "#4ec94e")
        self.score_label.setText(
            f"<span style='color:{sc_color}'>점수: {score:.4f}</span>  {issue_str}")

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
            r["has_duplicate"] = compute_has_duplicate(r["txt_path"])

        self._update_right_panel()
        self._update_list_selection()
        # 목록 항목 아이콘 갱신 (로드 후 dup 계산 완료)
        self._refresh_list_item(idx)

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
            prev_idx = self.file_list.item(row - 1).data(Qt.UserRole)
            self._load_image_at(prev_idx)
        elif self.results:
            self.save_labels()

    def _navigate_next(self):
        row = self._current_list_row()
        if row >= 0 and row < self.file_list.count() - 1:
            self.save_labels()
            next_idx = self.file_list.item(row + 1).data(Qt.UserRole)
            self._load_image_at(next_idx)
        elif self.results:
            self.save_labels()

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
        score     = r["score"]
        confirmed = r.get("confirmed", False)
        is_issue  = r["is_issue"]
        has_dup   = r.get("has_duplicate", False)
        icon = "✅" if confirmed else ("⚠️" if is_issue else "  ")
        dup  = "🔁" if has_dup else "  "
        name = r["img_path"].name
        if len(name) > 22:
            name = name[:19] + "..."
        text = f"{icon}{dup} {name}\n      점수: {score:.3f}"
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
            elif is_issue:
                target_item.setForeground(QColor("#e07070") if score < 0.5 else QColor("#e0a070"))
                target_item.setBackground(QColor(80, 20, 20, 80) if score < 0.5 else QColor())
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
        score     = r["score"]
        confirmed = r.get("confirmed", False)
        is_issue  = r["is_issue"]
        has_dup   = r.get("has_duplicate", False)

        sc_color = "#e07070" if score < 0.5 else ("#e0a070" if score < 0.8 else "#4ec94e")
        self.info_score.setText(
            f"<span style='color:{sc_color}; font-size:14px;'><b>{score:.4f}</b></span>")
        self.info_issue.setText("⚠️ <span style='color:#e07070;'>의심 라벨</span>" if is_issue
                                else "✅ <span style='color:#4ec94e;'>정상</span>")
        self.info_boxes.setText(f"박스 수: <b>{r['box_count']}</b>")
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
            item = QListWidgetItem(f"[{cls:2d}]  {name:<16}  × {cnt[cls]}")
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
        r["has_duplicate"] = compute_has_duplicate(path)
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
        issue_str = "⚠️ 의심" if r["is_issue"] else "✅ 정상"
        conf_str  = "  ✅ 확인됨" if confirmed else ""
        self.setWindowTitle(
            f"검수 - {r['img_path'].name}  [{self.cur_idx+1}/{len(self.results)}]"
            f"  점수: {r['score']:.4f}  {issue_str}{conf_str}"
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
        self.zoom = min(cw / self.w, ch / self.h)
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
                if _is_same_box_pair(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j):
                    dups.add(i)
                    dups.add(j)
        return dups

    # ══════════════════════════════════════════════════════════════════════════
    # 마우스 이벤트 (캔버스 좌표계 - label.mapFrom 불필요)
    # ══════════════════════════════════════════════════════════════════════════
    def _canvas_mouse_press(self, event):
        pos = event.pos()

        if event.modifiers() == Qt.ControlModifier and event.button() == Qt.LeftButton:
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

        if event.button() == Qt.LeftButton and self.panning:
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
            new_zoom = min(self.zoom * 1.1, 20.0) if dy > 0 else max(self.zoom * 0.9, 0.1)
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

    def _handle_key(self, event):
        key = event.key()

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
        self.update_display()

    def toggle_class_preview(self):
        self.show_class_preview = not self.show_class_preview
        self.preview_btn.setText(
            "클래스 미리보기 ON (P)" if self.show_class_preview else "클래스 미리보기 OFF (P)")
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

        # 현재 클래스 오버레이 (좌상단)
        cls  = self.current_class
        cc   = get_color_for_class(cls)
        lbl  = f"Class {cls}"
        if cls < len(self.class_names):
            lbl += f": {self.class_names[cls]}"
        or_  = QRectF(8, 8, 220, 30)
        painter.fillRect(or_, QColor(0, 0, 0, 160))
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawRect(or_)
        fnt2 = painter.font(); fnt2.setPointSize(10); fnt2.setBold(True)
        painter.setFont(fnt2)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(cc))
        painter.drawEllipse(QPointF(24, 23), 7, 7)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(220, 220, 220)))
        painter.drawText(or_.adjusted(40, 0, 0, 0), Qt.AlignVCenter, lbl)

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
        cls = self.current_class
        cc  = get_color_for_class(cls)
        lr  = QRectF(px-5, py+size+5, size+10, 22)
        painter.fillRect(lr, QColor(0, 0, 0, 180))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(cc))
        painter.drawEllipse(QPointF(px+10, py+size+16), 5, 5)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(220, 220, 220)))
        lbl = f"Class {cls}"
        if cls < len(self.class_names):
            lbl += f": {self.class_names[cls]}"
        painter.drawText(QRectF(px+20, py+size+5, size-15, 22), Qt.AlignVCenter, lbl)

    # ══════════════════════════════════════════════════════════════════════════
    # 폴더 선택 / 검수
    # ══════════════════════════════════════════════════════════════════════════
    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 + txt 폴더 선택")
        if not folder:
            return
        self.selected_folder = Path(folder).resolve()
        self.folder_label.setText(str(self.selected_folder))
        self.folder_label.setStyleSheet(
            "padding:3px 8px; border:1px solid #3a3a3a; border-radius:4px;"
            "background:#1e1e1e; color:#dcdcdc;")
        self.btn_review.setEnabled(True)
        self.btn_zip_txt.setEnabled(True)
        self.zip_name_edit.setText(self.selected_folder.name)
        self.results = []
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
        if not MODEL_PATH.exists():
            QMessageBox.critical(self, "오류", f"모델 없음:\n{MODEL_PATH}")
            return
        self.btn_review.setEnabled(False)
        self.results = []
        self.file_list.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = ReviewWorker(self.selected_folder, self._num_classes,
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
        self.worker = ReviewWorker(self.selected_folder, self._num_classes,
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
        self.worker = ReviewWorker(self.selected_folder, self._num_classes,
                                   file_filter=file_filter,
                                   auto_confirm=self.chk_auto_confirm.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_sel_review_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, cur, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(cur)
        self.status_label.setText(f"추론 중... {cur} / {total}")

    def _on_finished(self, results, from_cache=False):
        self.results = results
        self.progress_bar.setVisible(False)
        self.btn_review.setEnabled(True)

        if not from_cache:
            save_cache(self.selected_folder, results)

        # dup 미리 계산
        for r in self.results:
            if "has_duplicate" not in r:
                r["has_duplicate"] = compute_has_duplicate(r["txt_path"])

        self._sort_results()
        self._populate_list()

        has = bool(self.results)
        self.btn_confirm_all.setEnabled(has)
        self.btn_confirm_sel.setEnabled(has)
        self.btn_unconfirm_all.setEnabled(has)
        self._refresh_dup_btn()
        self._refresh_partial_btn()
        self._update_summary()
        self.status_label.setText("완료. 목록 클릭 → 이미지 편집")

        if self.results:
            if self.file_list.count() > 0:
                first_idx = self.file_list.item(0).data(Qt.UserRole)
                self._load_image_at(first_idx)

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
                r["has_duplicate"] = compute_has_duplicate(r["txt_path"])
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
                r["has_duplicate"] = compute_has_duplicate(r["txt_path"])
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
        """현재 이미지만 확인 처리 (좌측 목록 선택 행 없으므로 cur_idx 기준)."""
        if not self.results:
            return
        r = self.results[self.cur_idx]
        r["confirmed"] = True
        self.confirm_btn.blockSignals(True)
        self.confirm_btn.setChecked(True)
        self._apply_confirm_style(True)
        self.confirm_btn.blockSignals(False)
        save_cache(self.selected_folder, self.results)
        self._refresh_list_item(self.cur_idx)
        self._update_summary()

    # ══════════════════════════════════════════════════════════════════════════
    # 중복 박스 일괄 제거
    # ══════════════════════════════════════════════════════════════════════════
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
            r["has_duplicate"] = compute_has_duplicate(r["txt_path"])
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
        ni = sum(r["is_issue"]                 for r in self.results)
        nc = sum(r.get("confirmed", False)     for r in self.results)
        nd = sum(r.get("has_duplicate", False) for r in self.results)
        av = sum(r["score"]                    for r in self.results) / n
        self.summary_label.setText(
            f"총 {n}개  |  의심 {ni}  |  중복 {nd}\n"
            f"확인 {nc}/{n}  |  평균 {av:.3f}")

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
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ReviewApp()
    win.show()
    sys.exit(app.exec_())
