"""
라벨 검수 GUI - v2
- 폴더 선택 (이미지 + txt 동일 폴더)
- '전부 검수' → Cleanlab 품질 점수 계산, 결과를 .review_cache.pkl 에 저장
- 재실행 시 캐시 자동 로드 (수정된 파일 있으면 자동 무효화)
- 테이블: 점수/의심여부/박스수 (헤더 클릭 정렬)
- 더블클릭 → 수동 라벨링 프로그램(run.py)과 동일한 UI/단축키 뷰어
"""
import sys
import os
import pickle
import time
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def get_device() -> str:
    """환경에 따라 최적 디바이스 자동 감지 (mps → cuda → cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QMessageBox, QSlider, QSizePolicy,
    QAbstractItemView
)
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QFont, QPainter, QPen, QBrush, QCursor
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QPointF, QRectF

# ── 경로 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent   # = 프로젝트 루트
PROJECT_ROOT = SCRIPT_DIR
MODEL_PATH   = PROJECT_ROOT / "auto/runs/detect/train/weights/best.pt"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
CONFIG_DIR   = SCRIPT_DIR / "config"
REFS_DIR     = SCRIPT_DIR / "refs"


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════
def load_class_names_list(config_dir: Path = None):
    """obj.names → 리스트. 없으면 dataset.yaml fallback."""
    config_dir = config_dir or CONFIG_DIR
    names_path = config_dir / "obj.names"
    if names_path.exists():
        with open(names_path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    if DATASET_YAML.exists():
        with open(DATASET_YAML) as f:
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
    """txt 파일에서 클래스별 박스 수 집계 → 'LP×2, SI×1' 형태 문자열 반환."""
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
    """SCRIPT_DIR 아래 config* 폴더 목록 반환."""
    return sorted([p for p in SCRIPT_DIR.iterdir()
                   if p.is_dir() and p.name.startswith("config")])


def get_color_for_class(cls: int) -> QColor:
    colors = [
        QColor(0, 255, 0),   QColor(255, 0, 0),   QColor(0, 0, 255),
        QColor(208, 177, 52), QColor(255, 0, 255), QColor(0, 255, 255),
        QColor(255, 165, 0), QColor(128, 0, 128),  QColor(0, 128, 128),
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
    """YOLO txt → [(cls, x1, y1, x2, y2)] 절대 픽셀 좌표."""
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


# 기준: IoU >= 0.85  AND  박스 크기 비율 >= 0.75 (단순 포함 관계 제외)
_DUP_IOU_THRESH   = 0.85
_DUP_AREA_RATIO   = 0.75


def _is_same_box_pair(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j,
                      iou_thresh=_DUP_IOU_THRESH,
                      area_ratio_thresh=_DUP_AREA_RATIO) -> bool:
    """두 박스가 '같은 물체를 다른 클래스로 중복 감지'한 쌍인지 판정.

    조건:
      1. IoU >= iou_thresh   (거의 동일한 위치/크기)
      2. min_area / max_area >= area_ratio_thresh  (크기도 비슷해야 함)
         → 큰 박스가 작은 박스를 완전히 포함하는 케이스는 제외
    """
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
    # 크기 비율 검사 (둘 다 0 이상 보장)
    if ai <= 0 or aj <= 0:
        return False
    area_ratio = min(ai, aj) / max(ai, aj)
    return area_ratio >= area_ratio_thresh


def compute_has_duplicate(txt_path: Path) -> bool:
    """txt 파일 내 박스 중 클래스가 다르고 거의 동일한 위치/크기인 쌍이 있으면 True."""
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
    """txt 내 중복 박스 제거.
    각 중복 쌍(다른 클래스, 거의 동일 위치/크기)에서 인덱스가 낮은 쪽(맨 아래 레이어)을 유지,
    높은 쪽을 제거. 제거된 박스 수 반환."""
    if not txt_path.exists():
        return 0
    lines = [l for l in txt_path.read_text().strip().splitlines() if l.strip()]
    raw = []   # (class, cx, cy, w, h)
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            c = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            raw.append((c, cx, cy, w, h))
    n = len(raw)
    if n == 0:
        return 0

    # xyxy 좌표 (정규화)
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
                to_remove.add(j)   # i(낮은 인덱스=맨 아래 레이어) 유지, j 제거

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


def save_cache(folder: Path, results: list) -> tuple[bool, str]:
    """캐시 저장. (성공여부, 메시지) 반환."""
    try:
        cp = _cache_path(folder)
        data = {
            "version": CACHE_VERSION,
            "folder":  str(folder.resolve()),  # 심볼릭 링크 정규화
            "model":   str(MODEL_PATH),
            "ts":      time.time(),
            "results": results,
        }
        with open(cp, "wb") as f:
            pickle.dump(data, f)
        size_kb = cp.stat().st_size / 1024
        print(f"[캐시 저장] {cp}  ({size_kb:.1f} KB)")
        return True, f"캐시 저장 완료 ({size_kb:.0f} KB)"
    except Exception as e:
        print(f"[캐시 저장 실패] {e}")
        return False, f"캐시 저장 실패: {e}"


def load_cache(folder: Path):
    """(results, msg) 반환. 유효하지 않으면 results=None.
    수정 파일 감지는 _get_modified_files() 에서 별도 처리 — 여기선 하지 않음."""
    cp = _cache_path(folder)
    if not cp.exists():
        return None, "캐시 없음"
    try:
        with open(cp, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != CACHE_VERSION:
            return None, "캐시 버전 불일치"
        if data.get("folder") != str(folder.resolve()):  # 심볼릭 링크 정규화
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
# 검수 워커
# ══════════════════════════════════════════════════════════════════════════════
class ReviewWorker(QThread):
    progress   = pyqtSignal(int, int)
    status_msg = pyqtSignal(str)
    finished   = pyqtSignal(list)
    error      = pyqtSignal(str)

    def __init__(self, folder: Path, num_classes: int = 31, conf: float = 0.01,
                 file_filter: set | None = None, auto_confirm: bool = True):
        super().__init__()
        self.folder       = folder
        self.num_classes  = num_classes
        self.conf         = conf
        self.file_filter  = file_filter   # 파일명 집합, None 이면 전체 처리
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
# 이미지 뷰어 (run.py 의 ImageMarker 와 동일한 UI/단축키)
# ══════════════════════════════════════════════════════════════════════════════
class ImageReviewWindow(QMainWindow):
    """
    run.py ImageMarker 와 완전히 동일한 UI/단축키.
    results 리스트를 받아 내부에서 슬라이더로 탐색.
    타이틀에 품질 점수와 의심 여부를 추가 표시.
    """
    confirmed_changed = pyqtSignal(int, bool)   # (results 인덱스, confirmed)

    def __init__(self, results: list, start_index: int = 0, parent=None,
                 config_dir: Path = None):
        super().__init__(parent)
        self.results       = results          # 정렬된 검수 결과 목록
        self.image_index   = start_index
        self.config_dir    = config_dir or CONFIG_DIR
        self.class_names   = load_class_names_list(self.config_dir)
        self.class_count   = load_class_count(self.config_dir)

        # ── run.py 와 동일한 상태 변수 ────────────────────────────────────
        self.dragging_box_index  = None
        self.drag_start_pos      = None
        self.resizing_box_index  = None
        self.resize_handle       = None
        self.panning             = False
        self.pan_start_pos       = None
        self.HANDLE_SIZE         = 8
        self.minimap_size        = 200
        self.show_minimap        = True
        self.show_class_preview  = True
        self.class_preview_pixmap = None
        self.modified_boxes      = set()
        self.undo_stack          = []
        self.redo_stack          = []
        self._active_edit_snapshot = None

        self.zoom        = 1.0
        self.offset      = QPoint(0, 0)
        self.start_point = None
        self.end_point   = None
        self.drawing     = False
        self.current_class = 0
        self.boxes       = []

        # ── 다중 선택 상태 ───────────────────────────────────────────────
        self.selected_boxes = set()   # 선택된 박스 인덱스 집합
        self.selecting      = False   # 고무줄 선택 드래그 중
        self.select_start   = None    # 고무줄 시작 (image coords)
        self.select_end     = None    # 고무줄 끝   (image coords)

        self._build_ui()
        self._load_image_at(self.image_index)

    # ── UI 구성 (run.py 와 동일) ──────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle("NeuralD BoxLabeler v1.5 - 검수 모드")
        self.resize(1200, 800)
        self.setMinimumSize(960, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 2, 0, 0)  # 불필요한 여백 제거
        layout.setSpacing(2)

        # 버튼 행
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(4, 0, 4, 0)
        btn_layout.setSpacing(4)

        self.help_btn = QPushButton("도움말 (H)")
        self.help_btn.setFixedHeight(24)
        self.help_btn.clicked.connect(self.show_guide)
        btn_layout.addWidget(self.help_btn)

        self.minimap_btn = QPushButton("미니맵 ON (M)")
        self.minimap_btn.setFixedHeight(24)
        self.minimap_btn.clicked.connect(self.toggle_minimap)
        btn_layout.addWidget(self.minimap_btn)

        self.preview_btn = QPushButton("클래스 미리보기 (P)")
        self.preview_btn.setFixedHeight(24)
        self.preview_btn.clicked.connect(self.toggle_class_preview)
        btn_layout.addWidget(self.preview_btn)

        # 확인 토글 버튼
        self.confirm_btn = QPushButton("확인 (V)")
        self.confirm_btn.setFixedHeight(24)
        self.confirm_btn.setCheckable(True)
        self.confirm_btn.clicked.connect(self._toggle_confirmed)
        btn_layout.addWidget(self.confirm_btn)

        # 검수 점수 표시
        self.score_label = QLabel("")
        self.score_label.setFixedHeight(24)
        self.score_label.setStyleSheet("font-weight:bold; padding:0 8px;")
        btn_layout.addWidget(self.score_label)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)  # slider_layout 중간 레이어 제거

        # Image 슬라이더
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(max(len(self.results) - 1, 0))
        self.image_slider.setMaximumHeight(20)
        self.image_slider.valueChanged.connect(self._slider_image_changed)
        self.image_value_display = QLabel("0")
        self.image_value_display.setMaximumHeight(20)
        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        row.addWidget(QLabel("Image num:"))
        row.addWidget(self.image_slider)
        row.addWidget(self.image_value_display)
        layout.addLayout(row)

        # Class 슬라이더
        self.class_slider = QSlider(Qt.Horizontal)
        self.class_slider.setMinimum(0)
        self.class_slider.setMaximum(self.class_count - 1)
        self.class_slider.setValue(0)
        self.class_slider.setMaximumHeight(20)
        self.class_slider.valueChanged.connect(self._slider_class_changed)
        self.class_value_display = QLabel("0")
        self.class_value_display.setMaximumHeight(20)
        row2 = QHBoxLayout()
        row2.setContentsMargins(4, 0, 4, 0)
        row2.addWidget(QLabel("Object id:"))
        row2.addWidget(self.class_slider)
        row2.addWidget(self.class_value_display)
        layout.addLayout(row2)

        # 캔버스
        self.label = QLabel(self)
        self.label.setMinimumSize(640, 480)
        self.label.setMouseTracking(True)
        layout.addWidget(self.label)

    # ── 이미지 로드 ───────────────────────────────────────────────────────
    def _load_image_at(self, idx: int):
        if idx < 0 or idx >= len(self.results):
            return

        self.image_index = idx
        r = self.results[idx]

        score     = r["score"]
        is_issue  = r["is_issue"]
        confirmed = r.get("confirmed", False)
        fname     = r["img_path"].name
        issue_str = "⚠️ 의심" if is_issue else "✅ 정상"
        conf_str  = "  |  ✅ 확인됨" if confirmed else ""
        self.setWindowTitle(
            f"NeuralD BoxLabeler v1.5 - {fname} "
            f"[{idx+1}/{len(self.results)}]  |  점수: {score:.4f}  |  {issue_str}{conf_str}"
        )
        score_color = "#cc0000" if score < 0.5 else ("#e07000" if score < 0.8 else "#007700")
        self.score_label.setText(
            f"<span style='color:{score_color}'>점수: {score:.4f}</span>  {issue_str}"
        )
        # 확인 버튼 상태 동기화
        self.confirm_btn.blockSignals(True)
        self.confirm_btn.setChecked(confirmed)
        self._apply_confirm_style(confirmed)
        self.confirm_btn.blockSignals(False)

        self.image_slider.blockSignals(True)
        self.image_slider.setValue(idx)
        self.image_value_display.setText(str(idx))
        self.image_slider.blockSignals(False)

        img = imread_unicode(str(r["img_path"]))
        if img is None:
            return
        self.image = img
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

    # ── 슬라이더 핸들러 ───────────────────────────────────────────────────
    def _slider_image_changed(self, value):
        if self.image_index != value:
            self.save_labels()
            self.boxes = []
            self.modified_boxes.clear()
            self._load_image_at(value)

    def _slider_class_changed(self, value):
        self.current_class = value
        if value < len(self.class_names):
            self.class_value_display.setText(f"{value}: {self.class_names[value]}")
        else:
            self.class_value_display.setText(str(value))
        self._load_class_preview_pixmap(value)
        self.update_display()

    # ── 확인 토글 ─────────────────────────────────────────────────────────
    def _toggle_confirmed(self):
        confirmed = self.confirm_btn.isChecked()
        self.results[self.image_index]["confirmed"] = confirmed
        self._apply_confirm_style(confirmed)
        # 타이틀 갱신
        r         = self.results[self.image_index]
        conf_str  = "  |  ✅ 확인됨" if confirmed else ""
        issue_str = "⚠️ 의심" if r["is_issue"] else "✅ 정상"
        self.setWindowTitle(
            f"NeuralD BoxLabeler v1.5 - {r['img_path'].name} "
            f"[{self.image_index+1}/{len(self.results)}]  |  "
            f"점수: {r['score']:.4f}  |  {issue_str}{conf_str}"
        )
        # 메인 테이블에 반영
        self.confirmed_changed.emit(self.image_index, confirmed)

    # ── 중복 박스 감지 (다른 클래스인데 거의 동일 위치/크기) ──────────────
    def _compute_duplicate_boxes(self) -> set:
        """_is_same_box_pair 기준으로 중복인 박스 인덱스 집합 반환."""
        dups = set()
        n = len(self.boxes)
        for i in range(n):
            cls_i, x1i, y1i, x2i, y2i = self.boxes[i]
            for j in range(i + 1, n):
                cls_j, x1j, y1j, x2j, y2j = self.boxes[j]
                if cls_i == cls_j:
                    continue
                if _is_same_box_pair(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j):
                    dups.add(i)
                    dups.add(j)
        return dups

    def _apply_confirm_style(self, confirmed: bool):
        if confirmed:
            self.confirm_btn.setText("✅ 확인됨 (V)")
            self.confirm_btn.setStyleSheet(
                "QPushButton{background:#28a745;color:white;border-radius:3px;font-weight:bold;}"
                "QPushButton:hover{background:#218838;}"
            )
        else:
            self.confirm_btn.setText("확인 (V)")
            self.confirm_btn.setStyleSheet("")

    # ── 저장 ─────────────────────────────────────────────────────────────
    def save_labels(self):
        if not hasattr(self, "h") or not self.results:
            return
        r    = self.results[self.image_index]
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
        # 박스 수 업데이트
        self.results[self.image_index]["box_count"] = len(self.boxes)
        print(f"✅ 저장: {path.name}")

    # ── Undo / Redo ───────────────────────────────────────────────────────
    def _serialize_state(self):
        return {"boxes": [list(b) for b in self.boxes],
                "modified_boxes": sorted(self.modified_boxes)}

    def _apply_state(self, state):
        self.boxes = [tuple(b) for b in state.get("boxes", [])]
        self.modified_boxes = set(state.get("modified_boxes", []))
        self.update_display()

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

    # ── 미니맵 / 클래스 프리뷰 ───────────────────────────────────────────
    def toggle_minimap(self):
        self.show_minimap = not self.show_minimap
        self.minimap_btn.setText("미니맵 ON (M)" if self.show_minimap else "미니맵 OFF (M)")
        self.update_display()

    def toggle_class_preview(self):
        self.show_class_preview = not self.show_class_preview
        self.preview_btn.setText(
            "클래스 미리보기 ON (P)" if self.show_class_preview else "클래스 미리보기 OFF (P)"
        )
        self.update_display()

    def _load_class_preview_pixmap(self, cls):
        # config_dir 안에 refs 서브폴더가 있으면 우선, 없으면 공용 REFS_DIR
        refs = self.config_dir / "refs"
        if not refs.is_dir():
            refs = REFS_DIR
        img_path = refs / f"{cls}.png"
        if img_path.exists():
            self.class_preview_pixmap = QPixmap(str(img_path))
        else:
            self.class_preview_pixmap = None

    # ── Zoom / Fit ────────────────────────────────────────────────────────
    def reset_zoom(self):
        if hasattr(self, "w") and hasattr(self, "h"):
            self.zoom = min(self.label.width() / self.w, self.label.height() / self.h)
            self.offset = QPoint(
                (self.label.width()  - int(self.w * self.zoom)) // 2,
                (self.label.height() - int(self.h * self.zoom)) // 2,
            )
            self.update_display()

    def showEvent(self, event):
        super().showEvent(event)
        self.reset_zoom()

    def resizeEvent(self, event):
        self.reset_zoom()

    # ── 도움말 (run.py 동일) ──────────────────────────────────────────────
    def show_guide(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("NeuralD BoxLabeler v1.5 조작 가이드")
        msg.setTextFormat(Qt.RichText)
        msg.setText("""
        <h3>🎮 마우스 조작 (Mouse)</h3>
        <ul>
            <li><b>좌클릭 드래그</b> : 박스 그리기</li>
            <li><b>Ctrl + 좌클릭 드래그</b> : 이미지 이동 (Panning)</li>
            <li><b>우클릭 드래그</b> : 박스 위치 이동</li>
            <li><b>박스 모서리 드래그</b> : 박스 크기 조절 (Resize)</li>
            <li><b>마우스 휠 (세로)</b> : 확대 / 축소</li>
            <li><b>마우스 휠 (가로)</b> : 박스 위면 해당 박스 클래스 변경 / 아니면 현재 클래스 변경</li>
        </ul>
        <hr>
        <h3>⌨️ 키보드 단축키 (Keyboard)</h3>
        <table border="0" cellpadding="2">
            <tr><td><b>→ / D / Enter</b></td><td>: 저장 &amp; 다음 이미지</td></tr>
            <tr><td><b>← / A</b></td><td>: 저장 &amp; 이전 이미지</td></tr>
            <tr><td><b>M</b></td><td>: 미니맵 ON/OFF</td></tr>
            <tr><td><b>S</b></td><td>: 저장 (Save)</td></tr>
            <tr><td><b>F</b></td><td>: 화면 맞춤 (Fit to Screen)</td></tr>
            <tr><td><b>R</b></td><td>: 마우스 위치 박스 삭제 (Remove)</td></tr>
            <tr><td><b>BackSpace</b></td><td>: 마지막 박스 삭제</td></tr>
            <tr><td><b>C</b></td><td>: 전체 박스 삭제 (Clear All)</td></tr>
            <tr><td><b>Z</b></td><td>: 실행 취소 (Undo)</td></tr>
            <tr><td><b>U</b></td><td>: 다시 실행 (Redo)</td></tr>
            <tr><td><b>0 ~ 9</b></td><td>: 클래스 ID 변경</td></tr>
            <tr><td><b>H</b></td><td>: 도움말 보기</td></tr>
            <tr><td><b>P</b></td><td>: 클래스 미리보기 ON/OFF</td></tr>
            <tr><td><b>Q / Esc</b></td><td>: 창 닫기</td></tr>
        </table>""")
        msg.exec_()

    # ── 박스 유틸 ─────────────────────────────────────────────────────────
    def get_box_index_at_label_pos(self, label_pos):
        pos = label_pos - self.offset
        x = pos.x() / self.zoom
        y = pos.y() / self.zoom
        for i in reversed(range(len(self.boxes))):
            _, x1, y1, x2, y2 = self.boxes[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    def get_resize_handle_at(self, pos):
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

    def _rebuild_after_delete(self, deleted: list):
        """sorted(ascending) 인덱스 목록 삭제 후 modified_boxes / selected_boxes 재계산."""
        deleted_set = set(deleted)
        def shifted(idx):
            return idx - sum(1 for d in deleted if d < idx)
        self.modified_boxes = {shifted(i) for i in self.modified_boxes if i not in deleted_set}
        self.selected_boxes = {shifted(i) for i in self.selected_boxes if i not in deleted_set}

    # ── 마우스 이벤트 (run.py 동일) ───────────────────────────────────────
    def mousePressEvent(self, event):
        label_pos = self.label.mapFrom(self, event.pos())

        if event.modifiers() == Qt.ControlModifier and event.button() == Qt.LeftButton:
            self.panning = True
            self.pan_start_pos = label_pos
            self.setCursor(Qt.ClosedHandCursor)
            return

        pos = label_pos - self.offset
        x = pos.x() / self.zoom
        y = pos.y() / self.zoom

        # ── Shift+좌클릭: 개별 토글 또는 고무줄 시작 ─────────────────────
        if event.modifiers() & Qt.ShiftModifier and event.button() == Qt.LeftButton:
            hit = self.get_box_index_at_label_pos(label_pos)
            if hit is not None:
                if hit in self.selected_boxes:
                    self.selected_boxes.discard(hit)
                else:
                    self.selected_boxes.add(hit)
                self.update_display()
            else:
                self.selecting    = True
                self.select_start = (x, y)
                self.select_end   = (x, y)
            return

        if event.button() == Qt.LeftButton:
            # 일반 좌클릭 → 선택 해제
            if self.selected_boxes:
                self.selected_boxes.clear()
            idx, handle = self.get_resize_handle_at(label_pos)
            if idx is not None:
                self.resizing_box_index = idx
                self.resize_handle = handle
                self._active_edit_snapshot = self._serialize_state()
                self.drawing = False
                return
            if 0 <= x < self.w and 0 <= y < self.h:
                self.start_point = (x, y)
                self.end_point = None
                self.drawing = True

        elif event.button() == Qt.RightButton:
            for i in reversed(range(len(self.boxes))):
                cls, x1, y1, x2, y2 = self.boxes[i]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.dragging_box_index = i
                    self.drag_start_pos = (x, y)
                    self._active_edit_snapshot = self._serialize_state()
                    break

    def mouseMoveEvent(self, event):
        label_pos = self.label.mapFrom(self, event.pos())

        if self.panning:
            delta = label_pos - self.pan_start_pos
            self.offset += delta
            self.pan_start_pos = label_pos
            self.update_display()
            return

        pos = label_pos - self.offset
        x = max(0.0, min(float(self.w), pos.x() / self.zoom))
        y = max(0.0, min(float(self.h), pos.y() / self.zoom))

        # 고무줄 선택 드래그
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
            # 드래그 대상이 선택 집합에 있으면 선택 전체 이동, 아니면 단독 이동
            targets = (self.selected_boxes
                       if self.dragging_box_index in self.selected_boxes
                       else {self.dragging_box_index})
            for i in targets:
                cls, x1, y1, x2, y2 = self.boxes[i]
                self.boxes[i] = (cls, x1+dx, y1+dy, x2+dx, y2+dy)
                self.modified_boxes.add(i)
            self.drag_start_pos = (x, y)
            self.update_display()
            return

        idx, handle = self.get_resize_handle_at(label_pos)
        if idx is not None:
            self.setCursor(Qt.SizeFDiagCursor if handle in ("TL","BR") else Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        # 고무줄 선택 완료
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
            return

        if event.button() == Qt.LeftButton and self.panning:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
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
            return

        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            label_pos = self.label.mapFrom(self, event.pos())
            pos = label_pos - self.offset
            cx = max(0.0, min(float(self.w), pos.x() / self.zoom))
            cy = max(0.0, min(float(self.h), pos.y() / self.zoom))
            x1, y1 = self.start_point
            rx1, rx2 = min(x1, cx), max(x1, cx)
            ry1, ry2 = min(y1, cy), max(y1, cy)
            if abs(rx2-rx1) > 3 and abs(ry2-ry1) > 3:
                self._push_undo_state()
                ni = len(self.boxes)
                self.boxes.append((self.current_class, rx1, ry1, rx2, ry2))
                self.modified_boxes.add(ni)
            self.update_display()

        elif event.button() == Qt.RightButton:
            if self._active_edit_snapshot and self._active_edit_snapshot != self._serialize_state():
                self.undo_stack.append(self._active_edit_snapshot)
                self.redo_stack.clear()
            self.dragging_box_index = None
            self.drag_start_pos = None
            self._active_edit_snapshot = None

    # ── 휠 (run.py 동일) ──────────────────────────────────────────────────
    def wheelEvent(self, event):
        dy = event.angleDelta().y()
        dx = event.angleDelta().x()
        label_pos = self.label.mapFrom(self, event.pos())

        if dx != 0:
            box_index = self.get_box_index_at_label_pos(label_pos)
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
            old_pos = label_pos - self.offset
            rx = old_pos.x() / (self.w * self.zoom)
            ry = old_pos.y() / (self.h * self.zoom)
            self.zoom = new_zoom
            self.offset = QPoint(
                int(label_pos.x() - self.w * self.zoom * rx),
                int(label_pos.y() - self.h * self.zoom * ry),
            )
            self.update_display()

    # ── 키보드 (run.py 동일) ──────────────────────────────────────────────
    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_S:
            self.save_labels()

        elif key in (Qt.Key_Right, Qt.Key_Return, Qt.Key_N) or key == 0x44:  # D
            if self.image_index < len(self.results) - 1:
                self.image_slider.setValue(self.image_index + 1)
            else:
                self.save_labels()

        elif key == Qt.Key_Left or key == 0x41:  # A
            if self.image_index > 0:
                self.image_slider.setValue(self.image_index - 1)
            else:
                self.save_labels()

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

        elif key == Qt.Key_Backspace:
            if self.selected_boxes:
                self._push_undo_state()
                for i in sorted(self.selected_boxes, reverse=True):
                    del self.boxes[i]
                self._rebuild_after_delete(sorted(self.selected_boxes))
                self.selected_boxes.clear()
                self.update_display()
            elif self.boxes:
                self._push_undo_state()
                self.boxes.pop()
                self.update_display()

        elif key == Qt.Key_C:
            if self.boxes:
                self._push_undo_state()
                self.boxes.clear()
                self.modified_boxes.clear()
                self.update_display()

        elif key == Qt.Key_R:
            if self.selected_boxes:
                self._push_undo_state()
                for i in sorted(self.selected_boxes, reverse=True):
                    del self.boxes[i]
                self._rebuild_after_delete(sorted(self.selected_boxes))
                self.selected_boxes.clear()
                self.update_display()
            else:
                cursor_pos = self.mapFromGlobal(QCursor.pos())
                label_pos  = self.label.mapFrom(self, cursor_pos)
                pos = label_pos - self.offset
                x, y = pos.x() / self.zoom, pos.y() / self.zoom
                for i in reversed(range(len(self.boxes))):
                    _, x1, y1, x2, y2 = self.boxes[i]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self._push_undo_state()
                        del self.boxes[i]
                        self._rebuild_after_delete([i])
                        self.update_display()
                        break

        elif key == 0x60:  # backtick
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
                else:
                    cursor_pos = self.mapFromGlobal(QCursor.pos())
                    label_pos  = self.label.mapFrom(self, cursor_pos)
                    self.set_box_class(self.get_box_index_at_label_pos(label_pos), tc)

        elif key == Qt.Key_V:
            self.confirm_btn.setChecked(not self.confirm_btn.isChecked())
            self._toggle_confirmed()

        elif key == Qt.Key_Escape:
            if self.selected_boxes:
                self.selected_boxes.clear()
                self.update_display()
            else:
                self.close()

        elif key == Qt.Key_Q:
            self.close()

    def closeEvent(self, event):
        self.save_labels()
        super().closeEvent(event)

    # ── 그리기 (run.py 동일) ──────────────────────────────────────────────
    def draw_corner_bracket(self, painter, x, y, size, angle):
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        painter.drawLine(QPointF(0, 0), QPointF(size, 0))
        painter.drawLine(QPointF(0, 0), QPointF(0, size))
        painter.restore()

    def draw_minimap(self, painter):
        if not self.show_minimap or not hasattr(self, "qimage"):
            return
        margin = 10
        mx = self.label.width()  - self.minimap_size - margin
        my = self.label.height() - self.minimap_size - margin
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
        vw =  self.label.width()  / self.zoom
        vh =  self.label.height() / self.zoom
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.setBrush(QBrush(QColor(255, 255, 0, 50)))
        painter.drawRect(QRectF(mox + vx*sx, moy + vy*sy, vw*sx, vh*sy))
        for cls, x1, y1, x2, y2 in self.boxes:
            painter.setPen(QPen(get_color_for_class(cls), 1))
            painter.drawRect(QRectF(mox+x1*sx, moy+y1*sy,
                                    (x2-x1)*sx, (y2-y1)*sy))

    def draw_class_preview(self, painter):
        if not self.show_class_preview:
            return
        margin, size = 10, self.minimap_size
        px = self.label.width() - size - margin
        py = margin
        painter.fillRect(px-5, py-5, size+10, size+10, QColor(0, 0, 0, 180))
        if self.class_preview_pixmap and not self.class_preview_pixmap.isNull():
            scaled = self.class_preview_pixmap.scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(px + (size-scaled.width())//2,
                               py + (size-scaled.height())//2, scaled)
        else:
            painter.setPen(QPen(QColor(120, 120, 120)))
            painter.drawText(QRectF(px, py, size, size), Qt.AlignCenter,
                             f"No image\n({self.current_class}.png)")
        cls = self.current_class
        cc  = get_color_for_class(cls)
        lr  = QRectF(px-5, py+size+5, size+10, 22)
        painter.fillRect(lr, QColor(0, 0, 0, 180))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(cc))
        painter.drawEllipse(QPointF(px+10, py+size+16), 6, 6)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255)))
        lbl = f"Class {cls}"
        if cls < len(self.class_names):
            lbl += f": {self.class_names[cls]}"
        painter.drawText(QRectF(px+22, py+size+5, size-17, 22), Qt.AlignVCenter, lbl)

    def update_display(self):
        if not hasattr(self, "qimage"):
            return
        pix = QPixmap(self.label.size())
        pix.fill(Qt.lightGray)
        painter = QPainter(pix)

        sw = int(self.w * self.zoom)
        sh = int(self.h * self.zoom)
        scaled = self.qimage.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(self.offset, QPixmap.fromImage(scaled))

        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            color = get_color_for_class(cls)
            painter.setPen(QPen(color, 2))
            sx1 = x1 * self.zoom + self.offset.x()
            sy1 = y1 * self.zoom + self.offset.y()
            sw_ = (x2 - x1) * self.zoom
            sh_ = (y2 - y1) * self.zoom
            painter.drawRect(QRectF(sx1, sy1, sw_, sh_))
            lbl = (f"{cls} {self.class_names[cls]}"
                   if cls < len(self.class_names) else str(cls))
            painter.drawText(QPointF(sx1, sy1 - 5), lbl)

            if i in self.modified_boxes:
                painter.setPen(QPen(QColor(255, 255, 0), 3))
                bs = 15
                self.draw_corner_bracket(painter, sx1,      sy1,      bs,  0)
                self.draw_corner_bracket(painter, sx1+sw_,  sy1,      bs,  90)
                self.draw_corner_bracket(painter, sx1,      sy1+sh_,  bs, -90)
                self.draw_corner_bracket(painter, sx1+sw_,  sy1+sh_,  bs,  180)
                painter.setPen(QPen(color, 2))

            # 선택된 박스 하이라이트
            if i in self.selected_boxes:
                painter.setPen(QPen(QColor(0, 200, 255), 2, Qt.DashLine))
                painter.setBrush(QBrush(QColor(0, 200, 255, 40)))
                painter.drawRect(QRectF(sx1 - 3, sy1 - 3, sw_ + 6, sh_ + 6))
                painter.setBrush(Qt.NoBrush)

            painter.setBrush(QBrush(color))
            for cx, cy in [(sx1,sy1),(sx1+sw_,sy1),(sx1,sy1+sh_),(sx1+sw_,sy1+sh_)]:
                painter.drawEllipse(QPointF(cx, cy), 3.0, 3.0)
            painter.setBrush(Qt.NoBrush)

        # 고무줄 선택 영역
        if self.selecting and self.select_start and self.select_end:
            rx1 = min(self.select_start[0], self.select_end[0]) * self.zoom + self.offset.x()
            ry1 = min(self.select_start[1], self.select_end[1]) * self.zoom + self.offset.y()
            rw  = abs(self.select_end[0] - self.select_start[0]) * self.zoom
            rh  = abs(self.select_end[1] - self.select_start[1]) * self.zoom
            painter.setPen(QPen(QColor(0, 120, 255), 1, Qt.DashLine))
            painter.setBrush(QBrush(QColor(0, 120, 255, 30)))
            painter.drawRect(QRectF(rx1, ry1, rw, rh))
            painter.setBrush(Qt.NoBrush)

        if self.drawing and self.start_point and self.end_point:
            painter.setPen(QPen(Qt.red, 1, Qt.DashLine))
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            painter.drawRect(QRectF(
                min(x1,x2)*self.zoom+self.offset.x(),
                min(y1,y2)*self.zoom+self.offset.y(),
                abs(x2-x1)*self.zoom, abs(y2-y1)*self.zoom))

        # ── 중복 박스 하이라이트 (다른 클래스 높은 IoU) ──────────────────
        dup_indices = self._compute_duplicate_boxes()
        if dup_indices:
            dup_font = painter.font()
            dup_font.setPointSize(8)
            dup_font.setBold(True)
            painter.setFont(dup_font)
            for i in dup_indices:
                cls, x1, y1, x2, y2 = self.boxes[i]
                sx1 = x1 * self.zoom + self.offset.x()
                sy1 = y1 * self.zoom + self.offset.y()
                sw_ = (x2 - x1) * self.zoom
                sh_ = (y2 - y1) * self.zoom
                m = 4
                # 외곽 오렌지 두꺼운 점선 테두리
                painter.setPen(QPen(QColor(255, 120, 0), 3, Qt.DashLine))
                painter.setBrush(QBrush(QColor(255, 120, 0, 45)))
                painter.drawRect(QRectF(sx1 - m, sy1 - m, sw_ + m * 2, sh_ + m * 2))
                # "중복" 배지
                badge = QRectF(sx1, max(2, sy1 - 17), 38, 15)
                painter.setBrush(QBrush(QColor(255, 120, 0)))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(badge, 2, 2)
                painter.setPen(QPen(Qt.white))
                painter.drawText(badge, Qt.AlignCenter, "중복!")

        self.draw_minimap(painter)
        self.draw_class_preview(painter)

        # 현재 클래스 오버레이 (좌상단)
        cls = self.current_class
        cc  = get_color_for_class(cls)
        lbl = f"Class {cls}"
        if cls < len(self.class_names):
            lbl += f": {self.class_names[cls]}"
        or_ = QRectF(8, 8, 220, 32)
        painter.fillRect(or_, QColor(0, 0, 0, 160))
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(or_)
        font = painter.font()
        font.setPointSize(11); font.setBold(True)
        painter.setFont(font)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(cc))
        painter.drawEllipse(QPointF(26, 24), 8, 8)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(or_.adjusted(42, 0, 0, 0), Qt.AlignVCenter, lbl)

        painter.end()
        self.label.setPixmap(pix)


# ══════════════════════════════════════════════════════════════════════════════
# 메인 윈도우 (테이블)
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("라벨 검수 도구 v2")
        self.setMinimumSize(1000, 650)
        self.results      = []
        self.worker       = None
        self._sort_col    = 1
        self._sort_asc    = True
        self.selected_folder = None
        self.current_config_dir = CONFIG_DIR
        self._num_classes = load_class_count(self.current_config_dir)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # 컨트롤 행
        ctrl = QHBoxLayout()
        self.folder_label = QLabel("폴더를 선택하세요")
        self.folder_label.setStyleSheet(
            "padding:4px 8px; border:1px solid #ccc; border-radius:4px; background:#f9f9f9;")
        self.folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_sel = QPushButton("📁  폴더 선택")
        btn_sel.setFixedHeight(32)
        btn_sel.clicked.connect(self._select_folder)
        self.btn_review = QPushButton("🔍  전부 검수")
        self.btn_review.setFixedHeight(32)
        self.btn_review.setEnabled(False)
        self.btn_review.setStyleSheet(
            "QPushButton{background:#4a90d9;color:white;border-radius:4px;font-weight:bold;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover{background:#357abd;}")
        self.btn_review.clicked.connect(self._start_review)

        self.btn_partial = QPushButton("🔄  재검수 (수정분)")
        self.btn_partial.setFixedHeight(32)
        self.btn_partial.setEnabled(False)
        self.btn_partial.setStyleSheet(
            "QPushButton{background:#e07000;color:white;border-radius:4px;font-weight:bold;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover{background:#b85c00;}")
        self.btn_partial.clicked.connect(self._start_partial_review)

        self.btn_confirm_all = QPushButton("✅  전체 확인")
        self.btn_confirm_all.setFixedHeight(32)
        self.btn_confirm_all.setEnabled(False)
        self.btn_confirm_all.setStyleSheet(
            "QPushButton{background:#28a745;color:white;border-radius:4px;font-weight:bold;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover{background:#218838;}")
        self.btn_confirm_all.clicked.connect(self._confirm_all)

        self.btn_confirm_sel = QPushButton("☑️  선택 확인")
        self.btn_confirm_sel.setFixedHeight(32)
        self.btn_confirm_sel.setEnabled(False)
        self.btn_confirm_sel.setStyleSheet(
            "QPushButton{background:#5cb85c;color:white;border-radius:4px;font-weight:bold;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover{background:#4cae4c;}")
        self.btn_confirm_sel.clicked.connect(self._confirm_selected)

        self.btn_unconfirm_all = QPushButton("✖  전체 확인 해제")
        self.btn_unconfirm_all.setFixedHeight(32)
        self.btn_unconfirm_all.setEnabled(False)
        self.btn_unconfirm_all.setStyleSheet(
            "QPushButton{background:#6c757d;color:white;border-radius:4px;font-weight:bold;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover{background:#5a6268;}")
        self.btn_unconfirm_all.clicked.connect(self._unconfirm_all)

        self.btn_review_sel = QPushButton("☑️  선택 검수")
        self.btn_review_sel.setFixedHeight(32)
        self.btn_review_sel.setEnabled(False)
        self.btn_review_sel.setStyleSheet(
            "QPushButton{background:#8e44ad;color:white;border-radius:4px;font-weight:bold;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover{background:#7d3c98;}")
        self.btn_review_sel.clicked.connect(self._review_selected_items)

        ctrl.addWidget(self.folder_label)
        ctrl.addWidget(btn_sel)
        ctrl.addWidget(self.btn_review)
        ctrl.addWidget(self.btn_partial)
        ctrl.addWidget(self.btn_review_sel)
        ctrl.addWidget(self.btn_confirm_all)
        ctrl.addWidget(self.btn_confirm_sel)
        ctrl.addWidget(self.btn_unconfirm_all)
        main_layout.addLayout(ctrl)

        # 두 번째 컨트롤 행 (config 전환 + 중복 처리)
        ctrl2 = QHBoxLayout()

        from PyQt5.QtWidgets import QComboBox, QLabel as QLabel2, QCheckBox
        self.chk_auto_confirm = QCheckBox("검수 후 자동 확인 (점수=1.0)")
        self.chk_auto_confirm.setChecked(True)
        self.chk_auto_confirm.setFixedHeight(28)

        config_label = QLabel2("Config:")
        config_label.setFixedHeight(28)
        self.config_combo = QComboBox()
        self.config_combo.setFixedHeight(28)
        self.config_combo.setMinimumWidth(130)
        for cdir in find_config_dirs():
            self.config_combo.addItem(cdir.name, userData=cdir)
        # 현재 CONFIG_DIR 를 기본 선택
        default_idx = next((i for i in range(self.config_combo.count())
                            if self.config_combo.itemData(i) == CONFIG_DIR), 0)
        self.config_combo.setCurrentIndex(default_idx)
        self.config_combo.currentIndexChanged.connect(self._on_config_changed)

        self.btn_remove_dups = QPushButton("🗑️  중복 박스 일괄 제거 (아래 레이어 유지)")
        self.btn_remove_dups.setFixedHeight(28)
        self.btn_remove_dups.setEnabled(False)
        self.btn_remove_dups.setStyleSheet(
            "QPushButton{background:#c0392b;color:white;border-radius:4px;font-weight:bold;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover{background:#a93226;}")
        self.btn_remove_dups.clicked.connect(self._remove_all_duplicates)

        ctrl2.addWidget(self.chk_auto_confirm)
        ctrl2.addSpacing(16)
        ctrl2.addWidget(config_label)
        ctrl2.addWidget(self.config_combo)
        ctrl2.addSpacing(16)
        ctrl2.addWidget(self.btn_remove_dups)
        ctrl2.addStretch()
        main_layout.addLayout(ctrl2)

        # 상태 / 진행바
        self.status_label = QLabel("준비")
        self.status_label.setStyleSheet("color:#555;")
        main_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 요약
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-weight:bold; color:#333;")
        main_layout.addWidget(self.summary_label)

        # 테이블
        # col 0: 선택  col 1: 파일명  col 2: 점수  col 3: 의심  col 4: 박스수
        # col 5: 클래스 분포  col 6: 중복  col 7: 확인  col 8: 경로(hidden)
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(
            ["선택", "파일명", "품질 점수", "의심 여부", "박스 수",
             "클래스 분포", "중복", "확인", "경로"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        for c in (0, 2, 3, 4, 6, 7):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setColumnHidden(8, True)   # 경로 숨김
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(False)  # 행 색상은 confirmed로 직접 제어
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().sectionClicked.connect(self._on_header_click)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.table.itemChanged.connect(self._on_item_changed)
        main_layout.addWidget(self.table)

    # ── Config 전환 ────────────────────────────────────────────────────────
    def _on_config_changed(self, idx: int):
        cdir = self.config_combo.itemData(idx)
        if not cdir or cdir == self.current_config_dir:
            return
        self.current_config_dir = cdir
        self._num_classes = load_class_count(cdir)
        self.status_label.setText(
            f"Config 전환: {cdir.name}  ({len(load_class_names_list(cdir))}개 클래스)")

    # ── 폴더 선택 ─────────────────────────────────────────────────────────
    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 + txt 폴더 선택")
        if not folder:
            return
        self.selected_folder = Path(folder).resolve()   # 심볼릭 링크 정규화
        self.folder_label.setText(str(self.selected_folder))
        self.btn_review.setEnabled(True)
        self.table.setRowCount(0)
        self.summary_label.setText("")
        self.results = []

        # 캐시 자동 로드
        cached, msg = load_cache(self.selected_folder)
        if cached:
            self.results = cached
            # 수정된 파일 수 미리 계산해서 표시
            modified = self._get_modified_files()
            mod_str  = f"  |  수정됨: {len(modified)}개" if modified else ""
            self.status_label.setText(f"✅ 캐시 로드 완료{mod_str}  —  재검수 또는 전부 검수 가능")
            self.btn_partial.setEnabled(bool(modified))
            self._on_finished(self.results, from_cache=True)
        else:
            self.btn_partial.setEnabled(False)
            self.btn_review_sel.setEnabled(False)
            self.btn_confirm_all.setEnabled(False)
            self.btn_confirm_sel.setEnabled(False)
            self.btn_unconfirm_all.setEnabled(False)
            self.btn_remove_dups.setEnabled(False)
            self.status_label.setText(f"캐시 없음 ({msg}). '전부 검수' 버튼을 눌러주세요.")

    # ── 수정 파일 감지 헬퍼 ──────────────────────────────────────────────
    def _get_modified_files(self) -> set:
        """캐시 타임스탬프 이후 mtime이 갱신된 이미지 파일명 집합 반환."""
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
                # 수정된 경우 or 캐시에 없는 신규 파일
                if txt.stat().st_mtime > ts or p.name not in cached_names:
                    modified.add(p.name)
            return modified
        except Exception:
            return set()

    # ── 검수 시작 ─────────────────────────────────────────────────────────
    def _start_review(self):
        if not self.selected_folder:
            return
        if not MODEL_PATH.exists():
            QMessageBox.critical(self, "오류", f"모델 없음:\n{MODEL_PATH}")
            return
        self.btn_review.setEnabled(False)
        self.table.setRowCount(0)
        self.results = []
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = ReviewWorker(self.selected_folder, self._num_classes,
                                   auto_confirm=self.chk_auto_confirm.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.status_msg.connect(lambda m: self.status_label.setText(m))
        # lambda로 명시적 연결 → PyQt5 타입 매칭 오류 방지
        self.worker.finished.connect(lambda r: self._on_finished(r, from_cache=False))
        self.worker.error.connect(self._on_error)
        self.worker.start()

    # ── 재검수 (수정분만) ─────────────────────────────────────────────────
    def _start_partial_review(self):
        if not self.selected_folder or not self.results:
            return
        modified = self._get_modified_files()
        if not modified:
            QMessageBox.information(self, "재검수", "캐시 이후 수정된 파일이 없습니다.")
            return

        reply = QMessageBox.question(
            self, "재검수 확인",
            f"수정된 파일 {len(modified)}개만 재검수합니다.\n계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.btn_review.setEnabled(False)
        self.btn_partial.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"재검수 준비 중... ({len(modified)}개 파일)")

        self.worker = ReviewWorker(self.selected_folder, self._num_classes,
                                   file_filter=modified,
                                   auto_confirm=self.chk_auto_confirm.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.status_msg.connect(lambda m: self.status_label.setText(m))
        self.worker.finished.connect(lambda r: self._on_partial_finished(r))
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_partial_finished(self, new_results: list):
        """재검수 결과를 기존 캐시에 병합."""
        self.progress_bar.setVisible(False)
        self.btn_review.setEnabled(True)
        self.btn_partial.setEnabled(False)

        # 파일명 기준으로 기존 결과 덮어쓰기
        new_map = {r["img_path"].name: r for r in new_results}
        merged  = []
        updated = 0
        added   = 0
        existing_names = {r["img_path"].name for r in self.results}

        for r in self.results:
            name = r["img_path"].name
            if name in new_map:
                new_r = new_map.pop(name)
                # 수동으로 confirmed 했던 것은 재검수 후에도 유지
                if r.get("confirmed") and not new_r.get("confirmed"):
                    new_r["confirmed"] = True
                merged.append(new_r)
                updated += 1
            else:
                merged.append(r)

        # 기존 캐시에 없던 신규 파일 추가
        for r in new_map.values():
            merged.append(r)
            added += 1

        self.results = merged
        ok, cache_msg = save_cache(self.selected_folder, self.results)
        self._populate_table(self.results)
        self._update_summary(
            tag=f"[재검수 갱신 {updated}+신규 {added}]",
            cache_msg=f"{'✅ ' if ok else '⚠️ '}{cache_msg}"
        )
        self.status_label.setText("재검수 완료. 행 더블클릭 → 이미지 뷰어")
        has_results = bool(self.results)
        self.btn_confirm_all.setEnabled(has_results)
        self.btn_confirm_sel.setEnabled(has_results)
        self.btn_unconfirm_all.setEnabled(has_results)
        any_sel = any(r.get("selected", False) for r in self.results)
        self.btn_review_sel.setEnabled(any_sel)
        self._refresh_dup_btn()
        self._refresh_partial_btn()

    # ── 시그널 핸들러 ─────────────────────────────────────────────────────
    def _on_progress(self, cur, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(cur)
        self.status_label.setText(f"추론 중... {cur} / {total}")

    def _on_finished(self, results, from_cache=False):
        self.results = results
        self.progress_bar.setVisible(False)
        self.btn_review.setEnabled(True)

        cache_msg = ""
        if not from_cache:
            ok, msg = save_cache(self.selected_folder, results)
            cache_msg = f"{'✅ ' if ok else '⚠️ '}{msg}"

        tag = "[캐시]" if from_cache else "[신규]"
        self._populate_table(results)
        self._update_summary(tag=tag, cache_msg=cache_msg)
        self.status_label.setText("완료. 행 더블클릭 → 이미지 뷰어")
        has_results = bool(self.results)
        self.btn_confirm_all.setEnabled(has_results)
        self.btn_confirm_sel.setEnabled(has_results)
        self.btn_unconfirm_all.setEnabled(has_results)
        any_sel = any(r.get("selected", False) for r in self.results)
        self.btn_review_sel.setEnabled(any_sel)
        self._refresh_dup_btn()
        self._refresh_partial_btn()

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_review.setEnabled(True)
        self.status_label.setText("오류 발생")
        QMessageBox.critical(self, "오류", msg)

    # ── 테이블 채우기 ─────────────────────────────────────────────────────
    def _populate_table(self, results):
        class_names = load_class_names_list(self.current_config_dir)
        self.table.blockSignals(True)   # itemChanged 오발 방지
        self.table.setRowCount(0)
        self.table.setRowCount(len(results))

        for row, r in enumerate(results):
            confirmed = r.get("confirmed", False)
            selected  = r.get("selected",  False)   # 선택 체크박스 상태
            # 중복 여부 (없으면 lazy 계산 후 캐싱)
            if "has_duplicate" not in r:
                r["has_duplicate"] = compute_has_duplicate(r["txt_path"])
            has_dup = r["has_duplicate"]

            row_bg = QColor("#d4edda") if confirmed else QColor("#ffffff")

            # 0 선택 체크박스
            item_sel = QTableWidgetItem()
            item_sel.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item_sel.setCheckState(Qt.Checked if selected else Qt.Unchecked)
            item_sel.setTextAlignment(Qt.AlignCenter)
            item_sel.setBackground(row_bg)
            self.table.setItem(row, 0, item_sel)

            # 1 파일명
            item_name = QTableWidgetItem(r["img_path"].name)
            item_name.setBackground(row_bg)
            self.table.setItem(row, 1, item_name)

            # 2 품질 점수
            item_score = QTableWidgetItem(f"{r['score']:.4f}")
            item_score.setData(Qt.UserRole, r["score"])
            item_score.setTextAlignment(Qt.AlignCenter)
            item_score.setBackground(row_bg)
            if not confirmed:
                if r["score"] < 0.5:
                    item_score.setForeground(QColor("#cc0000"))
                elif r["score"] < 0.8:
                    item_score.setForeground(QColor("#e07000"))
            self.table.setItem(row, 2, item_score)

            # 3 의심 여부
            issue_text = "⚠️ 의심" if r["is_issue"] else "✅ 정상"
            item_issue = QTableWidgetItem(issue_text)
            item_issue.setTextAlignment(Qt.AlignCenter)
            item_issue.setBackground(row_bg)
            if r["is_issue"] and not confirmed:
                item_issue.setForeground(QColor("#cc0000"))
            self.table.setItem(row, 3, item_issue)

            # 4 박스 수
            item_box = QTableWidgetItem(str(r["box_count"]))
            item_box.setTextAlignment(Qt.AlignCenter)
            item_box.setBackground(row_bg)
            self.table.setItem(row, 4, item_box)

            # 5 클래스 분포
            dist_text = get_class_dist(r["txt_path"], class_names)
            item_dist = QTableWidgetItem(dist_text)
            item_dist.setBackground(row_bg)
            self.table.setItem(row, 5, item_dist)

            # 6 중복 여부
            dup_text = "⚠️ 중복" if has_dup else "—"
            item_dup = QTableWidgetItem(dup_text)
            item_dup.setTextAlignment(Qt.AlignCenter)
            item_dup.setBackground(row_bg)
            if has_dup and not confirmed:
                item_dup.setForeground(QColor("#e07000"))
            self.table.setItem(row, 6, item_dup)

            # 7 확인 체크박스
            item_chk = QTableWidgetItem()
            item_chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item_chk.setCheckState(Qt.Checked if confirmed else Qt.Unchecked)
            item_chk.setTextAlignment(Qt.AlignCenter)
            item_chk.setBackground(row_bg)
            self.table.setItem(row, 7, item_chk)

            # 8 경로 (숨김)
            self.table.setItem(row, 8, QTableWidgetItem(str(r["img_path"])))

        self.table.blockSignals(False)

    # ── 체크박스 토글 처리 ────────────────────────────────────────────────
    def _on_item_changed(self, item):
        col = item.column()
        row = item.row()
        if row >= len(self.results):
            return

        if col == 0:    # 선택 체크박스
            self.results[row]["selected"] = (item.checkState() == Qt.Checked)
            any_sel = any(r.get("selected", False) for r in self.results)
            self.btn_review_sel.setEnabled(any_sel)
            return

        if col == 7:    # 확인 체크박스
            confirmed = (item.checkState() == Qt.Checked)
            self.results[row]["confirmed"] = confirmed
            row_bg = QColor("#d4edda") if confirmed else QColor("#ffffff")
            for c in range(8):
                cell = self.table.item(row, c)
                if cell:
                    cell.setBackground(row_bg)
            save_cache(self.selected_folder, self.results)
            self._update_summary()

    # ── 중복 제거 버튼 상태 재확인 ────────────────────────────────────────
    def _refresh_dup_btn(self):
        has_dup = any(r.get("has_duplicate", False) for r in self.results)
        self.btn_remove_dups.setEnabled(has_dup)

    # ── 중복 박스 일괄 제거 ────────────────────────────────────────────────
    def _remove_all_duplicates(self):
        """중복 박스가 있는 전체 txt 파일에서 아래 레이어(낮은 인덱스) 유지, 나머지 제거."""
        if not self.results:
            return
        dup_results = [r for r in self.results if r.get("has_duplicate", False)]
        if not dup_results:
            QMessageBox.information(self, "중복 박스 제거", "중복 박스가 없습니다.")
            return

        reply = QMessageBox.question(
            self, "중복 박스 일괄 제거",
            f"중복 박스가 있는 이미지 {len(dup_results)}개를 처리합니다.\n\n"
            f"• 각 중복 그룹에서 맨 아래 레이어(먼저 그린 박스, 낮은 인덱스)를 유지\n"
            f"• 나머지 박스를 txt 파일에서 삭제\n\n"
            f"이 작업은 txt 파일을 직접 수정합니다. 계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        total_removed = 0
        affected = 0
        for r in dup_results:
            n_removed = remove_duplicates_from_txt(r["txt_path"])
            if n_removed > 0:
                total_removed += n_removed
                affected += 1
            # 결과 갱신 (중복 재계산, 박스 수 재계산)
            r["has_duplicate"] = compute_has_duplicate(r["txt_path"])
            if r["txt_path"].exists():
                r["box_count"] = sum(
                    1 for line in r["txt_path"].read_text().strip().splitlines()
                    if len(line.split()) == 5
                )

        self._populate_table(self.results)
        save_cache(self.selected_folder, self.results)
        self._update_summary()
        self._refresh_dup_btn()
        QMessageBox.information(
            self, "완료",
            f"처리된 파일: {affected}개  |  제거된 박스: {total_removed}개"
        )

    # ── 재검수 버튼 상태 재확인 ───────────────────────────────────────────
    def _refresh_partial_btn(self):
        """수정된 파일 유무에 따라 재검수 버튼 활성/비활성."""
        if not self.selected_folder:
            return
        modified = self._get_modified_files()
        self.btn_partial.setEnabled(bool(modified))
        if modified:
            self.status_label.setText(
                f"수정된 파일 {len(modified)}개 감지 — 재검수 가능")

    # ── 요약 라벨 갱신 ────────────────────────────────────────────────────
    def _update_summary(self, tag="", cache_msg=""):
        if not self.results:
            return
        n = len(self.results)
        n_issues    = sum(r["is_issue"]                 for r in self.results)
        n_confirmed = sum(r.get("confirmed", False)     for r in self.results)
        n_dups      = sum(r.get("has_duplicate", False) for r in self.results)
        avg_score   = sum(r["score"]                    for r in self.results) / n
        tag_str     = f" {tag}" if tag else ""
        cache_str   = f"  |  {cache_msg}" if cache_msg else ""
        self.summary_label.setText(
            f"총 {n}개{tag_str}  |  의심: {n_issues}개  |  중복: {n_dups}개  |  "
            f"확인됨: {n_confirmed}개 ({n_confirmed/n*100:.0f}%)  |  "
            f"평균 점수: {avg_score:.4f}{cache_str}"
        )

    # ── 헤더 정렬 ─────────────────────────────────────────────────────────
    def _on_header_click(self, col):
        # col 0 = 선택(정렬 없음), col 5 = 클래스분포(정렬 없음), col 8 = 경로(숨김)
        if col not in (1, 2, 3, 4, 6, 7):
            return
        self._sort_asc = not self._sort_asc if self._sort_col == col else True
        self._sort_col = col
        key_map = {
            1: lambda r: r["img_path"].name,
            2: lambda r: r["score"],
            3: lambda r: r["is_issue"],
            4: lambda r: r["box_count"],
            6: lambda r: r.get("has_duplicate", False),
            7: lambda r: r.get("confirmed", False),
        }
        self.results = sorted(self.results, key=key_map[col], reverse=not self._sort_asc)
        self._populate_table(self.results)

    # ── 선택 항목 재검수 ──────────────────────────────────────────────────
    def _review_selected_items(self):
        """선택(col 0) 체크된 항목만 Cleanlab 재검수."""
        if not self.selected_folder:
            return
        sel_results = [r for r in self.results if r.get("selected", False)]
        if not sel_results:
            QMessageBox.information(self, "선택 검수", "선택(☑️)된 항목이 없습니다.")
            return
        file_filter = {r["img_path"].name for r in sel_results}
        self.btn_review.setEnabled(False)
        self.btn_partial.setEnabled(False)
        self.btn_review_sel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"선택 {len(file_filter)}개 검수 중...")
        self.worker = ReviewWorker(self.selected_folder, self._num_classes,
                                   file_filter=file_filter,
                                   auto_confirm=self.chk_auto_confirm.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(lambda r: self._on_sel_review_finished(r))
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_sel_review_finished(self, new_results):
        self.progress_bar.setVisible(False)
        self.btn_review.setEnabled(True)
        # 기존 results 에 새 결과 병합 (confirmed 수동 설정은 보존)
        new_map = {r["img_path"].name: r for r in new_results}
        for i, r in enumerate(self.results):
            name = r["img_path"].name
            if name in new_map:
                was_confirmed = r.get("confirmed", False)
                self.results[i] = new_map[name]
                if was_confirmed and not new_map[name].get("confirmed", False):
                    self.results[i]["confirmed"] = True
                # 선택 체크는 유지 (재검수 후에도 선택 상태 보존)
                self.results[i]["selected"] = r.get("selected", False)
        self._populate_table(self.results)
        ok, msg = save_cache(self.selected_folder, self.results)
        self._update_summary(tag="[선택 재검수]",
                             cache_msg=f"{'✅' if ok else '⚠️'} {msg}")
        self.status_label.setText("선택 검수 완료.")
        self._refresh_dup_btn()
        self._refresh_partial_btn()
        # 선택된 항목 있으면 버튼 재활성화
        any_sel = any(r.get("selected", False) for r in self.results)
        self.btn_review_sel.setEnabled(any_sel)

    # ── 더블클릭 → 뷰어 (체크박스 컬럼 제외) ────────────────────────────
    def _on_double_click(self, row, col):
        if col in (0, 7):       # 선택/확인 체크박스 컬럼은 뷰어 열지 않음
            return
        if not self.results or row >= len(self.results):
            return
        viewer = ImageReviewWindow(self.results, start_index=row, parent=self,
                                   config_dir=self.current_config_dir)
        viewer.confirmed_changed.connect(self._on_viewer_confirmed)
        viewer.destroyed.connect(self._refresh_partial_btn)  # 뷰어 닫히면 재검수 버튼 재확인
        viewer.show()

    def _on_viewer_confirmed(self, idx: int, confirmed: bool):
        """뷰어에서 확인 토글 시 테이블 해당 행 즉시 갱신."""
        # results 는 이미 공유 참조라 값은 반영됨 — 테이블 셀만 갱신
        self.table.blockSignals(True)
        chk = self.table.item(idx, 7)   # col 7 = 확인 체크박스
        if chk:
            chk.setCheckState(Qt.Checked if confirmed else Qt.Unchecked)
        row_bg = QColor("#d4edda") if confirmed else QColor("#ffffff")
        for c in range(8):
            cell = self.table.item(idx, c)
            if cell:
                cell.setBackground(row_bg)
        self.table.blockSignals(False)
        save_cache(self.selected_folder, self.results)
        self._update_summary()

    # ── 일괄 확인 ─────────────────────────────────────────────────────────
    def _confirm_all(self):
        """미확인 항목 전부 확인 처리."""
        if not self.results:
            return
        unconfirmed = [i for i, r in enumerate(self.results)
                       if not r.get("confirmed", False)]
        if not unconfirmed:
            QMessageBox.information(self, "일괄 확인", "모든 항목이 이미 확인됨 상태입니다.")
            return
        reply = QMessageBox.question(
            self, "일괄 확인",
            f"미확인 항목 {len(unconfirmed)}개를 모두 확인 처리할까요?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.table.blockSignals(True)
        for i in unconfirmed:
            self.results[i]["confirmed"] = True
            row_bg = QColor("#d4edda")
            for c in range(8):
                cell = self.table.item(i, c)
                if cell:
                    cell.setBackground(row_bg)
            chk = self.table.item(i, 7)
            if chk:
                chk.setCheckState(Qt.Checked)
        self.table.blockSignals(False)
        save_cache(self.selected_folder, self.results)
        self._update_summary()

    def _unconfirm_all(self):
        """확인된 항목 전부 해제."""
        if not self.results:
            return
        confirmed_list = [i for i, r in enumerate(self.results) if r.get("confirmed", False)]
        if not confirmed_list:
            QMessageBox.information(self, "전체 확인 해제", "확인된 항목이 없습니다.")
            return
        reply = QMessageBox.question(
            self, "전체 확인 해제",
            f"확인된 항목 {len(confirmed_list)}개를 모두 해제할까요?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.table.blockSignals(True)
        for i in confirmed_list:
            self.results[i]["confirmed"] = False
            row_bg = QColor("#ffffff")
            for c in range(8):
                cell = self.table.item(i, c)
                if cell:
                    cell.setBackground(row_bg)
            chk = self.table.item(i, 7)
            if chk:
                chk.setCheckState(Qt.Unchecked)
        self.table.blockSignals(False)
        save_cache(self.selected_folder, self.results)
        self._update_summary()

    def _confirm_selected(self):
        """테이블에서 선택된 행만 확인 처리."""
        if not self.results:
            return
        selected_rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not selected_rows:
            QMessageBox.information(self, "선택 확인", "선택된 행이 없습니다.\n(Shift/Ctrl 클릭으로 다중 선택)")
            return
        self.table.blockSignals(True)
        for row in selected_rows:
            if row < len(self.results):
                self.results[row]["confirmed"] = True
                row_bg = QColor("#d4edda")
                for c in range(8):
                    cell = self.table.item(row, c)
                    if cell:
                        cell.setBackground(row_bg)
                chk = self.table.item(row, 7)
                if chk:
                    chk.setCheckState(Qt.Checked)
        self.table.blockSignals(False)
        save_cache(self.selected_folder, self.results)
        self._update_summary()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
