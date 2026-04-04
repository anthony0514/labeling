"""
merge_labels_gui.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
두 폴더의 YOLO 라벨(.txt)을 이미지 위에 겹쳐 시각화하고
git merge conflict 방식으로 병합하는 GUI 도구.

- 중복 (IoU ≥ dup_thresh)  → 자동 제거 (B 측 제거, A 유지)
- 충돌 (iou_thresh ≤ IoU < dup_thresh) → 클릭으로 수동 선택
- 겹침 없음 → 양쪽 모두 유지
- 결과를 출력 폴더에 txt + 이미지 저장

조작:
  - 이미지 위 박스 클릭  → 포함/제외 토글
  - 우측 충돌 목록 클릭  → 해당 박스 하이라이트
  - ⚡ 자동 해결 (E)     → 충돌 시 A 우선 적용
  - ⊕ 모두 합치기 (W)   → 충돌/중복 쌍을 외접 박스 1개로 합치기
  - 💾 저장 (S)          → 현재 이미지만 / 전체 이미지
  - H                    → 단축키 도움말
"""

import sys
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np
import yaml

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QDoubleSpinBox, QGroupBox, QSplitter, QSizePolicy,
    QStatusBar, QMessageBox,
)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QImage, QColor, QBrush, QFont,
)

# ── 상수 ─────────────────────────────────────────────────────────────
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

COL_A          = QColor(0,   200,  80)    # 초록  - 폴더 A
COL_B          = QColor(30,  130, 255)    # 파랑  - 폴더 B
COL_CONF_A     = QColor(255, 140,   0)    # 주황  - A 충돌
COL_CONF_B     = QColor(220,  50,  50)    # 빨강  - B 충돌
COL_DUP        = QColor(150, 150, 150)    # 회색  - 중복(자동 제거)
COL_MERGE      = QColor(180,  80, 255)    # 보라  - 합쳐진 박스
COL_SEL        = QColor(255, 255,   0)    # 노랑  - 선택(하이라이트)

DARK = """
QMainWindow, QWidget  { background:#1e1e1e; color:#d4d4d4; }
QLabel                { color:#d4d4d4; }
QListWidget           { background:#252526; color:#ccc;
                        border:1px solid #444; }
QListWidget::item:selected { background:#094771; }
QListWidget::item:hover    { background:#2a2d2e; }
QGroupBox             { color:#bbb; border:1px solid #444;
                        margin-top:8px; padding-top:6px; }
QGroupBox::title      { subcontrol-origin:margin; left:8px; top:0; color:#999; }
QPushButton           { background:#3c3c3c; color:#d4d4d4;
                        border:1px solid #555; padding:4px 10px;
                        border-radius:3px; }
QPushButton:hover     { background:#4c4c4c; }
QPushButton:pressed   { background:#2a2a2a; }
QDoubleSpinBox        { background:#3c3c3c; color:#d4d4d4;
                        border:1px solid #555; padding:2px; }
QSplitter::handle     { background:#444; }
QStatusBar            { background:#007acc; color:#fff; font-weight:bold; }
"""


# ── 데이터 클래스 ────────────────────────────────────────────────────
@dataclass
class Box:
    cls:          int
    cx:           float
    cy:           float
    w:            float
    h:            float
    source:       str           # 'A' | 'B' | 'M'(merged)
    included:     bool  = True
    conflict_idx: int   = -1   # 충돌 상대 박스 global index (-1: 없음)
    is_duplicate: bool  = False
    gid:          int   = 0    # global index (merge_boxes 에서 할당)


# ── 유틸 ─────────────────────────────────────────────────────────────
def iou(a: Box, b: Box) -> float:
    ax1, ay1 = a.cx - a.w / 2, a.cy - a.h / 2
    ax2, ay2 = a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1 = b.cx - b.w / 2, b.cy - b.h / 2
    bx2, by2 = b.cx + b.w / 2, b.cy + b.h / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / (a.w * a.h + b.w * b.h - inter)


def load_txt(path: Path, source: str) -> List[Box]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            boxes.append(Box(
                cls=int(parts[0]),
                cx=float(parts[1]), cy=float(parts[2]),
                w=float(parts[3]),  h=float(parts[4]),
                source=source,
            ))
        except ValueError:
            pass
    return boxes


def merge(boxes_a: List[Box], boxes_b: List[Box],
          iou_thr: float, dup_thr: float) -> List[Box]:
    all_boxes: List[Box] = []
    for i, b in enumerate(boxes_a):
        b.gid = i
        all_boxes.append(b)
    for i, b in enumerate(boxes_b):
        b.gid = len(boxes_a) + i
        all_boxes.append(b)

    seen = set()
    for i, a in enumerate(all_boxes):
        for j, b in enumerate(all_boxes):
            if j <= i:
                continue
            if a.source == b.source:
                continue
            pair = (i, j)
            if pair in seen:
                continue
            score = iou(a, b)
            if score >= dup_thr:
                # 중복: B 자동 제거
                b.is_duplicate = True
                b.included     = False
                b.conflict_idx = a.gid
                a.conflict_idx = b.gid
                seen.add(pair)
            elif score >= iou_thr:
                # 충돌: 수동 선택
                a.conflict_idx = b.gid
                b.conflict_idx = a.gid
                seen.add(pair)
    return all_boxes


def find_image(folder: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


def load_img(path: Path) -> Optional[np.ndarray]:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def union_box(a: Box, b: Box, next_gid: int) -> Box:
    """두 박스의 외접 bounding box를 반환. 클래스는 A 기준."""
    x1 = min(a.cx - a.w / 2, b.cx - b.w / 2)
    y1 = min(a.cy - a.h / 2, b.cy - b.h / 2)
    x2 = max(a.cx + a.w / 2, b.cx + b.w / 2)
    y2 = max(a.cy + a.h / 2, b.cy + b.h / 2)
    return Box(
        cls=a.cls,
        cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
        w=x2 - x1,        h=y2 - y1,
        source='M', included=True, gid=next_gid,
    )


# ── ImageCanvas ──────────────────────────────────────────────────────
class ImageCanvas(QLabel):
    """이미지 + YOLO 박스 표시. 클릭으로 박스 포함/제외 토글."""

    sig_toggled = pyqtSignal(int)   # gid
    sig_hovered = pyqtSignal(int)   # gid (-1 = none)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#1a1a1a;")
        self.setMouseTracking(True)

        self._img:         Optional[np.ndarray] = None
        self._boxes:       List[Box]            = []
        self._names:       List[str]            = []
        self._hover_gid:   int                  = -1
        self._sel_gid:     int                  = -1
        self._hidden:      set                  = set()   # 'A', 'B' 숨김

        # 렌더 캐시
        self._disp_rect = QRectF()

    # ── 공개 API ──────────────────────────────────────────────────────
    def load(self, img: Optional[np.ndarray],
             boxes: List[Box], names: List[str]):
        self._img      = img
        self._boxes    = boxes
        self._names    = names
        self._hover_gid = -1
        self._sel_gid   = -1
        self._paint()

    def refresh_boxes(self, boxes: List[Box]):
        self._boxes = boxes
        self._paint()

    def set_hidden(self, sources: set):
        """숨길 소스 집합 설정. 예: {'A'} or {'B'} or set()"""
        self._hidden = sources
        self._paint()

    def set_highlight(self, gid: int):
        self._sel_gid = gid
        self._paint()

    # ── 렌더링 ───────────────────────────────────────────────────────
    def _paint(self):
        if self._img is None:
            self.clear()
            self.setText("이미지를 선택하세요")
            return

        cw, ch = self.width(), self.height()
        ih, iw = self._img.shape[:2]
        scale = min(cw / iw, ch / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self._disp_rect = QRectF(ox, oy, dw, dh)

        rgb     = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
        qimg    = QImage(resized.data, dw, dh, dw * 3,
                         QImage.Format_RGB888).copy()

        canvas = QPixmap(cw, ch)
        canvas.fill(QColor('#1a1a1a'))
        p = QPainter(canvas)
        p.drawPixmap(ox, oy, QPixmap.fromImage(qimg))

        for box in self._boxes:
            if box.source in self._hidden:
                continue
            self._draw_box(p, box, dw, dh, ox, oy)
        p.end()
        self.setPixmap(canvas)

    def _box_color(self, box: Box) -> tuple[QColor, int]:
        """(색상, 알파) 반환."""
        if box.source == 'M':
            return (COL_MERGE, 240)
        if not box.included:
            return (COL_DUP, 80)
        if box.conflict_idx >= 0 and not box.is_duplicate:
            return (COL_CONF_A if box.source == 'A' else COL_CONF_B, 230)
        return (COL_A if box.source == 'A' else COL_B, 230)

    def _draw_box(self, p: QPainter, box: Box,
                  dw: int, dh: int, ox: int, oy: int):
        color, alpha = self._box_color(box)
        is_sel = box.gid in (self._sel_gid, self._hover_gid)

        x1 = int((box.cx - box.w / 2) * dw) + ox
        y1 = int((box.cy - box.h / 2) * dh) + oy
        x2 = int((box.cx + box.w / 2) * dw) + ox
        y2 = int((box.cy + box.h / 2) * dh) + oy
        bw, bh = x2 - x1, y2 - y1

        pen_col = COL_SEL if is_sel else QColor(color)
        pen_col.setAlpha(255 if is_sel else alpha)
        p.setPen(QPen(pen_col, 3 if is_sel else 2))

        fill = QColor(color)
        fill.setAlpha(35 if box.included else 20)
        p.setBrush(QBrush(fill))
        p.drawRect(x1, y1, bw, bh)
        p.setBrush(Qt.NoBrush)

        # 라벨
        name = (self._names[box.cls]
                if box.cls < len(self._names) else str(box.cls))
        src_tag = {'A': 'A', 'B': 'B', 'M': '⊕'}.get(box.source, box.source)
        tag = f"[{src_tag}] {box.cls}:{name}"
        if not box.included:
            tag += "  ✗"

        fnt = QFont("Arial", 8, QFont.Bold)
        p.setFont(fnt)
        fm  = p.fontMetrics()
        tw, th = fm.horizontalAdvance(tag), fm.height()

        lx = max(0, x1)
        ly = max(th, y1)
        bg = QColor(color)
        bg.setAlpha(190 if box.included else 90)
        p.fillRect(lx, ly - th + 2, tw + 6, th, bg)
        p.setPen(QColor(Qt.white))
        p.drawText(lx + 3, ly, tag)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._paint()

    # ── 히트 테스트 ──────────────────────────────────────────────────
    def _hit(self, x: int, y: int) -> int:
        if not self._disp_rect.contains(x, y):
            return -1
        dw, dh = self._disp_rect.width(), self._disp_rect.height()
        ox, oy = self._disp_rect.x(), self._disp_rect.y()
        for box in reversed(self._boxes):
            if box.source in self._hidden:
                continue
            rx1 = (box.cx - box.w / 2) * dw + ox
            ry1 = (box.cy - box.h / 2) * dh + oy
            rx2 = (box.cx + box.w / 2) * dw + ox
            ry2 = (box.cy + box.h / 2) * dh + oy
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                return box.gid
        return -1

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            gid = self._hit(ev.x(), ev.y())
            if gid >= 0:
                self.sig_toggled.emit(gid)

    def mouseMoveEvent(self, ev):
        gid = self._hit(ev.x(), ev.y())
        if gid != self._hover_gid:
            self._hover_gid = gid
            self.sig_hovered.emit(gid)
            self._paint()


# ── ConflictPanel ─────────────────────────────────────────────────────
class ConflictPanel(QWidget):
    """우측 패널: A 박스 / B 박스 / 충돌·중복 3섹션."""
    sig_jump = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(240)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)

        self._list_a    = self._make_section(lay, "A 박스",      "#00C851")
        self._list_b    = self._make_section(lay, "B 박스",      "#1E82FF")
        self._list_conf = self._make_section(lay, "충돌 · 중복", "#FF8800")

        hint = QLabel("클릭 → 이미지에서 하이라이트  /  이미지 박스 클릭 → 토글")
        hint.setStyleSheet("color:#666; font-size:9px; padding:2px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

    def _make_section(self, parent_lay, title: str, color: str) -> QListWidget:
        lbl = QLabel(title)
        lbl.setStyleSheet(
            f"color:{color}; font-weight:bold; font-size:11px;"
            f"padding:2px 6px; background:#252526;"
            f"border-left:3px solid {color};")
        parent_lay.addWidget(lbl)
        lst = QListWidget()
        lst.itemClicked.connect(self._on_clicked)
        parent_lay.addWidget(lst, 1)
        return lst

    def _on_clicked(self, item: QListWidgetItem):
        gid = item.data(Qt.UserRole)
        if gid is not None:
            self.sig_jump.emit(gid)

    def update(self, boxes: List[Box], names: List[str]):
        self._list_a.clear()
        self._list_b.clear()
        self._list_conf.clear()
        seen: set[tuple] = set()

        def _name(b: Box) -> str:
            n = names[b.cls] if b.cls < len(names) else "?"
            return f"{b.cls}:{n}"

        def _add(lst: QListWidget, label: str, color: str, gid: int):
            item = QListWidgetItem(label)
            item.setForeground(QColor(color))
            item.setData(Qt.UserRole, gid)
            lst.addItem(item)

        for b in boxes:
            n = _name(b)

            # ── 합쳐진 박스 → 충돌 섹션에
            if b.source == 'M':
                _add(self._list_conf, f"⊕ {n}", "#B450FF", b.gid)
                continue

            # ── A / B 섹션
            lst = self._list_a if b.source == 'A' else self._list_b
            if not b.included:
                _add(lst, f"✗ {n}", "#555555", b.gid)
            elif b.conflict_idx >= 0 and not b.is_duplicate:
                col = "#FF8800" if b.source == 'A' else "#DC3232"
                _add(lst, f"⚠ {n}", col, b.gid)
            else:
                col = "#00C851" if b.source == 'A' else "#1E82FF"
                _add(lst, f"✔ {n}", col, b.gid)

            # ── 충돌·중복 섹션
            if b.is_duplicate and not b.included:
                _add(self._list_conf, f"🔁 [{b.source}] {n}", "#969696", b.gid)
            elif b.conflict_idx >= 0 and not b.is_duplicate:
                pair = tuple(sorted([b.gid, b.conflict_idx]))
                if pair not in seen:
                    seen.add(pair)
                    other = next((x for x in boxes if x.gid == b.conflict_idx), None)
                    score = iou(b, other) if other else 0.0
                    col = "#FF8800" if b.source == 'A' else "#DC3232"
                    _add(self._list_conf,
                         f"⚠ [{b.source}] {n}  {score:.2f}",
                         col, b.gid)


# ── MergeTool (메인 윈도우) ──────────────────────────────────────────
class MergeTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralD LabelMerge v1.1")
        self.resize(1360, 820)
        self.setStyleSheet(DARK)

        self._folder_a:   Optional[Path] = None
        self._folder_b:   Optional[Path] = None
        self._folder_out: Optional[Path] = None
        self._folder_img: Optional[Path] = None   # 별도 이미지 폴더 (선택)
        self._stems:  List[str] = []
        self._cur_stem: Optional[str] = None
        self._boxes:  List[Box] = []
        self._names:  List[str] = []
        self._show_a: bool = True
        self._show_b: bool = True

        self._build_ui()
        self._try_load_names()

    # ── UI 구성 ───────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setSpacing(4)

        # 상단 툴바
        vlay.addLayout(self._make_toolbar())

        # 덮어씌우기 행
        vlay.addLayout(self._make_copy_row())

        # 메인 스플리터
        sp = QSplitter(Qt.Horizontal)

        # 왼쪽: 이미지 목록
        left = QWidget()
        left.setFixedWidth(190)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self._mk_label("이미지 목록", bold=True))
        self._img_list = QListWidget()
        self._img_list.currentItemChanged.connect(self._on_img_selected)
        ll.addWidget(self._img_list)
        sp.addWidget(left)

        # 중앙: 캔버스 + 내비
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(2)
        self._canvas = ImageCanvas()
        self._canvas.sig_toggled.connect(self._on_toggled)
        self._canvas.sig_hovered.connect(self._on_hovered)
        cl.addWidget(self._canvas)
        cl.addLayout(self._make_nav())
        sp.addWidget(center)

        # 오른쪽: 충돌 패널
        self._cpanel = ConflictPanel()
        self._cpanel.sig_jump.connect(
            lambda gid: self._canvas.set_highlight(gid))
        sp.addWidget(self._cpanel)

        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 0)
        vlay.addWidget(sp, 1)

        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._sb.showMessage("폴더 A와 폴더 B를 선택하세요.")

    def _make_toolbar(self):
        row = QHBoxLayout()
        row.setSpacing(6)

        # 폴더 4개
        for title, color, which in [
            ("A 폴더",    "#00C851", 'A'),
            ("B 폴더",    "#1E82FF", 'B'),
            ("이미지 폴더", "#aaaaaa", 'img'),
            ("출력 폴더",  "#FF8800", 'out'),
        ]:
            grp = QGroupBox(title)
            grp.setStyleSheet(
                f"QGroupBox{{color:{color}; border:1px solid {color}55;"
                f"margin-top:8px; padding-top:4px;}}")
            gl = QHBoxLayout(grp)
            gl.setSpacing(4)
            lbl = QLabel("—")
            lbl.setStyleSheet("color:#999; font-size:11px;")
            lbl.setMinimumWidth(100)
            gl.addWidget(lbl)
            btn = QPushButton("찾기")
            btn.setFixedWidth(46)
            gl.addWidget(btn)

            # 클로저 캡처
            def _cb(w=which, l=lbl):
                return lambda: self._browse(w, l)
            btn.clicked.connect(_cb())
            row.addWidget(grp)

        # 임계값
        grp2 = QGroupBox("IoU 임계값")
        grp2.setStyleSheet(
            "QGroupBox{color:#ccc; border:1px solid #555;"
            "margin-top:8px; padding-top:4px;}")
        tl = QVBoxLayout(grp2)
        tl.setSpacing(2)
        for lbl_txt, attr, val, lo, hi in [
            ("충돌 ≥", "_sp_conf", 0.30, 0.05, 0.90),
            ("중복 ≥", "_sp_dup",  0.95, 0.50, 1.00),
        ]:
            rl = QHBoxLayout()
            rl.addWidget(QLabel(lbl_txt))
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(0.05)
            sp.setValue(val)
            sp.valueChanged.connect(self._on_thresh_changed)
            rl.addWidget(sp)
            tl.addLayout(rl)
            setattr(self, attr, sp)
        row.addWidget(grp2)

        # 전체 합치기 + 저장
        for label, color, cb in [
            ("⊕ 전체\n합치기", "#5a2d8a", self._union_merge_all),
            ("💾 전체\n저장",   "#0e639c", self._save_all),
        ]:
            btn = QPushButton(label)
            btn.setFixedSize(76, 60)
            btn.setStyleSheet(
                f"background:{color}; color:#fff; font-weight:bold; border-radius:4px;")
            btn.clicked.connect(cb)
            row.addWidget(btn)

        return row

    def _make_copy_row(self):
        """A→B / B→A 덮어씌우기 버튼 행."""
        row = QHBoxLayout()
        row.setSpacing(4)

        lbl = QLabel("덮어씌우기  (한쪽에만 있는 박스 → 반대편에 추가):")
        lbl.setStyleSheet("color:#888; font-size:11px; padding:0 4px;")
        row.addWidget(lbl)

        for text, color, src, all_ in [
            ("A → B  (이 이미지)",  "#1a6e30", 'A', False),
            ("A → B  (전체)",       "#0e4020", 'A', True),
            ("B → A  (이 이미지)",  "#0e3d7a", 'B', False),
            ("B → A  (전체)",       "#082550", 'B', True),
        ]:
            btn = QPushButton(text)
            btn.setFixedHeight(26)
            btn.setStyleSheet(
                f"QPushButton{{background:{color}; color:#eee; border-radius:3px;"
                f"border:1px solid {color}cc; padding:2px 8px; font-size:12px;}}"
                f"QPushButton:hover{{background:{color}cc;}}"
                f"QPushButton:disabled{{background:#2a2a2a; color:#555;}}")
            # 클로저 캡처
            def _cb(s=src, a=all_):
                return lambda: (self._overwrite_all(s) if a else self._overwrite_cur(s))
            btn.clicked.connect(_cb())
            row.addWidget(btn)

        row.addStretch()
        return row

    def _make_nav(self):
        row = QHBoxLayout()
        for text, cb in [
            ("◀ 이전  [A / ←]", lambda: self._nav(-1)),
            ("다음 ▶  [D / →]", lambda: self._nav(1)),
        ]:
            b = QPushButton(text)
            b.clicked.connect(cb)
            row.addWidget(b)

        row.addSpacing(12)

        # A 토글
        self._btn_toggle_a = QPushButton("A 표시  (0)")
        self._btn_toggle_a.setCheckable(True)
        self._btn_toggle_a.setChecked(True)
        self._btn_toggle_a.setFixedHeight(26)
        self._btn_toggle_a.setStyleSheet(
            "QPushButton{background:#1a6e30; color:#eee; border-radius:3px;"
            "border:1px solid #00C851; padding:2px 8px; font-size:12px;}"
            "QPushButton:hover{background:#24a044;}"
            "QPushButton:!checked{background:#2a2a2a; color:#555; border-color:#444;}")
        self._btn_toggle_a.clicked.connect(self._toggle_a)
        row.addWidget(self._btn_toggle_a)

        # B 토글
        self._btn_toggle_b = QPushButton("B 표시  (0)")
        self._btn_toggle_b.setCheckable(True)
        self._btn_toggle_b.setChecked(True)
        self._btn_toggle_b.setFixedHeight(26)
        self._btn_toggle_b.setStyleSheet(
            "QPushButton{background:#0e3d7a; color:#eee; border-radius:3px;"
            "border:1px solid #1E82FF; padding:2px 8px; font-size:12px;}"
            "QPushButton:hover{background:#1558b0;}"
            "QPushButton:!checked{background:#2a2a2a; color:#555; border-color:#444;}")
        self._btn_toggle_b.clicked.connect(self._toggle_b)
        row.addWidget(self._btn_toggle_b)

        row.addStretch()

        for text, cb in [
            ("❓ 도움말  [H]",             self._show_help),
            ("⊕ 합치기+저장  [Space]",     self._merge_and_save),
            ("⊕ 합치기만  [W]",            self._union_merge),
            ("💾 이 이미지 저장  [S]",      self._save_cur),
        ]:
            b = QPushButton(text)
            b.clicked.connect(cb)
            row.addWidget(b)
        return row

    @staticmethod
    def _mk_label(text: str, bold=False) -> QLabel:
        l = QLabel(text)
        s = "color:#ccc; padding:4px;"
        if bold:
            s += "font-weight:bold;"
        l.setStyleSheet(s)
        return l

    # ── 폴더 선택 ─────────────────────────────────────────────────────
    def _browse(self, which: str, lbl: QLabel):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if not path:
            return
        p = Path(path)
        lbl.setText(p.name)
        lbl.setToolTip(str(p))
        if which == 'A':
            self._folder_a = p
        elif which == 'B':
            self._folder_b = p
        elif which == 'img':
            self._folder_img = p
        else:
            self._folder_out = p
        if self._folder_a and self._folder_b:
            self._build_list()

    # ── 이미지 목록 구성 ──────────────────────────────────────────────
    def _build_list(self):
        sa = {p.stem for p in self._folder_a.iterdir()
              if p.suffix.lower() in IMG_EXTS or p.suffix == '.txt'}
        sb = {p.stem for p in self._folder_b.iterdir()
              if p.suffix.lower() in IMG_EXTS or p.suffix == '.txt'}
        self._stems = sorted(sa | sb)

        self._img_list.blockSignals(True)
        self._img_list.clear()
        for stem in self._stems:
            icon = "🟡" if (stem in sa and stem in sb) \
                else ("🟢" if stem in sa else "🔵")
            item = QListWidgetItem(f"{icon} {stem}")
            item.setData(Qt.UserRole, stem)
            self._img_list.addItem(item)
        self._img_list.blockSignals(False)

        self._sb.showMessage(
            f"이미지 {len(self._stems)}개  "
            f"(A:{len(sa)}  B:{len(sb)}  공통:{len(sa & sb)})")
        if self._stems:
            self._img_list.setCurrentRow(0)

    # ── 이미지 로드 ───────────────────────────────────────────────────
    def _on_img_selected(self, cur, _prev):
        if cur is None:
            return
        self._load(cur.data(Qt.UserRole))

    def _load(self, stem: str):
        self._cur_stem = stem

        # 이미지 — 이미지 폴더 우선, 없으면 A/B 폴더에서 탐색
        img = None
        for folder in (self._folder_img, self._folder_a, self._folder_b):
            if folder:
                p = find_image(folder, stem)
                if p:
                    img = load_img(p)
                    break

        # 라벨
        boxes_a = (load_txt(self._folder_a / (stem + '.txt'), 'A')
                   if self._folder_a else [])
        boxes_b = (load_txt(self._folder_b / (stem + '.txt'), 'B')
                   if self._folder_b else [])
        self._boxes = merge(boxes_a, boxes_b,
                            self._sp_conf.value(),
                            self._sp_dup.value())

        self._canvas.load(img, self._boxes, self._names)
        self._apply_visibility()
        self._cpanel.update(self._boxes, self._names)
        self._update_toggle_counts()

        n_conf = len({tuple(sorted([b.gid, b.conflict_idx]))
                      for b in self._boxes
                      if b.conflict_idx >= 0 and not b.is_duplicate})
        n_dup  = sum(1 for b in self._boxes if b.is_duplicate)
        self._sb.showMessage(
            f"{stem}  │  A:{len(boxes_a)}개  B:{len(boxes_b)}개  "
            f"충돌:{n_conf}쌍  중복(자동제거):{n_dup}개")

    # ── A / B 토글 ───────────────────────────────────────────────────
    def _toggle_a(self):
        self._show_a = self._btn_toggle_a.isChecked()
        self._apply_visibility()

    def _toggle_b(self):
        self._show_b = self._btn_toggle_b.isChecked()
        self._apply_visibility()

    def _apply_visibility(self):
        hidden = set()
        if not self._show_a:
            hidden.add('A')
        if not self._show_b:
            hidden.add('B')
        self._canvas.set_hidden(hidden)

    def _update_toggle_counts(self):
        n_a = sum(1 for b in self._boxes if b.source == 'A')
        n_b = sum(1 for b in self._boxes if b.source == 'B')
        self._btn_toggle_a.setText(f"A 표시  ({n_a})")
        self._btn_toggle_b.setText(f"B 표시  ({n_b})")

    # ── 박스 인터랙션 ─────────────────────────────────────────────────
    def _on_toggled(self, gid: int):
        box = next((b for b in self._boxes if b.gid == gid), None)
        if box:
            box.included = not box.included
            self._canvas.refresh_boxes(self._boxes)
            self._cpanel.update(self._boxes, self._names)

    def _on_hovered(self, gid: int):
        if gid < 0:
            return
        box = next((b for b in self._boxes if b.gid == gid), None)
        if box:
            n = (self._names[box.cls]
                 if box.cls < len(self._names) else str(box.cls))
            self._sb.showMessage(
                f"[{box.source}] cls={n}  "
                f"cx={box.cx:.4f}  cy={box.cy:.4f}  "
                f"w={box.w:.4f}  h={box.h:.4f}")

    # ── 네비게이션 ───────────────────────────────────────────────────
    def _nav(self, delta: int):
        row = self._img_list.currentRow() + delta
        row = max(0, min(row, self._img_list.count() - 1))
        self._img_list.setCurrentRow(row)

    # ── 합치기 + 저장 (Space) ────────────────────────────────────────
    def _merge_and_save(self):
        """모두 합치기 후 즉시 저장."""
        self._union_merge()
        self._save_cur()

    # ── 모두 합치기 ──────────────────────────────────────────────────
    def _union_merge(self):
        """충돌/중복 쌍을 외접 bounding box 1개로 합치기.
        여러 번 눌러도 결과가 동일하도록 (멱등) 이전 합치기 결과를 먼저 초기화."""
        # ① 이전 합치기로 생긴 'M' 박스 제거
        self._boxes = [b for b in self._boxes if b.source != 'M']
        # ② 충돌/중복 박스 included 상태 초기화 (이전 합치기에서 False 됐을 수 있음)
        for b in self._boxes:
            if b.conflict_idx >= 0:
                b.included = True

        # ③ 충돌/중복 쌍마다 union 박스 생성
        next_gid = max((b.gid for b in self._boxes), default=-1) + 1
        seen: set[tuple] = set()
        new_boxes: List[Box] = []

        for b in self._boxes:
            if b.conflict_idx < 0:
                continue   # 충돌 없는 박스는 건드리지 않음
            pair = tuple(sorted([b.gid, b.conflict_idx]))
            if pair in seen:
                b.included = False
                continue
            seen.add(pair)
            other = next((x for x in self._boxes if x.gid == b.conflict_idx), None)
            if other:
                a_box = b if b.source == 'A' else other
                b_box = other if b.source == 'A' else b
                new_boxes.append(union_box(a_box, b_box, next_gid))
                next_gid += 1
                b.included     = False
                other.included = False

        self._boxes.extend(new_boxes)
        self._canvas.refresh_boxes(self._boxes)
        self._cpanel.update(self._boxes, self._names)
        self._sb.showMessage(f"모두 합치기 완료 — 합쳐진 박스 {len(new_boxes)}개 생성")

    # ── 저장 ──────────────────────────────────────────────────────────
    def _lines(self) -> List[str]:
        return [f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"
                for b in self._boxes if b.included]

    def _write(self, stem: str, lines: List[str]):
        out = self._folder_out / (stem + '.txt')
        out.write_text('\n'.join(lines), encoding='utf-8')
        # 이미지 복사
        for folder in (self._folder_a, self._folder_b):
            if folder:
                p = find_image(folder, stem)
                if p:
                    shutil.copy2(p, self._folder_out / p.name)
                    break

    def _save_cur(self):
        if not self._cur_stem:
            return
        if not self._folder_out:
            QMessageBox.warning(self, "경고", "출력 폴더를 먼저 선택하세요.")
            return
        self._folder_out.mkdir(parents=True, exist_ok=True)
        lines = self._lines()
        self._write(self._cur_stem, lines)
        self._sb.showMessage(
            f"저장 완료: {self._cur_stem}.txt  ({len(lines)}개 박스)")

    def _save_all(self):
        if not self._folder_out:
            QMessageBox.warning(self, "경고", "출력 폴더를 먼저 선택하세요.")
            return
        if not self._stems:
            return
        self._folder_out.mkdir(parents=True, exist_ok=True)

        prev_stem = self._cur_stem
        for stem in self._stems:
            self._load(stem)
            self._write(stem, self._lines())

        # 원래 이미지로 복귀
        if prev_stem:
            self._load(prev_stem)

        QMessageBox.information(
            self, "완료",
            f"{len(self._stems)}개 이미지 저장 완료\n→ {self._folder_out}")

    # ── A↔B 덮어씌우기 ──────────────────────────────────────────────
    def _apply_unique(self, src: str, tgt_folder: Path, stem: str) -> int:
        """src에만 있는 박스(충돌/중복 없음)를 tgt_folder/stem.txt 에 추가.
        반환: 추가된 박스 수."""
        unique = [b for b in self._boxes
                  if b.source == src and b.conflict_idx < 0]
        if not unique:
            return 0
        txt_path = tgt_folder / (stem + '.txt')
        existing = []
        if txt_path.exists():
            existing = [l.strip()
                        for l in txt_path.read_text(encoding='utf-8').splitlines()
                        if l.strip()]
        new_lines = [f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"
                     for b in unique]
        txt_path.write_text('\n'.join(existing + new_lines), encoding='utf-8')
        return len(new_lines)

    def _overwrite_cur(self, src: str):
        """현재 이미지: src 에만 있는 박스를 반대편 폴더 txt 에 추가."""
        if not self._cur_stem:
            return
        tgt = 'B' if src == 'A' else 'A'
        tgt_folder = self._folder_b if src == 'A' else self._folder_a
        if not tgt_folder:
            QMessageBox.warning(self, "경고", f"폴더 {tgt} 가 없습니다.")
            return
        n = self._apply_unique(src, tgt_folder, self._cur_stem)
        self._load(self._cur_stem)   # txt 변경 반영
        self._sb.showMessage(
            f"{src} → {tgt} 완료: {n}개 박스 추가  ({self._cur_stem})")

    def _overwrite_all(self, src: str):
        """전체 이미지: src 에만 있는 박스를 반대편 폴더 txt 에 추가."""
        tgt = 'B' if src == 'A' else 'A'
        tgt_folder = self._folder_b if src == 'A' else self._folder_a
        if not tgt_folder:
            QMessageBox.warning(self, "경고", f"폴더 {tgt} 가 없습니다.")
            return
        if not self._stems:
            return
        prev_stem = self._cur_stem
        total = 0
        for stem in self._stems:
            self._load(stem)
            total += self._apply_unique(src, tgt_folder, stem)
        if prev_stem:
            self._load(prev_stem)
        QMessageBox.information(
            self, "완료",
            f"전체 {len(self._stems)}개 이미지\n"
            f"{src} → {tgt}: 총 {total}개 박스 추가\n→ {tgt_folder}")

    # ── 전체 합치기 저장 ─────────────────────────────────────────────
    def _union_merge_all(self):
        """전체 이미지에 모두합치기 적용 후 저장."""
        if not self._folder_out:
            QMessageBox.warning(self, "경고", "출력 폴더를 먼저 선택하세요.")
            return
        if not self._stems:
            return
        self._folder_out.mkdir(parents=True, exist_ok=True)
        prev_stem = self._cur_stem
        for stem in self._stems:
            self._load(stem)
            self._union_merge()
            self._write(stem, self._lines())
        if prev_stem:
            self._load(prev_stem)
        QMessageBox.information(
            self, "완료",
            f"{len(self._stems)}개 이미지 합치기+저장 완료\n→ {self._folder_out}")

    # ── 임계값 변경 ──────────────────────────────────────────────────
    def _on_thresh_changed(self):
        if self._cur_stem:
            self._load(self._cur_stem)

    # ── 키보드 단축키 ────────────────────────────────────────────────
    def keyPressEvent(self, ev):
        key = ev.key()
        if key in (Qt.Key_Left,  Qt.Key_A):
            self._nav(-1)
        elif key in (Qt.Key_Right, Qt.Key_D):
            self._nav(1)
        elif key == Qt.Key_Space:
            self._merge_and_save()
        elif key == Qt.Key_S:
            self._save_cur()
        elif key == Qt.Key_W:
            self._union_merge()
        elif key == Qt.Key_H:
            self._show_help()
        elif key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(ev)

    # ── 도움말 ──────────────────────────────────────────────────────
    def _show_help(self):
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("단축키 도움말")
        msg.setStyleSheet("QLabel{color:#222;} QMessageBox{background:#fff;}")
        msg.setText("""
<h3>⌨️ 키보드 단축키</h3>
<table cellpadding="4">
<tr><td><b>A / ←</b></td><td>이전 이미지</td></tr>
<tr><td><b>D / →</b></td><td>다음 이미지</td></tr>
<tr><td><b>Space</b></td><td>합치기 + 저장 (한 번에)</td></tr>
<tr><td><b>W</b></td><td>합치기만 — 충돌/중복 쌍을 외접 박스 1개로</td></tr>
<tr><td><b>S</b></td><td>현재 이미지 저장</td></tr>
<tr><td><b>H</b></td><td>이 도움말 보기</td></tr>
<tr><td><b>Q / Esc</b></td><td>창 닫기</td></tr>
</table>
<h3>🖱️ 마우스</h3>
<table cellpadding="4">
<tr><td><b>이미지 박스 클릭</b></td><td>포함 / 제외 토글</td></tr>
<tr><td><b>우측 목록 클릭</b></td><td>해당 박스 하이라이트</td></tr>
</table>
<h3>🎨 색상 범례</h3>
<table cellpadding="4">
<tr><td style="color:#00C851">■</td><td>폴더 A 박스</td></tr>
<tr><td style="color:#1E82FF">■</td><td>폴더 B 박스</td></tr>
<tr><td style="color:#FF8800">■</td><td>A 충돌 박스</td></tr>
<tr><td style="color:#DC3232">■</td><td>B 충돌 박스</td></tr>
<tr><td style="color:#969696">■</td><td>중복 (자동 제거)</td></tr>
<tr><td style="color:#B450FF">■</td><td>⊕ 합쳐진 박스</td></tr>
</table>
""")
        msg.exec_()

    # ── 클래스명 자동 로드 ────────────────────────────────────────────
    def _try_load_names(self):
        here = Path(__file__).resolve().parent
        candidates = [
            here / 'auto' / 'dataset.yaml',
            here / 'dataset.yaml',
        ]
        for yp in candidates:
            if not yp.exists():
                continue
            try:
                with open(yp, encoding='utf-8') as f:
                    d = yaml.safe_load(f)
                names = d.get('names', {})
                if isinstance(names, dict):
                    mx = max(names.keys()) + 1
                    self._names = [names.get(i, str(i)) for i in range(mx)]
                elif isinstance(names, list):
                    self._names = names
                return
            except Exception:
                pass


# ── 진입점 ───────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MergeTool()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
