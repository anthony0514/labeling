#!/usr/bin/env python3
"""compare_labels_gui.py — 두 라벨 폴더 박스 비교 GUI

A폴더·B폴더의 동일 이미지를 불러와 박스 차이를 시각화합니다.
  - 빨강 (A만) : A 폴더에만 있는 박스
  - 파랑 (B만) : B 폴더에만 있는 박스
  - 회색 (공통) : 두 폴더 모두 있는 박스 (IoU ≥ 0.45 + 동일 클래스)

키보드: ↑↓ 이미지 이동, H 하이라이트 전용 모드(차이만 표시)
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QColor, QFont, QFontMetrics, QImage, QPainter, QPen,
    QPixmap, QBrush, QKeySequence
)
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget, QScrollArea,
    QSlider, QShortcut, QFrame, QTableWidget, QTableWidgetItem, QHeaderView
)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SCRIPT_DIR = Path(__file__).resolve().parent

# ── 색상 ──────────────────────────────────────────────────────────────────────
COL_A_BORDER  = QColor(255,  70,  70, 220)
COL_B_BORDER  = QColor( 70, 160, 255, 220)
COL_M_BORDER  = QColor(160, 160, 160, 120)
COL_A_FILL    = QColor(255,  70,  70,  30)
COL_B_FILL    = QColor( 70, 160, 255,  30)
COL_M_FILL    = QColor(180, 180, 180,  15)


# ── 데이터 ───────────────────────────────────────────────────────────────────

@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def rect_px(self, W: int, H: int) -> QRectF:
        x = (self.cx - self.w / 2) * W
        y = (self.cy - self.h / 2) * H
        return QRectF(x, y, self.w * W, self.h * H)

    def iou(self, other: "Box") -> float:
        ax1 = self.cx - self.w / 2;  ax2 = self.cx + self.w / 2
        ay1 = self.cy - self.h / 2;  ay2 = self.cy + self.h / 2
        bx1 = other.cx - other.w / 2; bx2 = other.cx + other.w / 2
        by1 = other.cy - other.h / 2; by2 = other.cy + other.h / 2
        ix1 = max(ax1, bx1); ix2 = min(ax2, bx2)
        iy1 = max(ay1, by1); iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


def load_boxes(path: Path) -> list[Box]:
    if not path or not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                boxes.append(Box(int(parts[0]), float(parts[1]),
                                 float(parts[2]), float(parts[3]),
                                 float(parts[4])))
            except ValueError:
                pass
    return boxes


def load_yaml_names(yaml_path: Optional[Path]) -> dict[int, str]:
    if not yaml_path or not yaml_path.exists():
        return {}
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def find_image(folder: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


def match_boxes(
    a_boxes: list[Box],
    b_boxes: list[Box],
    iou_thresh: float = 0.45,
) -> tuple[list[tuple[Box, Box]], list[Box], list[Box]]:
    """Greedy IoU matching (동일 클래스 필수).
    Returns: matched[(ba, bb)], only_a, only_b
    """
    used_b: set[int] = set()
    matched: list[tuple[Box, Box]] = []
    only_a: list[Box] = []

    for ba in a_boxes:
        best_iou = iou_thresh
        best_j = -1
        for j, bb in enumerate(b_boxes):
            if j in used_b or ba.cls != bb.cls:
                continue
            v = ba.iou(bb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0:
            matched.append((ba, b_boxes[best_j]))
            used_b.add(best_j)
        else:
            only_a.append(ba)

    only_b = [bb for j, bb in enumerate(b_boxes) if j not in used_b]
    return matched, only_a, only_b


# ── 캔버스 ───────────────────────────────────────────────────────────────────

class CompareCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("background:#111;")

        self._base_img: Optional[QImage] = None
        self._matched:  list[tuple[Box, Box]] = []
        self._only_a:   list[Box] = []
        self._only_b:   list[Box] = []
        self._names_a:  dict[int, str] = {}
        self._names_b:  dict[int, str] = {}

        self._show_matched = True
        self._show_a = True
        self._show_b = True
        self._highlight_only = False   # 차이 박스만 크게 하이라이트
        self._zoom = 1.0

    def set_data(
        self,
        img: Optional[QImage],
        matched: list[tuple[Box, Box]],
        only_a: list[Box],
        only_b: list[Box],
        names_a: dict[int, str],
        names_b: dict[int, str],
    ):
        self._base_img = img
        self._matched  = matched
        self._only_a   = only_a
        self._only_b   = only_b
        self._names_a  = names_a
        self._names_b  = names_b
        self._render()

    def set_show(
        self,
        matched: Optional[bool] = None,
        a: Optional[bool] = None,
        b: Optional[bool] = None,
        highlight_only: Optional[bool] = None,
    ):
        if matched is not None:        self._show_matched = matched
        if a is not None:              self._show_a = a
        if b is not None:              self._show_b = b
        if highlight_only is not None: self._highlight_only = highlight_only
        self._render()

    def set_zoom(self, factor: float):
        self._zoom = factor
        self._render()

    # ── rendering ────────────────────────────────────────────────────────────

    def _render(self):
        if self._base_img is None:
            self.clear()
            self.setText("이미지 없음")
            return

        W, H = self._base_img.width(), self._base_img.height()
        zW = max(1, int(W * self._zoom))
        zH = max(1, int(H * self._zoom))

        canvas = self._base_img.scaled(
            zW, zH, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        aW, aH = canvas.width(), canvas.height()

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        font_tag = QFont("Arial", 10)
        font_tag.setBold(True)
        painter.setFont(font_tag)
        fm = QFontMetrics(font_tag)

        # 하이라이트 전용 모드: 차이 박스 영역을 흐리게 마스킹
        if self._highlight_only and (self._only_a or self._only_b):
            overlay = QColor(0, 0, 0, 100)
            painter.fillRect(0, 0, aW, aH, overlay)

        def _label_a(cls: int) -> str:
            return self._names_a.get(cls, str(cls))

        def _label_b(cls: int) -> str:
            return self._names_b.get(cls, str(cls))

        def draw_box(
            box: Box,
            border_col: QColor,
            fill_col: QColor,
            tag: str,
            thick: int = 2,
        ):
            r = box.rect_px(aW, aH)
            painter.setBrush(QBrush(fill_col))
            painter.setPen(QPen(border_col, thick))
            painter.drawRect(r)

            tw = fm.horizontalAdvance(tag) + 6
            th = fm.height() + 3
            tx = max(0.0, r.x())
            ty = r.y() - th if r.y() >= th else r.y() + r.height()

            bg = QColor(border_col)
            bg.setAlpha(210)
            painter.fillRect(QRectF(tx, ty, tw, th), bg)
            painter.setPen(Qt.white)
            painter.drawText(QPointF(tx + 3, ty + th - 4), tag)

        # ① 공통 박스 (회색, 얇게)
        if self._show_matched:
            for ba, _ in self._matched:
                tag = f"{ba.cls}:{_label_a(ba.cls)}"
                draw_box(ba, COL_M_BORDER, COL_M_FILL, tag, thick=1)

        # ② A만 있는 박스 (빨강, 두껍게)
        if self._show_a:
            thick = 3 if self._highlight_only else 2
            for ba in self._only_a:
                tag = f"A {ba.cls}:{_label_a(ba.cls)}"
                draw_box(ba, COL_A_BORDER, COL_A_FILL, tag, thick=thick)

        # ③ B만 있는 박스 (파랑, 두껍게)
        if self._show_b:
            thick = 3 if self._highlight_only else 2
            for bb in self._only_b:
                tag = f"B {bb.cls}:{_label_b(bb.cls)}"
                draw_box(bb, COL_B_BORDER, COL_B_FILL, tag, thick=thick)

        # 차이 없을 때 안내 텍스트
        if not self._only_a and not self._only_b:
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            painter.setPen(QColor("#44CC44"))
            painter.drawText(
                QRectF(0, 0, aW, aH),
                Qt.AlignCenter,
                "✓  두 라벨이 동일합니다",
            )

        painter.end()
        self.setPixmap(QPixmap.fromImage(canvas))
        self.resize(aW, aH)


# ── 메인 창 ──────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("라벨 비교 — Compare Labels")
        self.resize(1440, 900)

        # 상태
        self._folder_a:   Optional[Path] = None
        self._folder_b:   Optional[Path] = None
        self._folder_img: Optional[Path] = None
        self._yaml_a:     Optional[Path] = None
        self._yaml_b:     Optional[Path] = None
        self._names_a:    dict[int, str] = {}
        self._names_b:    dict[int, str] = {}
        self._stems:      list[str] = []
        self._diff_info:  dict[str, dict] = {}   # stem → {diff, only_a, only_b, matched}

        self._build_ui()
        self._setup_shortcuts()
        self._apply_theme()

    # ── UI 구성 ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── 왼쪽 패널 ────────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(290)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 8)
        lv.setSpacing(4)

        # 폴더/yaml 선택 그룹
        pickers = [
            ("A 폴더 (기준)",    "#FF5555", "folder", "A"),
            ("B 폴더 (비교)",    "#4488FF", "folder", "B"),
            ("이미지 폴더 (선택)", "#888888", "folder", "img"),
            ("A Config (yaml)", "#FF5555", "yaml",   "A"),
            ("B Config (yaml)", "#4488FF", "yaml",   "B"),
        ]
        for title, color, kind, which in pickers:
            grp = QGroupBox(title)
            grp.setStyleSheet(
                f"QGroupBox {{ color:{color}; font-weight:bold; "
                f"border:1px solid #444; border-radius:4px; "
                f"margin-top:6px; padding-top:4px; }}"
            )
            gv = QVBoxLayout(grp)
            gv.setContentsMargins(6, 4, 6, 4)
            gv.setSpacing(2)
            lbl = QLabel("(선택 안 됨)")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color:#999; font-size:10px;")
            btn_text = "폴더 선택" if kind == "folder" else "yaml 선택"
            btn = QPushButton(btn_text)
            btn.setStyleSheet(
                f"background:{color}; color:white; font-weight:bold; "
                f"padding:3px 8px; border-radius:4px;"
            )
            btn.clicked.connect(
                (lambda _, k=kind, w=which: self._pick_folder(w))
                if kind == "folder"
                else (lambda _, w=which: self._pick_yaml(w))
            )
            gv.addWidget(lbl)
            gv.addWidget(btn)
            lv.addWidget(grp)

            # 레이블 참조 저장
            tag = f"{kind}_{which}"
            setattr(self, f"_lbl_{tag}", lbl)

        # 새로고침 버튼
        btn_go = QPushButton("⟳  비교 시작")
        btn_go.setStyleSheet(
            "background:#00C851; color:white; font-weight:bold; "
            "padding:6px; border-radius:5px; font-size:13px;"
        )
        btn_go.clicked.connect(self._build_list)
        lv.addWidget(btn_go)

        # 통계
        self._lbl_stats = QLabel("")
        self._lbl_stats.setWordWrap(True)
        self._lbl_stats.setStyleSheet("color:#aaa; font-size:10px; padding:2px;")
        lv.addWidget(self._lbl_stats)

        # 목록
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#444;")
        lv.addWidget(sep)

        lv.addWidget(QLabel("이미지 목록  (차이 큰 순)"))
        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget { background:#161616; color:#eee; border:none; }"
            "QListWidget::item { padding:3px 6px; }"
            "QListWidget::item:selected { background:#2a4a7f; }"
        )
        self._list.currentRowChanged.connect(self._on_select)
        lv.addWidget(self._list)

        splitter.addWidget(left)

        # ── 오른쪽 패널 ──────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.setSpacing(4)

        # 컨트롤 바
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        self._chk_a = QCheckBox("A만 있는 박스 (빨강)")
        self._chk_b = QCheckBox("B만 있는 박스 (파랑)")
        self._chk_m = QCheckBox("공통 박스 (회색)")
        self._chk_h = QCheckBox("차이만 하이라이트")

        for chk in (self._chk_a, self._chk_b, self._chk_m, self._chk_h):
            chk.setChecked(True)
            chk.stateChanged.connect(self._on_toggle)

        self._chk_m.setChecked(True)
        self._chk_h.setChecked(False)

        self._chk_a.setStyleSheet("color:#FF6666; font-weight:bold;")
        self._chk_b.setStyleSheet("color:#5599FF; font-weight:bold;")
        self._chk_m.setStyleSheet("color:#aaaaaa;")
        self._chk_h.setStyleSheet("color:#FFD700; font-weight:bold;")

        ctrl.addWidget(self._chk_a)
        ctrl.addWidget(self._chk_b)
        ctrl.addWidget(self._chk_m)
        ctrl.addWidget(self._chk_h)
        ctrl.addStretch()

        # 줌
        ctrl.addWidget(QLabel("확대:"))
        self._zoom_sl = QSlider(Qt.Horizontal)
        self._zoom_sl.setRange(20, 400)
        self._zoom_sl.setValue(100)
        self._zoom_sl.setFixedWidth(130)
        self._zoom_sl.valueChanged.connect(self._on_zoom)
        self._lbl_zoom = QLabel("100%")
        self._lbl_zoom.setFixedWidth(44)
        ctrl.addWidget(self._zoom_sl)
        ctrl.addWidget(self._lbl_zoom)

        rv.addLayout(ctrl)

        # 정보 바
        self._info_bar = QLabel("폴더를 선택한 뒤 [비교 시작]을 눌러주세요.")
        self._info_bar.setStyleSheet(
            "background:#222; color:#ccc; padding:5px 8px; "
            "font-size:12px; border-radius:3px;"
        )
        rv.addWidget(self._info_bar)

        # 클래스별 박스 개수 테이블
        self._cls_table = QTableWidget(0, 5)
        self._cls_table.setHorizontalHeaderLabels(["cls", "이름", "A", "B", "Δ"])
        self._cls_table.setMaximumHeight(140)
        self._cls_table.setMinimumHeight(60)
        self._cls_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._cls_table.setSelectionMode(QTableWidget.NoSelection)
        self._cls_table.setFocusPolicy(Qt.NoFocus)
        self._cls_table.verticalHeader().setVisible(False)
        self._cls_table.setShowGrid(True)
        self._cls_table.setStyleSheet(
            "QTableWidget { background:#1a1a1a; color:#eee; "
            "  gridline-color:#333; border:none; font-size:11px; }"
            "QHeaderView::section { background:#2a2a2a; color:#aaa; "
            "  padding:2px 4px; border:none; font-size:11px; }"
        )
        hdr = self._cls_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._cls_table.verticalHeader().setDefaultSectionSize(20)
        rv.addWidget(self._cls_table)

        # 캔버스 스크롤
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setStyleSheet("QScrollArea { background:#111; border:none; }")
        self._canvas = CompareCanvas()
        scroll.setWidget(self._canvas)
        rv.addWidget(scroll)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Up),   self, self._prev_image)
        QShortcut(QKeySequence(Qt.Key_Down), self, self._next_image)
        QShortcut(QKeySequence("H"),         self, self._toggle_highlight)

    def _apply_theme(self):
        self.setStyleSheet(
            "QMainWindow, QWidget { background:#1e1e1e; color:#eee; }"
            "QLabel { color:#eee; }"
            "QPushButton { border-radius:4px; }"
            "QCheckBox { spacing:5px; }"
            "QSlider::groove:horizontal { background:#444; height:4px; border-radius:2px; }"
            "QSlider::handle:horizontal { background:#aaa; width:12px; height:12px; "
            "  margin:-4px 0; border-radius:6px; }"
        )

    # ── 슬롯 ─────────────────────────────────────────────────────────────────

    def _pick_folder(self, which: str):
        path = QFileDialog.getExistingDirectory(self, f"{which} 폴더 선택")
        if not path:
            return
        p = Path(path)
        lbl = getattr(self, f"_lbl_folder_{which}", None)
        if lbl:
            lbl.setText(p.name)
            lbl.setToolTip(str(p))

        if which == "A":
            self._folder_a = p
            if not self._yaml_a:
                self._try_auto_yaml("A", p)
        elif which == "B":
            self._folder_b = p
            if not self._yaml_b:
                self._try_auto_yaml("B", p)
        else:
            self._folder_img = p

    def _try_auto_yaml(self, which: str, folder: Path):
        """폴더 근방에서 yaml 자동 감지."""
        candidates = (
            list(folder.glob("*.yaml"))
            + list(folder.parent.glob("*.yaml"))
            + list((folder.parent / "config").glob("*.yaml"))
        )
        # dataset yaml 우선 (names 키 있는 것)
        for c in candidates:
            names = load_yaml_names(c)
            if names:
                self._pick_yaml_path(which, c)
                return

    def _pick_yaml(self, which: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"{which} yaml 선택", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self._pick_yaml_path(which, Path(path))

    def _pick_yaml_path(self, which: str, p: Path):
        lbl = getattr(self, f"_lbl_yaml_{which}", None)
        if lbl:
            lbl.setText(p.name)
            lbl.setToolTip(str(p))
        if which == "A":
            self._yaml_a  = p
            self._names_a = load_yaml_names(p)
        else:
            self._yaml_b  = p
            self._names_b = load_yaml_names(p)

    # ── 목록 구성 ────────────────────────────────────────────────────────────

    def _build_list(self):
        if not self._folder_a or not self._folder_b:
            self._lbl_stats.setText("A·B 폴더를 모두 선택하세요.")
            return

        def stems(folder: Path) -> set[str]:
            return {
                p.stem
                for p in folder.iterdir()
                if p.suffix.lower() in IMG_EXTS or p.suffix == ".txt"
            }

        sa = stems(self._folder_a)
        sb = stems(self._folder_b)
        common = sorted(sa & sb)

        self._diff_info = {}
        for stem in common:
            ba = load_boxes(self._folder_a / (stem + ".txt"))
            bb = load_boxes(self._folder_b / (stem + ".txt"))
            matched, only_a, only_b = match_boxes(ba, bb)
            self._diff_info[stem] = {
                "diff":    len(only_a) + len(only_b),
                "only_a":  len(only_a),
                "only_b":  len(only_b),
                "matched": len(matched),
            }

        # 차이 큰 순 정렬
        self._stems = sorted(common, key=lambda s: -self._diff_info[s]["diff"])

        n_diff = sum(1 for v in self._diff_info.values() if v["diff"] > 0)
        self._lbl_stats.setText(
            f"공통: {len(common)}장  |  차이 있음: {n_diff}장\n"
            f"A전용 파일: {len(sa-sb)}  |  B전용 파일: {len(sb-sa)}"
        )

        self._list.clear()
        for stem in self._stems:
            info = self._diff_info[stem]
            d = info["diff"]
            item_text = (
                f"[±{d:3d}]  {stem}"
                if d > 0
                else f"[ =  ]  {stem}"
            )
            item = QListWidgetItem(item_text)
            if d > 0:
                item.setForeground(QColor("#FF8866"))
                item.setToolTip(
                    f"A만: {info['only_a']}  B만: {info['only_b']}  공통: {info['matched']}"
                )
            else:
                item.setForeground(QColor("#66BB66"))
                item.setToolTip("두 라벨이 동일합니다.")
            self._list.addItem(item)

        if self._stems:
            self._list.setCurrentRow(0)

    # ── 이미지 로드 & 표시 ───────────────────────────────────────────────────

    def _on_select(self, row: int):
        if row < 0 or row >= len(self._stems):
            return
        stem = self._stems[row]
        ba   = load_boxes(self._folder_a / (stem + ".txt"))
        bb   = load_boxes(self._folder_b / (stem + ".txt"))
        matched, only_a, only_b = match_boxes(ba, bb)

        # 이미지 탐색: 이미지 폴더 → A 폴더 → B 폴더
        img: Optional[QImage] = None
        for folder in (self._folder_img, self._folder_a, self._folder_b):
            if folder:
                p = find_image(folder, stem)
                if p:
                    q = QImage(str(p))
                    if not q.isNull():
                        img = q
                        break

        self._canvas.set_data(img, matched, only_a, only_b,
                              self._names_a, self._names_b)

        info = self._diff_info.get(stem, {})
        d = info.get("diff", 0)
        self._info_bar.setText(
            f"📄 {stem}    "
            f"공통: {info.get('matched', 0)}  │  "
            f"A만 (빨강): {info.get('only_a', 0)}  │  "
            f"B만 (파랑): {info.get('only_b', 0)}  │  "
            f"총 차이: ±{d}    "
            f"{'⚠ 차이 있음' if d > 0 else '✓ 동일'}"
        )
        self._update_cls_table(matched, only_a, only_b)

    # ── 클래스별 개수 테이블 ─────────────────────────────────────────────────

    def _update_cls_table(
        self,
        matched: list[tuple[Box, Box]],
        only_a:  list[Box],
        only_b:  list[Box],
    ):
        from collections import defaultdict

        cnt_a: dict[int, int] = defaultdict(int)
        cnt_b: dict[int, int] = defaultdict(int)

        for ba, bb in matched:
            cnt_a[ba.cls] += 1
            cnt_b[bb.cls] += 1
        for b in only_a:
            cnt_a[b.cls] += 1
        for b in only_b:
            cnt_b[b.cls] += 1

        all_cls = sorted(set(cnt_a) | set(cnt_b))

        tbl = self._cls_table
        tbl.setRowCount(len(all_cls))

        for row, cls in enumerate(all_cls):
            a_cnt = cnt_a.get(cls, 0)
            b_cnt = cnt_b.get(cls, 0)
            delta = b_cnt - a_cnt
            name  = self._names_a.get(cls) or self._names_b.get(cls) or "?"

            def _cell(text: str, align=Qt.AlignCenter) -> QTableWidgetItem:
                item = QTableWidgetItem(text)
                item.setTextAlignment(align)
                return item

            tbl.setItem(row, 0, _cell(str(cls)))
            name_item = _cell(name, Qt.AlignLeft | Qt.AlignVCenter)
            tbl.setItem(row, 1, name_item)

            a_item = _cell(str(a_cnt))
            a_item.setForeground(QColor("#FF7777") if a_cnt > 0 else QColor("#555"))
            tbl.setItem(row, 2, a_item)

            b_item = _cell(str(b_cnt))
            b_item.setForeground(QColor("#5599FF") if b_cnt > 0 else QColor("#555"))
            tbl.setItem(row, 3, b_item)

            d_text  = f"{delta:+d}" if delta != 0 else "="
            d_item  = _cell(d_text)
            if delta == 0:
                d_item.setForeground(QColor("#66BB66"))
            elif delta > 0:
                d_item.setForeground(QColor("#5599FF"))   # B 더 많음
            else:
                d_item.setForeground(QColor("#FF7777"))   # A 더 많음
            tbl.setItem(row, 4, d_item)

    # ── 컨트롤 슬롯 ──────────────────────────────────────────────────────────

    def _on_toggle(self):
        self._canvas.set_show(
            matched=self._chk_m.isChecked(),
            a=self._chk_a.isChecked(),
            b=self._chk_b.isChecked(),
            highlight_only=self._chk_h.isChecked(),
        )

    def _on_zoom(self, val: int):
        self._lbl_zoom.setText(f"{val}%")
        self._canvas.set_zoom(val / 100)

    def _prev_image(self):
        row = self._list.currentRow()
        if row > 0:
            self._list.setCurrentRow(row - 1)

    def _next_image(self):
        row = self._list.currentRow()
        if row < self._list.count() - 1:
            self._list.setCurrentRow(row + 1)

    def _toggle_highlight(self):
        self._chk_h.setChecked(not self._chk_h.isChecked())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self._prev_image()
        elif event.key() == Qt.Key_Down:
            self._next_image()
        else:
            super().keyPressEvent(event)


# ── 엔트리 ───────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
