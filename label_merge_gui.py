#!/usr/bin/env python3
"""label_merge_gui.py — 라벨 폴더 비교·병합 GUI (git diff · merge 방식)

scripts/label_merge.py 의 GUI 버전. 시작하면 폴더 두 개(기존·새 버전)를 고르고,
이미지마다 어떤 박스가 추가·삭제·변경됐는지 목록과 이미지 위 박스로 보여준다.

실행
----
  python label_merge_gui.py

화면
----
  왼쪽  : 파일 목록 (M 변경 / A 새 버전에만 / D 기존에만 / = 동일), 박스 수 +, -, ~
  가운데: 이미지 위 박스
            빨강 = 기존에만 (-)    파랑 = 새 버전에만 (+)
            주황 = 같은 위치, 클래스 다름 (~)    회색 = 양쪽 같음
  오른쪽: 선택한 이미지의 박스 차이 목록 (클릭하면 해당 박스 강조)

단축키
------
  ↑ / ↓ (W / S)  이전 / 다음 파일
  1 / 2 / 3 / 4  기존에만 · 새 버전에만 · 클래스 다름 · 같음 박스 보이기 토글
  F              필터 전환 (변경만 ↔ 전체)
  Ctrl+O         폴더 다시 고르기
  Ctrl+M         병합 저장

입력 폴더는 수정하지 않는다. 병합 결과는 따로 고른 폴더에만 저장한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QShortcut, QSizePolicy, QSplitter, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))
from label_merge import Box, FileDiff, diff_boxes, read_boxes, run_merge, scan  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")

# BGR (cv2 그리기용)
COL_MINUS = (60, 60, 230)    # 빨강: 기존에만
COL_PLUS = (230, 140, 30)    # 파랑: 새 버전에만
COL_TILDE = (0, 165, 255)    # 주황: 클래스 다름
COL_SAME = (170, 170, 170)   # 회색: 같음
COL_HL = (0, 255, 255)       # 노랑: 강조


def imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def config_dirs() -> List[Path]:
    return sorted(p for p in SCRIPT_DIR.glob("config*") if (p / "obj.names").exists())


def load_names(path: Optional[Path]) -> Optional[List[str]]:
    if not path or not Path(path).exists():
        return None
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# 시작 다이얼로그: 폴더 고르기
# ══════════════════════════════════════════════════════════════════════════════


class FolderDialog(QDialog):
    def __init__(self, parent=None, prev: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("라벨 비교 — 폴더 선택")
        self.setMinimumWidth(640)
        prev = prev or {}
        form = QFormLayout(self)

        self.ours = self._path_row(form, "기존 폴더 (ours)", prev.get("ours", ""))
        self.theirs = self._path_row(form, "새 폴더 (theirs)", prev.get("theirs", ""))

        self.config = QComboBox()
        self.config.addItem("(클래스 이름 없음: 번호로 표시)", None)
        for c in config_dirs():
            self.config.addItem(c.name, c / "obj.names")
        want = prev.get("config", "config_indonesia")
        for i in range(self.config.count()):
            if self.config.itemText(i) == want:
                self.config.setCurrentIndex(i)
        form.addRow("클래스 config", self.config)

        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.05, 0.99)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(prev.get("iou", 0.5))
        form.addRow("같은 박스 IoU 기준", self.iou)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("비교 시작")
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _path_row(self, form, label, value):
        row = QHBoxLayout()
        edit = QLineEdit(value)
        btn = QPushButton("찾아보기…")
        btn.clicked.connect(lambda: self._browse(edit, label))
        row.addWidget(edit)
        row.addWidget(btn)
        form.addRow(label, row)
        return edit

    def _browse(self, edit, label):
        d = QFileDialog.getExistingDirectory(self, label, edit.text() or str(Path.home()))
        if d:
            edit.setText(d)

    def _ok(self):
        o, t = Path(self.ours.text().strip()), Path(self.theirs.text().strip())
        if not o.is_dir() or not t.is_dir():
            QMessageBox.warning(self, "폴더 확인", "두 폴더를 모두 골라 주세요.")
            return
        if o.resolve() == t.resolve():
            QMessageBox.warning(self, "폴더 확인", "서로 다른 폴더를 골라 주세요.")
            return
        self.accept()

    def values(self) -> dict:
        return dict(ours=self.ours.text().strip(), theirs=self.theirs.text().strip(),
                    names=self.config.currentData(), config=self.config.currentText(),
                    iou=self.iou.value())


# ══════════════════════════════════════════════════════════════════════════════
# 병합 다이얼로그
# ══════════════════════════════════════════════════════════════════════════════


class MergeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("병합 저장")
        self.setMinimumWidth(620)
        g = QGridLayout(self)

        g.addWidget(QLabel("결과 폴더"), 0, 0)
        self.out = QLineEdit()
        b = QPushButton("찾아보기…")
        b.clicked.connect(lambda: self._browse(self.out))
        g.addWidget(self.out, 0, 1)
        g.addWidget(b, 0, 2)

        g.addWidget(QLabel("base 폴더 (선택)"), 1, 0)
        self.base = QLineEdit()
        self.base.setPlaceholderText("비워 두면 2-way, 넣으면 3-way 병합")
        b2 = QPushButton("찾아보기…")
        b2.clicked.connect(lambda: self._browse(self.base))
        g.addWidget(self.base, 1, 1)
        g.addWidget(b2, 1, 2)

        g.addWidget(QLabel("2-way 방식"), 2, 0)
        self.strategy = QComboBox()
        self.strategy.addItem("합집합 (union)", "union")
        self.strategy.addItem("기존 그대로 (ours)", "ours")
        self.strategy.addItem("새 버전 그대로 (theirs)", "theirs")
        g.addWidget(self.strategy, 2, 1)

        g.addWidget(QLabel("충돌 시"), 3, 0)
        self.conflict = QComboBox()
        self.conflict.addItem("기존 우선 (ours)", "ours")
        self.conflict.addItem("새 버전 우선 (theirs)", "theirs")
        g.addWidget(self.conflict, 3, 1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("병합 저장")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        g.addWidget(bb, 4, 0, 1, 3)

    def _browse(self, edit):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택", edit.text() or str(Path.home()))
        if d:
            edit.setText(d)


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 뷰
# ══════════════════════════════════════════════════════════════════════════════


class ImageView(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#202020; color:#bbb;")
        self._bgr = None

    def set_bgr(self, bgr):
        self._bgr = bgr
        self._refresh()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

    def _refresh(self):
        if self._bgr is None:
            return
        h, w = self._bgr.shape[:2]
        rgb = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class Row(QTreeWidgetItem):
    """+, -, ~ 열은 숫자로 정렬."""

    def __lt__(self, other):
        col = self.treeWidget().sortColumn() if self.treeWidget() else 0
        if col in (2, 3, 4):
            return int(self.text(col)) < int(other.text(col))
        return self.text(col) < other.text(col)


# ══════════════════════════════════════════════════════════════════════════════
# 메인 창
# ══════════════════════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.entries: List[dict] = []
        self.cur: Optional[dict] = None
        self.highlight: Optional[Box] = None
        self._build_ui()
        self._shortcuts()
        self.load(cfg)

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.resize(1600, 950)
        root = QWidget()
        v = QVBoxLayout(root)

        top = QHBoxLayout()
        self.summary = QLabel()
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.summary, 1)
        b_reopen = QPushButton("폴더 다시 고르기 (Ctrl+O)")
        b_reopen.clicked.connect(self.reopen)
        b_merge = QPushButton("병합 저장… (Ctrl+M)")
        b_merge.clicked.connect(self.merge)
        top.addWidget(b_reopen)
        top.addWidget(b_merge)
        v.addLayout(top)

        flt = QHBoxLayout()
        flt.addWidget(QLabel("필터"))
        self.filter = QComboBox()
        for text, key in [("변경만 (M·A·D)", "changed"), ("전체", "all"), ("M 변경", "M"),
                          ("A 새 버전에만", "A"), ("D 기존에만", "D"), ("~ 클래스 다름 있음", "tilde"),
                          ("= 동일", "=")]:
            self.filter.addItem(text, key)
        self.filter.currentIndexChanged.connect(self.refill)
        flt.addWidget(self.filter)
        self.search = QLineEdit()
        self.search.setPlaceholderText("파일명 검색")
        self.search.textChanged.connect(self.refill)
        flt.addWidget(self.search, 1)
        self.chk = {}
        for key, text, col in [("minus", "1 기존에만 (-)", "#e63c3c"), ("plus", "2 새 버전에만 (+)", "#1e8ce6"),
                               ("tilde", "3 클래스 다름 (~)", "#ffa500"), ("same", "4 같음", "#aaaaaa")]:
            c = QCheckBox(text)
            c.setChecked(True)
            c.setStyleSheet(f"color:{col}; font-weight:bold;")
            c.toggled.connect(self.redraw)
            self.chk[key] = c
            flt.addWidget(c)
        v.addLayout(flt)

        split = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["상태", "파일", "+", "-", "~", "박스"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setSortingEnabled(True)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in (0, 2, 3, 4, 5):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.tree.currentItemChanged.connect(self._on_select)
        split.addWidget(self.tree)

        self.view = ImageView()
        split.addWidget(self.view)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rv.addWidget(self.info)
        self.detail = QListWidget()
        self.detail.currentItemChanged.connect(self._on_detail)
        rv.addWidget(self.detail, 1)
        split.addWidget(right)

        split.setSizes([430, 900, 300])
        v.addWidget(split, 1)
        self.setCentralWidget(root)

    def _shortcuts(self):
        for keys, fn in [(("Down", "S"), lambda: self.step(1)), (("Up", "W"), lambda: self.step(-1)),
                         (("F",), self.toggle_filter), (("Ctrl+O",), self.reopen), (("Ctrl+M",), self.merge)]:
            for k in keys:
                QShortcut(QKeySequence(k), self, activated=fn)
        for i, key in enumerate(["minus", "plus", "tilde", "same"], 1):
            QShortcut(QKeySequence(str(i)), self, activated=lambda k=key: self.chk[k].toggle())

    # ── 데이터 ───────────────────────────────────────────────────────────────
    def load(self, cfg: dict):
        self.cfg = cfg
        self.names = load_names(cfg.get("names"))
        self.ours, self.theirs = Path(cfg["ours"]), Path(cfg["theirs"])
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            A, B = scan(self.ours), scan(self.theirs)
            self.entries = []
            tot = dict(M=0, A=0, D=0, same=0, plus=0, minus=0, tilde=0)
            for k in sorted(set(A) | set(B)):
                a, b = read_boxes(A.get(k)), read_boxes(B.get(k))
                d = diff_boxes(a, b, cfg["iou"])
                st = "A" if k not in A else "D" if k not in B else "M" if d.dirty else "="
                if st == "=":
                    tot["same"] += 1
                else:
                    tot[st] += 1
                tot["plus"] += len(d.only_b)
                tot["minus"] += len(d.only_a)
                tot["tilde"] += len(d.changed)
                self.entries.append(dict(key=k, status=st, a=a, b=b, diff=d,
                                         a_path=A.get(k), b_path=B.get(k)))
        finally:
            QApplication.restoreOverrideCursor()
        self.setWindowTitle(f"라벨 비교 — {self.ours.name}  ↔  {self.theirs.name}")
        self.summary.setText(
            f"<b>기존</b> {self.ours}<br><b>새 버전</b> {self.theirs}<br>"
            f"파일 {len(self.entries)}개: 변경 M {tot['M']} / 새 버전에만 A {tot['A']} / 기존에만 D {tot['D']} / "
            f"동일 {tot['same']} &nbsp;|&nbsp; 박스 <span style='color:#1e8ce6'>+{tot['plus']}</span> "
            f"<span style='color:#e63c3c'>-{tot['minus']}</span> "
            f"<span style='color:#ffa500'>~{tot['tilde']}</span> &nbsp;(IoU {cfg['iou']:.2f})")
        self.refill()

    def refill(self):
        key = self.filter.currentData()
        q = self.search.text().strip().lower()
        self.tree.setSortingEnabled(False)
        self.tree.clear()
        colors = {"M": "#ffa500", "A": "#1e8ce6", "D": "#e63c3c", "=": "#888888"}
        for e in self.entries:
            st, d = e["status"], e["diff"]
            if key == "changed" and st == "=":
                continue
            if key in ("M", "A", "D", "=") and st != key:
                continue
            if key == "tilde" and not d.changed:
                continue
            if q and q not in e["key"].lower():
                continue
            it = Row([st, e["key"], str(len(d.only_b)), str(len(d.only_a)),
                                  str(len(d.changed)), f"{len(e['a'])} → {len(e['b'])}"])
            it.setForeground(0, QColor(colors[st]))
            it.setData(0, Qt.UserRole + 1, e)
            self.tree.addTopLevelItem(it)
        self.tree.setSortingEnabled(True)
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        else:
            self.cur = None
            self.view.clear()
            self.view.setText("조건에 맞는 파일이 없습니다.")
            self.detail.clear()
            self.info.clear()

    # ── 선택 / 그리기 ────────────────────────────────────────────────────────
    def _on_select(self, it, _prev=None):
        if it is None:
            return
        self.cur = it.data(0, Qt.UserRole + 1)
        self.highlight = None
        self._fill_detail()
        self.redraw()

    def _find_image(self, e) -> Optional[Path]:
        for p in (e["b_path"], e["a_path"]):
            if p is None:
                continue
            for ext in IMG_EXTS:
                c = p.with_suffix(ext)
                if c.exists():
                    return c
        return None

    def cname(self, c: int) -> str:
        return self.names[c] if self.names and 0 <= c < len(self.names) else str(c)

    def _fill_detail(self):
        e = self.cur
        d: FileDiff = e["diff"]
        self.detail.blockSignals(True)
        self.detail.clear()

        def add(text, color, box):
            item = QListWidgetItem(text)
            item.setForeground(QColor(color))
            item.setData(Qt.UserRole, box)
            self.detail.addItem(item)

        for x in d.only_a:
            add(f"-  {self.cname(x.cls)}   ({x.cx:.3f}, {x.cy:.3f})  {x.w:.3f}×{x.h:.3f}", "#e63c3c", x)
        for y in d.only_b:
            add(f"+  {self.cname(y.cls)}   ({y.cx:.3f}, {y.cy:.3f})  {y.w:.3f}×{y.h:.3f}", "#1e8ce6", y)
        for x, y in d.changed:
            add(f"~  {self.cname(x.cls)} → {self.cname(y.cls)}   IoU {x.iou(y):.2f}", "#ff8c00", y)
        for x, _ in d.same:
            add(f"=  {self.cname(x.cls)}   ({x.cx:.3f}, {x.cy:.3f})", "#888888", x)
        self.detail.blockSignals(False)

        img = self._find_image(e)
        self.info.setText(
            f"<b>{e['key']}</b><br>상태 {e['status']} · 박스 {len(e['a'])} → {len(e['b'])}<br>"
            f"<span style='color:#e63c3c'>-{len(d.only_a)}</span> "
            f"<span style='color:#1e8ce6'>+{len(d.only_b)}</span> "
            f"<span style='color:#ffa500'>~{len(d.changed)}</span> = {len(d.same)}<br>"
            f"<small>기존: {e['a_path'] or '(없음)'}<br>새 버전: {e['b_path'] or '(없음)'}<br>"
            f"이미지: {img or '(없음: 회색 배경에 표시)'}</small>")

    def _on_detail(self, it, _prev=None):
        self.highlight = it.data(Qt.UserRole) if it else None
        self.redraw()

    def redraw(self):
        e = self.cur
        if e is None:
            return
        img_path = self._find_image(e)
        img = imread_unicode(img_path) if img_path else None
        if img is None:
            img = np.full((720, 1280, 3), 60, np.uint8)
            cv2.putText(img, "no image", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        h, w = img.shape[:2]
        t = max(2, round(min(h, w) / 400))
        d: FileDiff = e["diff"]

        def draw(box: Box, col, label):
            x1, y1 = int((box.cx - box.w / 2) * w), int((box.cy - box.h / 2) * h)
            x2, y2 = int((box.cx + box.w / 2) * w), int((box.cy + box.h / 2) * h)
            hl = self.highlight is not None and self.highlight == box
            cv2.rectangle(img, (x1, y1), (x2, y2), COL_HL if hl else col, t * 2 if hl else t)
            fs = 0.45 * t
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
            ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), COL_HL if hl else col, -1)
            cv2.putText(img, label, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(1, t // 2))

        if self.chk["same"].isChecked():
            for x, _ in d.same:
                draw(x, COL_SAME, self.cname(x.cls))
        if self.chk["minus"].isChecked():
            for x in d.only_a:
                draw(x, COL_MINUS, "- " + self.cname(x.cls))
        if self.chk["plus"].isChecked():
            for y in d.only_b:
                draw(y, COL_PLUS, "+ " + self.cname(y.cls))
        if self.chk["tilde"].isChecked():
            for x, y in d.changed:
                draw(y, COL_TILDE, f"~ {self.cname(x.cls)}>{self.cname(y.cls)}")
        self.view.set_bgr(img)

    # ── 동작 ─────────────────────────────────────────────────────────────────
    def step(self, n: int):
        cnt = self.tree.topLevelItemCount()
        if not cnt:
            return
        cur = self.tree.currentItem()
        i = self.tree.indexOfTopLevelItem(cur) if cur else -1
        self.tree.setCurrentItem(self.tree.topLevelItem(max(0, min(cnt - 1, i + n))))

    def toggle_filter(self):
        self.filter.setCurrentIndex(1 if self.filter.currentData() == "changed" else 0)

    def reopen(self):
        dlg = FolderDialog(self, self.cfg)
        if dlg.exec_() == QDialog.Accepted:
            self.load(dlg.values())

    def merge(self):
        dlg = MergeDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        out = dlg.out.text().strip()
        if not out:
            QMessageBox.warning(self, "병합", "결과 폴더를 골라 주세요.")
            return
        base = dlg.base.text().strip() or None
        force = False
        if Path(out).exists() and any(Path(out).iterdir()):
            if QMessageBox.question(self, "병합", f"결과 폴더가 비어 있지 않습니다.\n{out}\n그래도 덮어쓸까요?") \
                    != QMessageBox.Yes:
                return
            force = True
        logs = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            run_merge(self.ours, self.theirs, out, base=base, iou=self.cfg["iou"],
                      strategy=dlg.strategy.currentData(), conflict=dlg.conflict.currentData(),
                      names=self.names, force=force, log=logs.append)
        except ValueError as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "병합", str(e))
            return
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "병합 완료", "\n".join(l for l in logs if not l.startswith("─")))


def main():
    app = QApplication(sys.argv)
    dlg = FolderDialog()
    if dlg.exec_() != QDialog.Accepted:
        return 0
    win = MainWindow(dlg.values())
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
