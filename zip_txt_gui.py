"""
TXT 압축 GUI
- 폴더 선택 → 내부 .txt 파일만 zip 으로 압축
- 저장 위치: 선택 폴더와 같은 경로, 파일명은 폴더명.zip
"""
import sys
import zipfile
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QProgressBar, QTextEdit, QFileDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


# ══════════════════════════════════════════════════════════════════════════════
# 워커
# ══════════════════════════════════════════════════════════════════════════════
class ZipWorker(QThread):
    progress = pyqtSignal(int, int)   # (현재, 전체)
    log      = pyqtSignal(str)
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
            self.log.emit(f".txt 파일 {total}개 발견 → 압축 시작")

            with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, f in enumerate(txt_files, 1):
                    zf.write(f, f.name)
                    self.progress.emit(i, total)

            self.log.emit(f"저장 완료: {self.zip_path}")
            self.finished.emit(str(self.zip_path), total)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# 메인 윈도우
# ══════════════════════════════════════════════════════════════════════════════
class ZipTxtWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("TXT 압축기")
        self.resize(540, 300)
        self.setMinimumSize(420, 240)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # ── 폴더 선택 ─────────────────────────────────────────────────────
        root.addWidget(self._section("폴더 선택"))

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText(".txt 파일이 있는 폴더")
        self.folder_edit.textChanged.connect(self._on_folder_changed)
        folder_row.addWidget(self.folder_edit)
        btn_browse = QPushButton("찾아보기...")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse)
        folder_row.addWidget(btn_browse)
        root.addLayout(folder_row)

        # ── 출력 zip 경로 (자동 설정, 읽기 전용) ─────────────────────────
        root.addWidget(self._section("저장될 zip 파일"))

        self.zip_label = QLabel("—")
        self.zip_label.setStyleSheet(
            "color:#555; font-size:12px; padding:4px 2px;")
        self.zip_label.setWordWrap(True)
        root.addWidget(self.zip_label)

        # ── 실행 버튼 ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  압축 시작")
        self.btn_run.setFixedHeight(36)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#1a73e8;color:white;border-radius:5px;"
            "font-weight:bold;font-size:13px;}"
            "QPushButton:hover{background:#1558b0;}"
            "QPushButton:disabled{background:#aaa;}")
        self.btn_run.clicked.connect(self._run)
        btn_row.addWidget(self.btn_run)
        root.addLayout(btn_row)

        # ── 진행바 ────────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        root.addWidget(self.progress_bar)

        # ── 로그 ──────────────────────────────────────────────────────────
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Menlo", 10))
        self.log_box.setMaximumHeight(90)
        root.addWidget(self.log_box)

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight:bold; color:#333; margin-top:2px;")
        return lbl

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택",
                                             str(Path.home()))
        if d:
            self.folder_edit.setText(d)

    def _on_folder_changed(self, text: str):
        p = Path(text.strip())
        if p.is_dir():
            zip_path = p.parent / f"{p.name}.zip"
            self.zip_label.setText(str(zip_path))
            self.btn_run.setEnabled(True)
        else:
            self.zip_label.setText("—")
            self.btn_run.setEnabled(False)

    def _run(self):
        folder = Path(self.folder_edit.text().strip())
        if not folder.is_dir():
            self._log("[오류] 유효한 폴더를 선택하세요.")
            return

        zip_path = folder.parent / f"{folder.name}.zip"

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._log(f"폴더: {folder}")

        self.worker = ZipWorker(folder, zip_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, cur: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(cur)
        self.progress_bar.setFormat(f"{cur} / {total}  ({cur * 100 // total}%)")

    def _on_finished(self, zip_path: str, count: int):
        self._log(f"[완료] .txt {count}개 → {zip_path}")
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.btn_run.setEnabled(True)

    def _on_error(self, msg: str):
        self._log(f"[오류]\n{msg}")
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)

    def _log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ZipTxtWindow()
    win.show()
    sys.exit(app.exec_())
