"""
1차 자동 라벨링 GUI
- 모델 파일 선택
- 이미지 소스 폴더 선택
- 출력 폴더 선택 (덮어쓰기 / 추가 모드)
- 클래스 역변환: 모델 yaml ↔ 출력 yaml 이름 매칭
- Confidence / IoU 조정
- 진행 상황 실시간 표시
"""
import sys
from pathlib import Path

import torch
import yaml
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QDoubleSpinBox, QComboBox, QLineEdit, QSizePolicy, QButtonGroup, QRadioButton,
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ── 경로 기본값 ───────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR          # 파일이 프로젝트 루트에 있을 때
RUNS_DETECT  = PROJECT_ROOT / "auto" / "runs" / "detect"
UNLABELED    = PROJECT_ROOT / "auto" / "unlabeled"
AUTO_DIR     = PROJECT_ROOT / "auto"

def get_device() -> str:
    """환경에 따라 최적 디바이스 자동 감지 (mps → cuda → cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ══════════════════════════════════════════════════════════════════════════════
# yaml 유틸
# ══════════════════════════════════════════════════════════════════════════════
def load_yaml_names(yaml_path: Path) -> dict[int, str]:
    """yaml의 names 필드를 {int: str} 으로 반환."""
    with open(yaml_path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    names = d.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def build_class_remap(model_yaml: Path, target_yaml: Path) -> dict[int, int] | None:
    """모델 yaml 클래스 idx → 출력 yaml 클래스 idx 매핑 (이름 기준).
    모든 클래스가 1:1 매핑되면 dict 반환, 동일한 yaml이면 None(변환 불필요)."""
    model_names  = load_yaml_names(model_yaml)
    target_names = load_yaml_names(target_yaml)

    # 이름 → 출력 idx 역인덱스
    name_to_target = {v: k for k, v in target_names.items()}

    remap = {}
    for model_idx, name in model_names.items():
        if name not in name_to_target:
            raise ValueError(f"클래스 '{name}' 을 출력 yaml에서 찾을 수 없습니다.")
        remap[model_idx] = name_to_target[name]

    # 모든 idx가 동일하면 변환 불필요
    if all(k == v for k, v in remap.items()):
        return None
    return remap


def find_weight_files() -> list[Path]:
    """runs/detect/ 하위의 best.pt 파일들을 반환."""
    if not RUNS_DETECT.exists():
        return []
    return sorted(RUNS_DETECT.glob("*/weights/best.pt"))


def find_model_data_yaml(model_path: Path) -> Path | None:
    """학습 시 사용한 data yaml을 args.yaml 에서 추출."""
    args_yaml = model_path.parent.parent / "args.yaml"
    if not args_yaml.exists():
        return None
    with open(args_yaml, encoding="utf-8") as f:
        args = yaml.safe_load(f)
    data_val = args.get("data", "")
    # 절대경로 or 파일명만 있는 경우 모두 처리
    p = Path(data_val)
    if p.is_absolute() and p.exists():
        return p
    # auto/ 폴더 기준으로 탐색
    candidate = AUTO_DIR / p.name
    if candidate.exists():
        return candidate
    return None


def find_yaml_files() -> list[Path]:
    """AUTO_DIR 내 *.yaml 파일 목록."""
    if not AUTO_DIR.exists():
        return []
    return sorted(AUTO_DIR.glob("*.yaml"))


# ══════════════════════════════════════════════════════════════════════════════
# 워커 스레드
# ══════════════════════════════════════════════════════════════════════════════
class AutoLabelWorker(QThread):
    progress  = pyqtSignal(int, int)
    log       = pyqtSignal(str)
    finished  = pyqtSignal(int)
    error     = pyqtSignal(str)

    def __init__(self, model_path: Path, source: Path, output: Path,
                 conf: float = 0.5, iou: float = 0.45, imgsz: int = 640,
                 append: bool = False, class_remap: dict | None = None):
        super().__init__()
        self.model_path  = model_path
        self.source      = source
        self.output      = output
        self.conf        = conf
        self.iou         = iou
        self.imgsz       = imgsz
        self.append      = append
        self.class_remap = class_remap   # {model_cls: output_cls} or None
        self._stop       = False

    def stop(self):
        self._stop = True

    def _remap_cls(self, cls_id: int) -> int:
        if self.class_remap is None:
            return cls_id
        return self.class_remap.get(cls_id, cls_id)

    def run(self):
        try:
            from ultralytics import YOLO

            img_paths = sorted(
                p for p in self.source.iterdir()
                if p.suffix.lower() in IMG_EXTS
            )
            if not img_paths:
                self.error.emit(f"이미지가 없습니다: {self.source}")
                return

            total = len(img_paths)
            self.log.emit(f"이미지 {total}개 발견. 모델 로딩 중...")

            model = YOLO(str(self.model_path))
            self.log.emit(f"모델 로드 완료: {self.model_path.name}")

            if self.class_remap:
                self.log.emit(f"클래스 역변환: {self.class_remap}")

            self.output.mkdir(parents=True, exist_ok=True)
            batch_size = 32
            saved = 0

            for i in range(0, total, batch_size):
                if self._stop:
                    self.log.emit("중단됨.")
                    return

                batch = img_paths[i : i + batch_size]
                results = model.predict(
                    [str(p) for p in batch],
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=get_device(),
                    verbose=False,
                )

                for img_path, r in zip(batch, results):
                    txt_path = self.output / (img_path.stem + ".txt")
                    boxes = r.boxes
                    new_lines = []
                    if boxes is not None and len(boxes) > 0:
                        xywhn = boxes.xywhn.cpu().numpy()
                        cls   = boxes.cls.cpu().numpy().astype(int)
                        new_lines = [
                            f"{self._remap_cls(c)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"
                            for c, (cx, cy, bw, bh) in zip(cls, xywhn)
                        ]

                    if self.append and txt_path.exists():
                        existing = [l for l in txt_path.read_text().splitlines() if l.strip()]
                        merged   = existing + [l for l in new_lines if l not in existing]
                        txt_path.write_text("\n".join(merged) + "\n" if merged else "")
                    else:
                        txt_path.write_text("\n".join(new_lines) + "\n" if new_lines else "")
                    saved += 1

                self.progress.emit(min(i + batch_size, total), total)

            self.log.emit(f"완료. txt 파일 {saved}개 저장 → {self.output}")
            self.finished.emit(saved)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# 메인 윈도우
# ══════════════════════════════════════════════════════════════════════════════
class AutoLabelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._build_ui()
        self._refresh_model_combo()

    def _build_ui(self):
        self.setWindowTitle("NeuralD Auto Labeler  —  1차 자동 라벨링")
        self.resize(760, 580)
        self.setMinimumSize(620, 460)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # ── 모델 선택 ─────────────────────────────────────────────────────
        layout.addWidget(self._section("모델"))

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(220)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        model_row.addWidget(self.model_combo)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("모델 경로 (.pt)")
        self.model_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_row.addWidget(self.model_edit)

        btn_model = QPushButton("찾아보기...")
        btn_model.setFixedWidth(90)
        btn_model.clicked.connect(self._browse_model)
        model_row.addWidget(btn_model)
        layout.addLayout(model_row)

        # ── 클래스 역변환 ─────────────────────────────────────────────────
        layout.addWidget(self._section("클래스 역변환  (서브셋 모델 사용 시)"))

        remap_row1 = QHBoxLayout()
        remap_row1.addWidget(QLabel("모델 yaml:"))
        self.model_yaml_edit = QLineEdit()
        self.model_yaml_edit.setPlaceholderText("모델 학습에 사용한 yaml (자동 감지)")
        remap_row1.addWidget(self.model_yaml_edit)
        btn_my = QPushButton("찾아보기...")
        btn_my.setFixedWidth(90)
        btn_my.clicked.connect(self._browse_model_yaml)
        remap_row1.addWidget(btn_my)
        layout.addLayout(remap_row1)

        remap_row2 = QHBoxLayout()
        remap_row2.addWidget(QLabel("출력 yaml:"))
        self.target_yaml_combo = QComboBox()
        self.target_yaml_combo.setMinimumWidth(200)
        remap_row2.addWidget(self.target_yaml_combo)
        self.target_yaml_edit = QLineEdit()
        self.target_yaml_edit.setPlaceholderText("최종 출력 클래스 기준 yaml")
        remap_row2.addWidget(self.target_yaml_edit)
        btn_ty = QPushButton("찾아보기...")
        btn_ty.setFixedWidth(90)
        btn_ty.clicked.connect(self._browse_target_yaml)
        remap_row2.addWidget(btn_ty)
        layout.addLayout(remap_row2)

        self.remap_hint = QLabel("")
        self.remap_hint.setStyleSheet("color:#555; font-size:11px; margin-left:4px;")
        layout.addWidget(self.remap_hint)

        # ── 소스 폴더 ─────────────────────────────────────────────────────
        layout.addWidget(self._section("이미지 소스 폴더"))

        src_row = QHBoxLayout()
        self.src_edit = QLineEdit(str(UNLABELED))
        self.src_edit.setPlaceholderText("라벨링할 이미지가 있는 폴더")
        src_row.addWidget(self.src_edit)
        btn_src = QPushButton("찾아보기...")
        btn_src.setFixedWidth(90)
        btn_src.clicked.connect(self._browse_source)
        src_row.addWidget(btn_src)
        layout.addLayout(src_row)

        # ── 출력 폴더 ─────────────────────────────────────────────────────
        layout.addWidget(self._section("출력 폴더  (txt 저장)"))

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(str(UNLABELED))
        self.out_edit.setPlaceholderText("txt 파일을 저장할 폴더")
        out_row.addWidget(self.out_edit)
        btn_out = QPushButton("찾아보기...")
        btn_out.setFixedWidth(90)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        # ── 쓰기 모드 ─────────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.radio_overwrite = QRadioButton("덮어쓰기  (기존 txt 교체)")
        self.radio_append    = QRadioButton("추가  (기존 내용 유지, 새 박스 병합)")
        self.radio_overwrite.setChecked(True)
        self.mode_group.addButton(self.radio_overwrite, 0)
        self.mode_group.addButton(self.radio_append,    1)
        mode_row.addWidget(self.radio_overwrite)
        mode_row.addSpacing(20)
        mode_row.addWidget(self.radio_append)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ── 파라미터 ──────────────────────────────────────────────────────
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setFixedWidth(80)
        param_row.addWidget(self.conf_spin)

        param_row.addSpacing(20)
        param_row.addWidget(QLabel("IoU:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setFixedWidth(80)
        param_row.addWidget(self.iou_spin)
        param_row.addStretch()
        layout.addLayout(param_row)

        # ── 실행 버튼 + 진행바 ────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  자동 라벨링 시작")
        self.btn_run.setFixedHeight(36)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#1a73e8;color:white;border-radius:5px;"
            "font-weight:bold;font-size:13px;}"
            "QPushButton:hover{background:#1558b0;}"
            "QPushButton:disabled{background:#aaa;}")
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)

        self.btn_stop = QPushButton("■  중단")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setFixedWidth(80)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "QPushButton{background:#c0392b;color:white;border-radius:5px;font-weight:bold;}"
            "QPushButton:disabled{background:#aaa;}")
        self.btn_stop.clicked.connect(self._stop)
        run_row.addWidget(self.btn_stop)
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # ── 로그 ──────────────────────────────────────────────────────────
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Menlo", 10))
        self.log_box.setMaximumHeight(130)
        layout.addWidget(self.log_box)

        # 출력 yaml 콤보 초기화
        self._refresh_target_yaml_combo()

    # ── 섹션 레이블 헬퍼 ──────────────────────────────────────────────────
    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight:bold; color:#333; margin-top:4px;")
        return lbl

    # ── 모델 콤보 초기화 ──────────────────────────────────────────────────
    def _refresh_model_combo(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        weights = find_weight_files()
        for w in weights:
            label = w.parent.parent.name   # runs/detect/<name>/weights/best.pt → <name>
            self.model_combo.addItem(label, userData=w)
        self.model_combo.addItem("직접 선택...", userData=None)
        self.model_combo.blockSignals(False)

        if weights:
            self.model_combo.setCurrentIndex(0)
            self._apply_model_path(weights[0])

    def _refresh_target_yaml_combo(self):
        self.target_yaml_combo.blockSignals(True)
        self.target_yaml_combo.clear()
        yamls = find_yaml_files()
        for y in yamls:
            self.target_yaml_combo.addItem(y.name, userData=y)
        self.target_yaml_combo.addItem("직접 선택...", userData=None)
        self.target_yaml_combo.blockSignals(False)
        self.target_yaml_combo.currentIndexChanged.connect(self._on_target_yaml_combo_changed)

        # 기본값: dataset.yaml (가장 많은 클래스)
        default_idx = next(
            (i for i in range(self.target_yaml_combo.count())
             if self.target_yaml_combo.itemText(i) == "dataset.yaml"), 0
        )
        self.target_yaml_combo.setCurrentIndex(default_idx)
        data = self.target_yaml_combo.itemData(default_idx)
        if data:
            self.target_yaml_edit.setText(str(data))

    def _on_target_yaml_combo_changed(self, idx: int):
        path = self.target_yaml_combo.itemData(idx)
        if path is not None:
            self.target_yaml_edit.setText(str(path))
        self._update_remap_hint()

    def _apply_model_path(self, model_path: Path):
        """모델 경로 변경 시 model_edit + model_yaml 자동 감지."""
        self.model_edit.setText(str(model_path))
        detected = find_model_data_yaml(model_path)
        if detected:
            self.model_yaml_edit.setText(str(detected))
        else:
            self.model_yaml_edit.clear()
        self._update_remap_hint()

    def _on_model_combo_changed(self, idx: int):
        path = self.model_combo.itemData(idx)
        if path is not None:
            self._apply_model_path(path)

    def _update_remap_hint(self):
        """두 yaml이 모두 설정됐을 때 매핑 미리보기 표시."""
        my = self.model_yaml_edit.text().strip()
        ty = self.target_yaml_edit.text().strip()
        if not my or not ty:
            self.remap_hint.setText("")
            return
        try:
            remap = build_class_remap(Path(my), Path(ty))
            if remap is None:
                self.remap_hint.setText("✅ 클래스 ID 동일 — 변환 불필요")
                self.remap_hint.setStyleSheet("color:green; font-size:11px;")
            else:
                model_names  = load_yaml_names(Path(my))
                target_names = load_yaml_names(Path(ty))
                parts = [f"{model_names[k]}({k}→{v})" for k, v in sorted(remap.items())]
                self.remap_hint.setText("🔄 " + "  ".join(parts))
                self.remap_hint.setStyleSheet("color:#1a73e8; font-size:11px;")
        except Exception as e:
            self.remap_hint.setText(f"⚠️ {e}")
            self.remap_hint.setStyleSheet("color:red; font-size:11px;")

    # ── 파일/폴더 브라우저 ────────────────────────────────────────────────
    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "모델 파일 선택", str(RUNS_DETECT), "PyTorch 모델 (*.pt)"
        )
        if path:
            self._apply_model_path(Path(path))
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def _browse_model_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "모델 yaml 선택", str(AUTO_DIR), "YAML 파일 (*.yaml *.yml)"
        )
        if path:
            self.model_yaml_edit.setText(path)
            self._update_remap_hint()

    def _browse_target_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "출력 yaml 선택", str(AUTO_DIR), "YAML 파일 (*.yaml *.yml)"
        )
        if path:
            self.target_yaml_edit.setText(path)
            self.target_yaml_combo.setCurrentIndex(self.target_yaml_combo.count() - 1)
            self._update_remap_hint()

    def _browse_source(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 소스 폴더 선택",
                                                   self.src_edit.text())
        if folder:
            self.src_edit.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "출력 폴더 선택",
                                                   self.out_edit.text())
        if folder:
            self.out_edit.setText(folder)

    # ── 실행 / 중단 ───────────────────────────────────────────────────────
    def _run(self):
        model_path = Path(self.model_edit.text().strip())
        source     = Path(self.src_edit.text().strip())
        output     = Path(self.out_edit.text().strip())

        if not model_path.exists():
            self._log(f"[오류] 모델 파일 없음: {model_path}")
            return
        if not source.exists():
            self._log(f"[오류] 소스 폴더 없음: {source}")
            return

        # 클래스 역변환 계산
        class_remap = None
        my = self.model_yaml_edit.text().strip()
        ty = self.target_yaml_edit.text().strip()
        if my and ty:
            try:
                class_remap = build_class_remap(Path(my), Path(ty))
            except Exception as e:
                self._log(f"[오류] 클래스 매핑 실패: {e}")
                return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_box.clear()
        self._log(f"모델: {model_path}")
        self._log(f"소스: {source}  →  출력: {output}")
        append = self.radio_append.isChecked()
        self._log(f"conf={self.conf_spin.value():.2f}  iou={self.iou_spin.value():.2f}"
                  f"  모드={'추가' if append else '덮어쓰기'}"
                  f"  역변환={'ON' if class_remap else 'OFF'}")

        self.worker = AutoLabelWorker(
            model_path=model_path,
            source=source,
            output=output,
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            append=append,
            class_remap=class_remap,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.btn_stop.setEnabled(False)

    # ── 시그널 핸들러 ─────────────────────────────────────────────────────
    def _on_progress(self, cur: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(cur)
        self.progress_bar.setFormat(f"{cur} / {total}  ({cur*100//total}%)")

    def _on_finished(self, saved: int):
        self._log(f"[완료] txt 파일 {saved}개 저장됨.")
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_error(self, msg: str):
        self._log(f"[오류]\n{msg}")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    win = AutoLabelWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
