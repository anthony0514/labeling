import sys
import os
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QFileDialog, QSlider,
    QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QPushButton
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QCursor, QColor, QBrush
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF, QRectF


def load_class_count():
    try:
        path = resource_path(os.path.join("config", "class_count.txt"))
        with open(path, "r") as f:
            return int(f.read().strip())
    except:
        return 30

def imread_unicode(path):
    """한글 경로 포함된 이미지 안전하게 읽기"""
    try:
        with open(path, "rb") as f:
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"이미지 읽기 오류: {path} - {e}")
        return None

def resource_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.join(base_path, relative_path)

def get_color_for_class(cls: int) -> QColor:
    colors = [
        QColor(0, 255, 0), QColor(255, 0, 0), QColor(0, 0, 255), QColor(208, 177, 52),
        QColor(255, 0, 255), QColor(0, 255, 255), QColor(255, 165, 0),
        QColor(128, 0, 128), QColor(0, 128, 128), QColor(128, 128, 0),
    ]
    return colors[cls % len(colors)]

class ImageMarker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.class_count = load_class_count()
        obj_names_path = resource_path(os.path.join("config", "obj.names"))
        if os.path.exists(obj_names_path):
            with open(obj_names_path, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            self.class_names = []

        self.setWindowTitle("NeuralD BoxLabeler v1.5 - Ready")

        # 상태 변수들
        self.dragging_box_index = None
        self.drag_start_pos = None
        
        # 리사이즈 및 패닝 관련 변수
        self.resizing_box_index = None
        self.resize_handle = None
        self.panning = False
        self.pan_start_pos = None
        self.HANDLE_SIZE = 8
        
        # 미니맵 관련 변수
        self.minimap_size = 200  # 미니맵 크기
        self.show_minimap = True  # 미니맵 표시 여부

        # 클래스 프리뷰 관련 변수
        self.show_class_preview = True
        self.class_preview_pixmap = None
        
        # 수정된 박스 추적
        self.modified_boxes = set()  # 수정된 박스의 인덱스 저장


        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.slider_layout = QVBoxLayout()
        self.layout.addLayout(self.slider_layout)

        # 상단 버튼 레이아웃
        btn_layout = QHBoxLayout()
        self.help_btn = QPushButton("도움말 (H)")
        self.help_btn.setFixedWidth(100)
        self.help_btn.clicked.connect(self.show_guide)
        btn_layout.addWidget(self.help_btn)
        
        # 미니맵 토글 버튼 추가
        self.minimap_btn = QPushButton("미니맵 ON (M)")
        self.minimap_btn.setFixedWidth(120)
        self.minimap_btn.clicked.connect(self.toggle_minimap)
        btn_layout.addWidget(self.minimap_btn)

        # 클래스 프리뷰 토글 버튼
        self.preview_btn = QPushButton("클래스 미리보기 (P)")
        self.preview_btn.setFixedWidth(150)
        self.preview_btn.clicked.connect(self.toggle_class_preview)
        btn_layout.addWidget(self.preview_btn)
        
        btn_layout.addStretch() 
        self.slider_layout.addLayout(btn_layout)

        # 이미지 슬라이더
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximumHeight(20)
        self.image_slider.valueChanged.connect(self.slider_image_changed)
        self.image_value_display = QLabel("0")
        self.image_value_display.setMaximumHeight(20)
        
        image_slider_row = QHBoxLayout()
        image_slider_row.addWidget(QLabel("Image num:"))
        image_slider_row.addWidget(self.image_slider)
        image_slider_row.addWidget(self.image_value_display)
        self.slider_layout.addLayout(image_slider_row)

        # 클래스 슬라이더
        self.class_slider = QSlider(Qt.Horizontal)
        self.class_slider.setMinimum(0)
        self.class_slider.setMaximum(self.class_count - 1)
        self.class_slider.setValue(0)
        self.class_slider.setMaximumHeight(20)
        self.class_slider.valueChanged.connect(self.slider_class_changed)
        self.class_value_display = QLabel("0")
        self.class_value_display.setMaximumHeight(20)
        
        class_slider_row = QHBoxLayout()
        class_slider_row.addWidget(QLabel("Object id:"))
        class_slider_row.addWidget(self.class_slider)
        class_slider_row.addWidget(self.class_value_display)
        self.slider_layout.addLayout(class_slider_row)

        # 메인 캔버스
        self.label = QLabel(self)
        self.label.setMinimumSize(640, 480)
        self.label.setMouseTracking(True)
        self.layout.addWidget(self.label)

        self.resize(1200, 800)
        self.setMinimumSize(960, 600)

        self.image_folder = QFileDialog.getExistingDirectory(self, "마킹할 이미지 폴더 선택")
        if not self.image_folder:
            sys.exit()

        self.stack_file_path = os.path.join(self.image_folder, ".undo_redo_stack.json")
        self.undo_stack = []
        self.redo_stack = []
        self._active_edit_snapshot = None
        self._init_stack_file()

        self.image_list = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.png'))]
        self.image_index = 0
        self.current_class = 0
        self.boxes = []
        self.zoom = 1.0
        self.offset = QPoint(0, 0)
        self.start_point = None
        self.end_point = None
        self.drawing = False

        self.image_slider.setMaximum(len(self.image_list) - 1)
        self._load_class_preview_pixmap(self.current_class)
        self.load_image()

    def _serialize_state(self):
        return {
            "boxes": [list(box) for box in self.boxes],
            "modified_boxes": sorted(self.modified_boxes),
        }

    def _apply_state(self, state):
        self.boxes = [tuple(box) for box in state.get("boxes", [])]
        self.modified_boxes = set(state.get("modified_boxes", []))
        self.update_display()

    def _write_stack_file(self):
        payload = {
            "image_index": self.image_index if hasattr(self, "image_index") else 0,
            "image_file": self.image_list[self.image_index] if hasattr(self, "image_list") and self.image_list else "",
            "undo_stack": self.undo_stack,
            "redo_stack": self.redo_stack,
        }
        with open(self.stack_file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _init_stack_file(self):
        self.undo_stack = []
        self.redo_stack = []
        self._write_stack_file()

    def _push_undo_state(self):
        self.undo_stack.append(self._serialize_state())
        self.redo_stack.clear()
        self._write_stack_file()

    def undo_action(self):
        if not self.undo_stack:
            return
        current_state = self._serialize_state()
        prev_state = self.undo_stack.pop()
        self.redo_stack.append(current_state)
        self._apply_state(prev_state)
        self._write_stack_file()

    def redo_action(self):
        if not self.redo_stack:
            return
        current_state = self._serialize_state()
        next_state = self.redo_stack.pop()
        self.undo_stack.append(current_state)
        self._apply_state(next_state)
        self._write_stack_file()

    def toggle_minimap(self):
        """미니맵 표시/숨김 토글"""
        self.show_minimap = not self.show_minimap
        if self.show_minimap:
            self.minimap_btn.setText("미니맵 ON (M)")
        else:
            self.minimap_btn.setText("미니맵 OFF (M)")
        self.update_display()

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
            <tr><td><b>→ (Right)</b></td><td>: 저장 & 다음 이미지 (Next)</td></tr>
            <tr><td><b>← (Left)</b></td><td>: 저장 & 이전 이미지 (Prev)</td></tr>
            <tr><td><b>M</b></td><td>: 미니맵 ON/OFF <span style='color:red;'>(NEW v1.5)</span></td></tr>
            <tr><td><b>S</b></td><td>: 저장 (Save)</td></tr>
            <tr><td><b>F</b></td><td>: 화면 맞춤 (Fit to Screen)</td></tr>
            <tr><td><b>R</b></td><td>: 마우스 위치 박스 삭제 (Remove)</td></tr>
            <tr><td><b>BackSpace</b></td><td>: 마지막 박스 삭제</td></tr>
            <tr><td><b>C</b></td><td>: 전체 박스 삭제 (Clear All)</td></tr>
            <tr><td><b>Z</b></td><td>: 실행 취소 (Undo)</td></tr>
            <tr><td><b>U</b></td><td>: 다시 실행 (Redo)</td></tr>
            <tr><td><b>0 ~ 9</b></td><td>: 클래스 ID 변경 (커서가 박스 위면 해당 박스도 변경)</td></tr>
            <tr><td><b>H</b></td><td>: 도움말 보기 (Help)</td></tr>
            <tr><td><b>Q / Esc</b></td><td>: 종료 (Quit)</td></tr>
        </table>
        <hr>
        <h3>✨ v1.5 새로운 기능</h3>
        <ul>
            <li><b>미니맵</b> : 우측 하단에 전체 이미지 미리보기 표시</li>
            <li><b>코너 브래킷</b> : 수정된 박스는 모서리에 특별한 표시</li>
        </ul>
        """)
        msg.exec_()

    def showEvent(self, event):
        super().showEvent(event)
        self.reset_zoom()

    def reset_zoom(self):
        if hasattr(self, 'w') and hasattr(self, 'h'):
            self.initial_zoom = min(self.label.width() / self.w, self.label.height() / self.h)
            self.zoom = self.initial_zoom
            self.offset = QPoint((self.label.width() - int(self.w * self.zoom)) // 2,
                                 (self.label.height() - int(self.h * self.zoom)) // 2)
            self.update_display()

    def slider_image_changed(self, value):
        if self.image_index != value:
             if self.image_index < len(self.image_list):
                self.save_labels()
             self.image_index = value
             self.image_value_display.setText(str(value))
             self.boxes = []
             self.modified_boxes.clear()  # 새 이미지 로드 시 수정 기록 초기화
             self._init_stack_file()
             self.load_image()

    def slider_class_changed(self, value):
        self.current_class = value
        if value < len(self.class_names):
            self.class_value_display.setText(f"{value}: {self.class_names[value]}")
        else:
            self.class_value_display.setText(str(value))
        self._load_class_preview_pixmap(value)
        self.update_display()

    def _load_class_preview_pixmap(self, cls):
        img_path = resource_path(os.path.join("refs", f"{cls}.png"))
        if os.path.exists(img_path):
            self.class_preview_pixmap = QPixmap(img_path)
        else:
            self.class_preview_pixmap = None

    def toggle_class_preview(self):
        self.show_class_preview = not self.show_class_preview
        if self.show_class_preview:
            self.preview_btn.setText("클래스 미리보기 ON (P)")
        else:
            self.preview_btn.setText("클래스 미리보기 OFF (P)")
        self.update_display()

    def load_image(self):
        if self.image_index >= len(self.image_list):
            self.setWindowTitle("완료 - 더 이상 이미지가 없습니다.")
            return

        current_file = self.image_list[self.image_index]
        self.setWindowTitle(f"NeuralD Labeler v1.5 - {current_file} [{self.image_index + 1}/{len(self.image_list)}]")

        path = os.path.join(self.image_folder, current_file)
        self.image = imread_unicode(path)
        if self.image is None:
            return
            
        self.h, self.w = self.image.shape[:2]
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(rgb_image.data, self.w, self.h, self.w * 3, QImage.Format_RGB888)

        self.boxes = []
        self.modified_boxes.clear()
        label_path = os.path.join(self.image_folder, os.path.splitext(current_file)[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, bw, bh = map(float, parts)
                        x1 = (xc - bw / 2) * self.w
                        y1 = (yc - bh / 2) * self.h
                        x2 = (xc + bw / 2) * self.w
                        y2 = (yc + bh / 2) * self.h
                        self.boxes.append((int(cls), x1, y1, x2, y2))

        self.update_display()

    def draw_corner_bracket(self, painter, x, y, size, angle):
        """코너 브래킷 그리기"""
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        
        # L자 모양 그리기
        painter.drawLine(QPointF(0, 0), QPointF(size, 0))
        painter.drawLine(QPointF(0, 0), QPointF(0, size))
        
        painter.restore()

    def draw_minimap(self, painter):
        """미니맵 그리기"""
        if not self.show_minimap or not hasattr(self, 'qimage'):
            return
        
        # 미니맵 위치 (우측 하단)
        margin = 10
        minimap_x = self.label.width() - self.minimap_size - margin
        minimap_y = self.label.height() - self.minimap_size - margin
        
        # 미니맵 배경
        painter.fillRect(minimap_x - 5, minimap_y - 5, 
                        self.minimap_size + 10, self.minimap_size + 10, 
                        QColor(0, 0, 0, 180))
        
        # 이미지 축소판
        aspect_ratio = self.w / self.h
        if aspect_ratio > 1:
            mini_w = self.minimap_size
            mini_h = int(self.minimap_size / aspect_ratio)
        else:
            mini_h = self.minimap_size
            mini_w = int(self.minimap_size * aspect_ratio)
        
        mini_offset_x = minimap_x + (self.minimap_size - mini_w) // 2
        mini_offset_y = minimap_y + (self.minimap_size - mini_h) // 2
        
        scaled_mini = self.qimage.scaled(mini_w, mini_h, Qt.KeepAspectRatio, Qt.FastTransformation)
        painter.drawPixmap(mini_offset_x, mini_offset_y, QPixmap.fromImage(scaled_mini))
        
        # 현재 뷰포트 표시
        mini_scale_x = mini_w / self.w
        mini_scale_y = mini_h / self.h
        
        # 현재 보이는 영역 계산
        view_x = -self.offset.x() / self.zoom
        view_y = -self.offset.y() / self.zoom
        view_w = self.label.width() / self.zoom
        view_h = self.label.height() / self.zoom
        
        # 미니맵상의 뷰포트 사각형
        viewport_x = mini_offset_x + view_x * mini_scale_x
        viewport_y = mini_offset_y + view_y * mini_scale_y
        viewport_w = view_w * mini_scale_x
        viewport_h = view_h * mini_scale_y
        
        # 뷰포트 테두리
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.setBrush(QBrush(QColor(255, 255, 0, 50)))
        painter.drawRect(QRectF(viewport_x, viewport_y, viewport_w, viewport_h))
        
        # 박스들을 미니맵에 표시
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            color = get_color_for_class(cls)
            painter.setPen(QPen(color, 1))
            
            mini_x1 = mini_offset_x + x1 * mini_scale_x
            mini_y1 = mini_offset_y + y1 * mini_scale_y
            mini_x2 = mini_offset_x + x2 * mini_scale_x
            mini_y2 = mini_offset_y + y2 * mini_scale_y
            
            painter.drawRect(QRectF(mini_x1, mini_y1, mini_x2 - mini_x1, mini_y2 - mini_y1))

    def draw_class_preview(self, painter):
        """클래스 이미지 미리보기 (우상단)"""
        if not self.show_class_preview:
            return

        margin = 10
        size = self.minimap_size
        px = self.label.width() - size - margin
        py = margin

        # 배경
        painter.fillRect(px - 5, py - 5, size + 10, size + 10, QColor(0, 0, 0, 180))

        if self.class_preview_pixmap and not self.class_preview_pixmap.isNull():
            scaled = self.class_preview_pixmap.scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            draw_x = px + (size - scaled.width()) // 2
            draw_y = py + (size - scaled.height()) // 2
            painter.drawPixmap(draw_x, draw_y, scaled)
        else:
            painter.setPen(QPen(QColor(120, 120, 120)))
            painter.drawText(QRectF(px, py, size, size), Qt.AlignCenter,
                             f"No image\n({self.current_class}.png)")

        # 클래스 이름 레이블 (하단)
        cls = self.current_class
        cls_color = get_color_for_class(cls)
        label_rect = QRectF(px - 5, py + size + 5, size + 10, 22)
        painter.fillRect(label_rect, QColor(0, 0, 0, 180))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(cls_color))
        painter.drawEllipse(QPointF(px + 10, py + size + 16), 6, 6)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255)))
        cls_label = f"Class {cls}"
        if cls < len(self.class_names):
            cls_label += f": {self.class_names[cls]}"
        painter.drawText(QRectF(px + 22, py + size + 5, size - 17, 22), Qt.AlignVCenter, cls_label)

    def update_display(self):
        if not hasattr(self, 'qimage'): return

        display_pixmap = QPixmap(self.label.size())
        display_pixmap.fill(Qt.lightGray)
        painter = QPainter(display_pixmap)
        
        # 배경 이미지
        scaled_w = int(self.w * self.zoom)
        scaled_h = int(self.h * self.zoom)
        scaled = self.qimage.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(self.offset, QPixmap.fromImage(scaled))

        # 박스 그리기
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            color = get_color_for_class(cls)
            pen = QPen(color, 2)
            painter.setPen(pen)

            sx1 = (x1 * self.zoom) + self.offset.x()
            sy1 = (y1 * self.zoom) + self.offset.y()
            sw = (x2 - x1) * self.zoom
            sh = (y2 - y1) * self.zoom
            
            rect_f = QRectF(sx1, sy1, sw, sh)
            painter.drawRect(rect_f)
            
            label_text = f"{cls} {self.class_names[cls]}" if cls < len(self.class_names) else str(cls)
            painter.drawText(QPointF(sx1, sy1 - 5), label_text)

            # 수정된 박스는 코너 브래킷으로 표시
            if i in self.modified_boxes:
                painter.setPen(QPen(QColor(255, 255, 0), 3))  # 노란색 강조
                bracket_size = 15
                
                # 네 모서리에 L자 브래킷 그리기
                self.draw_corner_bracket(painter, sx1, sy1, bracket_size, 0)  # 좌상단
                self.draw_corner_bracket(painter, sx1 + sw, sy1, bracket_size, 90)  # 우상단
                self.draw_corner_bracket(painter, sx1, sy1 + sh, bracket_size, -90)  # 좌하단
                self.draw_corner_bracket(painter, sx1 + sw, sy1 + sh, bracket_size, 180)  # 우하단
                
                painter.setPen(pen)  # 원래 색상으로 복원
            
            # 기본 코너 핸들
            painter.setBrush(QBrush(color))
            r = 3.0
            corners = [(sx1, sy1), (sx1+sw, sy1), (sx1, sy1+sh), (sx1+sw, sy1+sh)]
            for cx, cy in corners:
                painter.drawEllipse(QPointF(cx, cy), r, r)
            painter.setBrush(Qt.NoBrush)

        # 그리기 중인 임시 박스
        if self.drawing and self.start_point and self.end_point:
            pen = QPen(Qt.red, 1, Qt.DashLine)
            painter.setPen(pen)
            
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            rx = min(x1, x2) * self.zoom + self.offset.x()
            ry = min(y1, y2) * self.zoom + self.offset.y()
            rw = abs(x2 - x1) * self.zoom
            rh = abs(y2 - y1) * self.zoom
            
            painter.drawRect(QRectF(rx, ry, rw, rh))

        # 미니맵 그리기
        self.draw_minimap(painter)

        # 클래스 프리뷰 그리기 (우상단)
        self.draw_class_preview(painter)

        # 현재 클래스 오버레이 (좌상단)
        cls = self.current_class
        cls_color = get_color_for_class(cls)
        cls_label = f"Class {cls}"
        if cls < len(self.class_names):
            cls_label += f": {self.class_names[cls]}"
        overlay_rect = QRectF(8, 8, 220, 32)
        painter.fillRect(overlay_rect, QColor(0, 0, 0, 160))
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(overlay_rect)
        font = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        # 색상 동그라미
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(cls_color))
        painter.drawEllipse(QPointF(26, 24), 8, 8)
        painter.setBrush(Qt.NoBrush)
        # 텍스트 (흰색 고정)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(overlay_rect.adjusted(42, 0, 0, 0), Qt.AlignVCenter, cls_label)

        painter.end()
        self.label.setPixmap(display_pixmap)

    def resizeEvent(self, event):
        self.reset_zoom()

    def wheelEvent(self, event):
        delta_y = event.angleDelta().y()
        delta_x = event.angleDelta().x()

        if delta_x != 0:
            # 가로 스크롤: 커서가 박스 위면 해당 박스 클래스 변경, 아니면 현재 클래스 변경
            label_pos = self.label.mapFrom(self, event.pos())
            box_index = self.get_box_index_at_label_pos(label_pos)

            if box_index is not None:
                cls, _, _, _, _ = self.boxes[box_index]
                if delta_x > 0:
                    new_class = min(cls + 1, self.class_count - 1)
                else:
                    new_class = max(cls - 1, 0)
                self.set_box_class(box_index, new_class)
                self.class_slider.setValue(new_class)
            else:
                if delta_x > 0:
                    new_class = min(self.current_class + 1, self.class_count - 1)
                else:
                    new_class = max(self.current_class - 1, 0)
                self.class_slider.setValue(new_class)
        elif delta_y != 0:
            # 세로 스크롤: 확대/축소
            if delta_y > 0:
                new_zoom = min(self.zoom * 1.1, 20.0)
            else:
                new_zoom = max(self.zoom * 0.9, 0.1)

            label_pos_qpoint = self.label.mapFrom(self, event.pos())
            old_pos = label_pos_qpoint - self.offset

            ratio_x = old_pos.x() / (self.w * self.zoom)
            ratio_y = old_pos.y() / (self.h * self.zoom)

            self.zoom = new_zoom

            new_pixmap_w = self.w * self.zoom
            new_pixmap_h = self.h * self.zoom

            new_offset_x = label_pos_qpoint.x() - new_pixmap_w * ratio_x
            new_offset_y = label_pos_qpoint.y() - new_pixmap_h * ratio_y

            self.offset = QPoint(int(new_offset_x), int(new_offset_y))
            self.update_display()

    def get_box_index_at_label_pos(self, label_pos):
        pos = label_pos - self.offset
        x = pos.x() / self.zoom
        y = pos.y() / self.zoom

        for i in reversed(range(len(self.boxes))):
            _, x1, y1, x2, y2 = self.boxes[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    def set_box_class(self, box_index, new_class):
        if box_index is None:
            return
        if not (0 <= new_class < self.class_count):
            return

        cls, x1, y1, x2, y2 = self.boxes[box_index]
        if cls == new_class:
            return

        self._push_undo_state()
        self.boxes[box_index] = (new_class, x1, y1, x2, y2)
        self.modified_boxes.add(box_index)
        self.update_display()

    def get_resize_handle_at(self, pos):
        img_x, img_y = (pos.x() - self.offset.x()) / self.zoom, (pos.y() - self.offset.y()) / self.zoom
        radius = self.HANDLE_SIZE / self.zoom 

        for i in reversed(range(len(self.boxes))):
            _, x1, y1, x2, y2 = self.boxes[i]
            if abs(img_x - x1) < radius and abs(img_y - y1) < radius: return i, 'TL'
            if abs(img_x - x2) < radius and abs(img_y - y1) < radius: return i, 'TR'
            if abs(img_x - x1) < radius and abs(img_y - y2) < radius: return i, 'BL'
            if abs(img_x - x2) < radius and abs(img_y - y2) < radius: return i, 'BR'
        return None, None

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
        
        if event.button() == Qt.LeftButton:
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
        raw_x = pos.x() / self.zoom
        raw_y = pos.y() / self.zoom

        x = max(0.0, min(float(self.w), raw_x))
        y = max(0.0, min(float(self.h), raw_y))

        if self.resizing_box_index is not None:
            cls, x1, y1, x2, y2 = self.boxes[self.resizing_box_index]
            
            if self.resize_handle == 'TL': x1, y1 = x, y
            elif self.resize_handle == 'TR': x2, y1 = x, y
            elif self.resize_handle == 'BL': x1, y2 = x, y
            elif self.resize_handle == 'BR': x2, y2 = x, y
            
            self.boxes[self.resizing_box_index] = (cls, x1, y1, x2, y2)
            self.modified_boxes.add(self.resizing_box_index)  # 수정됨 표시
            self.update_display()
            return

        if self.drawing:
            self.end_point = (x, y) 
            self.update_display()
            return

        if self.dragging_box_index is not None and self.drag_start_pos:
            dx = x - self.drag_start_pos[0]
            dy = y - self.drag_start_pos[1]
            
            cls, x1, y1, x2, y2 = self.boxes[self.dragging_box_index]
            new_box = (cls, x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            self.boxes[self.dragging_box_index] = new_box
            self.modified_boxes.add(self.dragging_box_index)  # 수정됨 표시
            
            self.drag_start_pos = (x, y)
            self.update_display()
            return

        idx, handle = self.get_resize_handle_at(label_pos)
        if idx is not None:
            if handle in ['TL', 'BR']: self.setCursor(Qt.SizeFDiagCursor)
            elif handle in ['TR', 'BL']: self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.panning:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            return

        if self.resizing_box_index is not None:
            prev_state = self._active_edit_snapshot
            cls, x1, y1, x2, y2 = self.boxes[self.resizing_box_index]
            self.boxes[self.resizing_box_index] = (cls, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if prev_state and prev_state != self._serialize_state():
                self.undo_stack.append(prev_state)
                self.redo_stack.clear()
                self._write_stack_file()
            
            self.resizing_box_index = None
            self.resize_handle = None
            self._active_edit_snapshot = None
            self.update_display()
            return

        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            
            label_pos = self.label.mapFrom(self, event.pos())
            pos = label_pos - self.offset
            
            raw_x = pos.x() / self.zoom
            raw_y = pos.y() / self.zoom
            
            curr_x = max(0.0, min(float(self.w), raw_x))
            curr_y = max(0.0, min(float(self.h), raw_y))
            
            x1, y1 = self.start_point
            x2, y2 = curr_x, curr_y
            
            real_x1, real_x2 = min(x1, x2), max(x1, x2)
            real_y1, real_y2 = min(y1, y2), max(y1, y2)

            if abs(real_x2 - real_x1) > 3 and abs(real_y2 - real_y1) > 3:
                self._push_undo_state()
                new_index = len(self.boxes)
                self.boxes.append((self.current_class, real_x1, real_y1, real_x2, real_y2))
                self.modified_boxes.add(new_index)  # 새로 그린 박스도 수정됨으로 표시
            
            self.update_display()

        elif event.button() == Qt.RightButton:
            if self._active_edit_snapshot and self._active_edit_snapshot != self._serialize_state():
                self.undo_stack.append(self._active_edit_snapshot)
                self.redo_stack.clear()
                self._write_stack_file()
            self.dragging_box_index = None
            self.drag_start_pos = None
            self._active_edit_snapshot = None

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_S:
            self.save_labels()
        elif key == Qt.Key_Z:
            self.undo_action()
        elif key == Qt.Key_U:
            self.redo_action()
        elif key == Qt.Key_N or key == Qt.Key_Return or key == Qt.Key_Right:
            if self.image_index < len(self.image_list) - 1:
                self.image_slider.setValue(self.image_index + 1)
            else:
                self.save_labels()
                print("마지막 이미지입니다.")

        elif key == Qt.Key_Left:
            if self.image_index > 0:
                self.image_slider.setValue(self.image_index - 1)
            else:
                self.save_labels()
                print("첫 번째 이미지입니다.")

        elif key == Qt.Key_Backspace:
            if self.boxes:
                self._push_undo_state()
                removed_idx = len(self.boxes) - 1
                self.boxes.pop()
                self.modified_boxes.discard(removed_idx)
                self.update_display()
        elif key == 0x60:  # backtick (`)
            self.current_class = 0
            self.class_slider.setValue(0)
        elif Qt.Key_0 <= key <= Qt.Key_9:
            target_class = key - Qt.Key_0
            if target_class < self.class_count:
                self.current_class = target_class
                self.class_slider.setValue(self.current_class)

                cursor_pos = self.mapFromGlobal(QCursor.pos())
                label_pos = self.label.mapFrom(self, cursor_pos)
                box_index = self.get_box_index_at_label_pos(label_pos)
                self.set_box_class(box_index, target_class)
        elif key == 0x44:  # D
            if self.image_index < len(self.image_list) - 1:
                self.image_slider.setValue(self.image_index + 1)
            else:
                self.save_labels()
                print("마지막 이미지입니다.")
        elif key == 0x41:  # A
            if self.image_index > 0:
                self.image_slider.setValue(self.image_index - 1)
            else:
                self.save_labels()
                print("첫 번째 이미지입니다.")
        elif key == Qt.Key_F:
            self.reset_zoom()
        elif key == Qt.Key_M:
            self.toggle_minimap()
        elif key == Qt.Key_P:
            self.toggle_class_preview()
        elif key == Qt.Key_H:
            self.show_guide()
        elif key == Qt.Key_Q or key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_R:
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            label_pos = self.label.mapFrom(self, cursor_pos)
            pos = label_pos - self.offset

            x = pos.x() / self.zoom
            y = pos.y() / self.zoom

            for i in reversed(range(len(self.boxes))):
                _, x1, y1, x2, y2 = self.boxes[i]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._push_undo_state()
                    del self.boxes[i]
                    self.modified_boxes.discard(i)
                    # 인덱스 재조정
                    self.modified_boxes = {idx - 1 if idx > i else idx for idx in self.modified_boxes}
                    self.update_display()
                    break
        elif key == Qt.Key_C:
            if self.boxes:
                self._push_undo_state()
                self.boxes.clear()
                self.modified_boxes.clear()
                self.update_display()

    def save_labels(self):
        h, w = self.h, self.w
        name = os.path.splitext(self.image_list[self.image_index])[0]
        save_path = self.image_folder
        with open(os.path.join(save_path, name + ".txt"), 'w') as f:
            for cls, x1, y1, x2, y2 in self.boxes:
                real_x1, real_x2 = min(x1, x2), max(x1, x2)
                real_y1, real_y2 = min(y1, y2), max(y1, y2)
                
                xc = ((real_x1 + real_x2) / 2) / w
                yc = ((real_y1 + real_y2) / 2) / h
                bw = (real_x2 - real_x1) / w
                bh = (real_y2 - real_y1) / h

                f.write(f"{cls} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n")

        print(f"✅ 저장 완료: {name}.txt")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageMarker()
    win.show()
    sys.exit(app.exec_())
