"""
Login UI Module
================
Professional login dialog with configurable logo and background.
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QWidget, QGraphicsDropShadowEffect, QApplication
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPoint, QTimer
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter, QLinearGradient, QBrush

# ============================================================================
# CONFIGURATION
# ============================================================================

LOGO_PATH = "assets/logo.png"
BG_IMAGE_PATH = "assets/bg.jpg"
VALID_USER = "admin"
VALID_PASS = "123"
APP_TITLE = "GNSS-RO Processing"

# ============================================================================
# STYLES
# ============================================================================

CARD_STYLE = """
QWidget#loginCard {
    background-color: #ffffff;
    border-radius: 16px;
}
"""

INPUT_STYLE = """
QLineEdit {
    background-color: #f5f5f7;
    border: 2px solid transparent;
    border-radius: 12px;
    padding: 0 16px;
    font-size: 14px;
    color: #1d1d1f;
    selection-background-color: #0071e3;
}
QLineEdit:focus {
    border: 2px solid #0071e3;
    background-color: #ffffff;
}
QLineEdit:hover:!focus {
    background-color: #ebebed;
}
"""

BUTTON_STYLE = """
QPushButton {
    background-color: #0071e3;
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #0077ed;
}
QPushButton:pressed {
    background-color: #006edb;
}
QPushButton:disabled {
    background-color: #c7c7cc;
}
"""

ERROR_STYLE = """
QLabel {
    color: #ff3b30;
    font-size: 12px;
}
"""

TITLE_STYLE = """
QLabel {
    color: #1d1d1f;
    font-size: 22px;
    font-weight: 600;
}
"""

SUBTITLE_STYLE = """
QLabel {
    color: #86868b;
    font-size: 13px;
}
"""


# ============================================================================
# BACKGROUND WIDGET
# ============================================================================

class GradientBackground(QWidget):
    """Background widget with gradient or image."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_pixmap = None
        if os.path.exists(BG_IMAGE_PATH):
            self.bg_pixmap = QPixmap(BG_IMAGE_PATH)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.bg_pixmap and not self.bg_pixmap.isNull():
            scaled = self.bg_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            
            # Subtle overlay for readability
            painter.fillRect(self.rect(), QColor(255, 255, 255, 40))
        else:
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0.0, QColor("#f5f5f7"))
            gradient.setColorAt(1.0, QColor("#e8e8ed"))
            painter.fillRect(self.rect(), QBrush(gradient))


# ============================================================================
# LOGIN DIALOG
# ============================================================================

class LoginDialog(QDialog):
    """Professional login dialog with animation and validation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(480, 640)
        self._drag_pos = None
        self._setup_ui()
        self._center_on_screen()
    
    def _setup_ui(self):
        # Background
        self.background = GradientBackground(self)
        self.background.setGeometry(0, 0, 480, 640)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Center container
        center_layout = QHBoxLayout()
        center_layout.addStretch()
        
        # Login card
        self.card = QWidget()
        self.card.setObjectName("loginCard")
        self.card.setFixedWidth(360)
        self.card.setStyleSheet(CARD_STYLE)
        
        # Card shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.card.setGraphicsEffect(shadow)
        
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(32, 40, 32, 40)
        card_layout.setSpacing(0)
        
        # Logo
        self.logo_label = QLabel()
        self.logo_label.setFixedHeight(80)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if os.path.exists(LOGO_PATH):
            pixmap = QPixmap(LOGO_PATH)
            scaled = pixmap.scaledToHeight(72, Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(scaled)
        else:
            self.logo_label.setText("🛰")
            self.logo_label.setStyleSheet("font-size: 48px;")
        
        card_layout.addWidget(self.logo_label)
        card_layout.addSpacing(16)
        
        # Title
        title = QLabel(APP_TITLE)
        title.setStyleSheet(TITLE_STYLE)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Sign in to continue")
        subtitle.setStyleSheet(SUBTITLE_STYLE)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(32)
        
        # Username field
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.username_input.setFixedHeight(48)
        self.username_input.setStyleSheet(INPUT_STYLE)
        card_layout.addWidget(self.username_input)
        card_layout.addSpacing(12)
        
        # Password field
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setFixedHeight(48)
        self.password_input.setStyleSheet(INPUT_STYLE)
        card_layout.addWidget(self.password_input)
        card_layout.addSpacing(8)
        
        # Error label
        self.error_label = QLabel()
        self.error_label.setStyleSheet(ERROR_STYLE)
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setFixedHeight(20)
        card_layout.addWidget(self.error_label)
        card_layout.addSpacing(16)
        
        # Login button
        self.login_btn = QPushButton("Sign In")
        self.login_btn.setFixedHeight(48)
        self.login_btn.setStyleSheet(BUTTON_STYLE)
        self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_btn.clicked.connect(self._attempt_login)
        card_layout.addWidget(self.login_btn)
        card_layout.addStretch()
        
        # Close hint
        close_hint = QLabel("Press Esc to exit")
        close_hint.setStyleSheet("color: #c7c7cc; font-size: 11px;")
        close_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addSpacing(24)
        card_layout.addWidget(close_hint)
        
        center_layout.addWidget(self.card)
        center_layout.addStretch()
        
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.addStretch()
        wrapper_layout.addLayout(center_layout)
        wrapper_layout.addStretch()
        
        layout.addWidget(wrapper)
        
        # Enter key triggers login
        self.username_input.returnPressed.connect(self._focus_password)
        self.password_input.returnPressed.connect(self._attempt_login)
    
    def _center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
    
    def _focus_password(self):
        self.password_input.setFocus()
    
    def _attempt_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        
        if username == VALID_USER and password == VALID_PASS:
            self.accept()
        else:
            self.error_label.setText("Invalid username or password")
            self.password_input.clear()
            self.password_input.setFocus()
            self._shake_card()
    
    def _shake_card(self):
        """Subtle shake animation on failed login."""
        anim = QPropertyAnimation(self.card, b"pos", self)
        anim.setDuration(400)
        anim.setEasingCurve(QEasingCurve.Type.OutElastic)
        
        start = self.card.pos()
        anim.setKeyValueAt(0, start)
        anim.setKeyValueAt(0.2, start + QPoint(-8, 0))
        anim.setKeyValueAt(0.4, start + QPoint(8, 0))
        anim.setKeyValueAt(0.6, start + QPoint(-4, 0))
        anim.setKeyValueAt(0.8, start + QPoint(4, 0))
        anim.setKeyValueAt(1, start)
        anim.start()
        
        self._anim = anim  # prevent garbage collection
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
    
    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
    
    @staticmethod
    def authenticate(app: QApplication) -> bool:
        """Static method to run login flow. Returns True if authenticated."""
        dialog = LoginDialog()
        result = dialog.exec()
        return result == QDialog.DialogCode.Accepted
