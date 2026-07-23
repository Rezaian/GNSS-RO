"""
Login UI Module
================
Professional login dialog with configurable logo and background.
"""

import os
import sys

from qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QWidget, QGraphicsDropShadowEffect, QApplication,
    Qt, QPropertyAnimation, QEasingCurve, QPoint, QTimer,
    QPixmap, QFont, QColor, QPainter, QLinearGradient, QBrush,
    exec_app
)
# ============================================================================
# RESOURCE PATH HELPER (PyInstaller compatible)
# ============================================================================

def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and PyInstaller bundle."""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller extracts to temp folder
        base_path = sys._MEIPASS
    else:
        # Development mode
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ============================================================================
# CONFIGURATION
# ============================================================================

LOGO_PATH = resource_path("assets/logo.png")
BG_IMAGE_PATH = resource_path("assets/bg.jpg")
VALID_USER = "admin"
VALID_PASS = "123"
APP_TITLE = "GNSS-RO Processing"

# ============================================================================
# v3.4.7 — LOGIN SCALING
# ============================================================================
# The login card is laid out from a single pair of scale factors so the size
# can be re-tuned in one place.  1.0 reproduces the v3.4.6 appearance exactly.
#
#   LOGIN_FONT_SCALE = 1.5  -> all login text +50%   (v3.4.7 request)
#   LOGIN_LOGO_SCALE = 2.0  -> logo 2x                (v3.4.7 request)
#
# Box heights are scaled at half the font rate so the card stays compact
# enough for a 1366x768 laptop; _compute_scale() shrinks both factors further
# if the screen is genuinely too short.
# ============================================================================

LOGIN_FONT_SCALE = 1.5
LOGIN_LOGO_SCALE = 2.0

# Card height (px) required at the nominal 1.5 / 2.0 scale.
_CARD_HEIGHT_AT_FULL_SCALE = 660


def _compute_scale():
    """Return (font_scale, logo_scale), reduced on short screens.

    On a 1366x768 laptop the usable height is ~728 px after the taskbar, which
    still fits the full-scale card.  Anything shorter (e.g. 1280x720, or a
    768 px panel with a large taskbar) scales down proportionally instead of
    clipping the Sign In button off the bottom.
    """
    font_s, logo_s = LOGIN_FONT_SCALE, LOGIN_LOGO_SCALE
    try:
        avail = QApplication.primaryScreen().availableGeometry().height()
    except Exception:
        avail = 768
    if avail < _CARD_HEIGHT_AT_FULL_SCALE + 40:
        k = max(0.55, (avail - 40) / float(_CARD_HEIGHT_AT_FULL_SCALE))
        font_s = 1.0 + (font_s - 1.0) * k
        logo_s = 1.0 + (logo_s - 1.0) * k
    return font_s, logo_s


def _px(base, scale):
    """Scale a pixel value and round to int."""
    return max(1, int(round(base * scale)))


# ============================================================================
# STYLES  (base values = v3.4.6 sizes; multiplied by the font scale)
# ============================================================================

CARD_STYLE = """
QWidget#loginCard {
    background-color: #ffffff;
    border-radius: 16px;
}
"""


def input_style(fs):
    return """
QLineEdit {
    background-color: #f5f5f7;
    border: 2px solid transparent;
    border-radius: 12px;
    padding: 0 16px;
    font-size: %dpx;
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
""" % _px(14, fs)


def button_style(fs):
    return """
QPushButton {
    background-color: #0071e3;
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-size: %dpx;
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
""" % _px(15, fs)


def error_style(fs):
    return "QLabel { color: #ff3b30; font-size: %dpx; }" % _px(12, fs)


def title_style(fs):
    return ("QLabel { color: #1d1d1f; font-size: %dpx; font-weight: 600; }"
            % _px(22, fs))


def subtitle_style(fs):
    return "QLabel { color: #86868b; font-size: %dpx; }" % _px(13, fs)


def hint_style(fs):
    return "QLabel { color: #c7c7cc; font-size: %dpx; }" % _px(11, fs)


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
        # v3.4.7 — resolve the scale factors before any widget is built.
        self.fs, self.ls = _compute_scale()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Card width follows the font scale; height is driven by the content
        # and clamped to the available screen height.
        self._card_w = _px(360, self.fs)
        card_h = _px(_CARD_HEIGHT_AT_FULL_SCALE, self.fs / LOGIN_FONT_SCALE)
        dlg_w = self._card_w + _px(120, self.fs)
        dlg_h = card_h + 40
        try:
            avail = QApplication.primaryScreen().availableGeometry()
            dlg_w = min(dlg_w, avail.width() - 40)
            dlg_h = min(dlg_h, avail.height() - 20)
        except Exception:
            pass

        self.setFixedSize(dlg_w, dlg_h)
        self._drag_pos = None
        self._setup_ui()
        self._center_on_screen()

    def _setup_ui(self):
        fs, ls = self.fs, self.ls

        # Box heights grow at half the font rate so the card stays compact.
        box_scale = 1.0 + (fs - 1.0) * 0.5

        # Background
        self.background = GradientBackground(self)
        self.background.setGeometry(0, 0, self.width(), self.height())

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Center container
        center_layout = QHBoxLayout()
        center_layout.addStretch()

        # Login card
        self.card = QWidget()
        self.card.setObjectName("loginCard")
        self.card.setFixedWidth(self._card_w)
        self.card.setStyleSheet(CARD_STYLE)

        # Card shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(_px(32, fs), _px(32, box_scale),
                                       _px(32, fs), _px(32, box_scale))
        card_layout.setSpacing(0)

        # Logo — v3.4.7: 2x the v3.4.6 size (72 px -> 144 px tall)
        logo_h = _px(72, ls)
        self.logo_label = QLabel()
        self.logo_label.setFixedHeight(logo_h + 8)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if os.path.exists(LOGO_PATH):
            pixmap = QPixmap(LOGO_PATH)
            scaled = pixmap.scaledToHeight(
                logo_h, Qt.TransformationMode.SmoothTransformation)
            # Guard: at 2x a wide logo could exceed the card's inner width.
            max_logo_w = self._card_w - 2 * _px(32, fs)
            if not scaled.isNull() and scaled.width() > max_logo_w:
                scaled = pixmap.scaledToWidth(
                    max_logo_w, Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(scaled)
        else:
            self.logo_label.setText("\U0001F6F0")
            self.logo_label.setStyleSheet("font-size: %dpx;" % _px(48, ls))

        card_layout.addWidget(self.logo_label)
        card_layout.addSpacing(_px(16, box_scale))

        # Title
        title = QLabel(APP_TITLE)
        title.setStyleSheet(title_style(fs))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        card_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Sign in to continue")
        subtitle.setStyleSheet(subtitle_style(fs))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(_px(24, box_scale))

        field_h = _px(48, box_scale)

        # Username field
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.username_input.setFixedHeight(field_h)
        self.username_input.setStyleSheet(input_style(fs))
        card_layout.addWidget(self.username_input)
        card_layout.addSpacing(_px(12, box_scale))

        # Password field
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setFixedHeight(field_h)
        self.password_input.setStyleSheet(input_style(fs))
        card_layout.addWidget(self.password_input)
        card_layout.addSpacing(_px(8, box_scale))

        # Error label
        self.error_label = QLabel()
        self.error_label.setStyleSheet(error_style(fs))
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setFixedHeight(_px(20, box_scale))
        card_layout.addWidget(self.error_label)
        card_layout.addSpacing(_px(16, box_scale))

        # Login button
        self.login_btn = QPushButton("Sign In")
        self.login_btn.setFixedHeight(field_h)
        self.login_btn.setStyleSheet(button_style(fs))
        self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_btn.clicked.connect(self._attempt_login)
        card_layout.addWidget(self.login_btn)
        card_layout.addStretch()

        # Close hint
        close_hint = QLabel("Press Esc to exit")
        close_hint.setStyleSheet(hint_style(fs))
        close_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addSpacing(_px(14, box_scale))
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
    
    @staticmethod
    def _global_pos(event):
        """Return global position as QPoint for PyQt5 and PyQt6.
        PyQt6: event.globalPosition().toPoint()
        PyQt5: event.globalPos()
        """
        try:
            return event.globalPosition().toPoint()
        except AttributeError:
            return event.globalPos()

    def mousePressEvent(self, event):
        try:
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag_pos = self._global_pos(event) - self.frameGeometry().topLeft()
        except Exception:
            self._drag_pos = None

    def mouseMoveEvent(self, event):
        try:
            if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
                self.move(self._global_pos(event) - self._drag_pos)
        except Exception:
            pass

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
    
    @staticmethod
    def authenticate(app: QApplication) -> bool:
        """Static method to run login flow. Returns True if authenticated."""
        dialog = LoginDialog()
        result = exec_app(dialog)
        return result == QDialog.DialogCode.Accepted
