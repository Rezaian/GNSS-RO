# Qt compatibility layer for PyQt5/PyQt6
# This module allows the app to work with both Qt versions
# 3.5.5: exports QMenu (Recent folders menu); high-DPI scaling on Qt5.

try:
    # Try PyQt6 first (modern systems)
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog,
        QListWidget, QListWidgetItem, QTabWidget, QProgressBar,
        QSplitter, QMessageBox, QDialog, QGraphicsDropShadowEffect,
        QCheckBox, QFormLayout, QDoubleSpinBox, QSpinBox, QScrollArea,
        QToolButton, QSizePolicy, QFrame, QMenu
    )
    from PyQt6.QtCore import (
        Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QSize
    )
    from PyQt6.QtGui import (
        QColor, QFont, QPixmap, QPainter, QLinearGradient, QBrush
    )
    PYQT_VERSION = 6
    
except ImportError:
    # Fall back to PyQt5 (Windows 7 compatibility)
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog,
        QListWidget, QListWidgetItem, QTabWidget, QProgressBar,
        QSplitter, QMessageBox, QDialog, QGraphicsDropShadowEffect,
        QCheckBox, QFormLayout, QDoubleSpinBox, QSpinBox, QScrollArea,
        QToolButton, QSizePolicy, QFrame, QMenu
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QSize
    )
    from PyQt5.QtGui import (
        QColor, QFont, QPixmap, QPainter, QLinearGradient, QBrush
    )
    PYQT_VERSION = 5
    # Qt5 does not scale for high-DPI screens by default (Qt6 does): without this,
    # text and icons are tiny or blurry on 125-200 % Windows display scaling.
    # Must be set before the QApplication is created.
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass


def exec_app(app_or_dialog):
    """
    Cross-compatible exec() for QApplication and QDialog.
    PyQt6 uses exec(), PyQt5 uses exec_()
    """
    if PYQT_VERSION == 6:
        return app_or_dialog.exec()
    else:
        return app_or_dialog.exec_()
