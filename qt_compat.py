# Qt compatibility layer for PyQt5/PyQt6
# This module allows the app to work with both Qt versions

try:
    # Try PyQt6 first (modern systems)
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog,
        QListWidget, QListWidgetItem, QTabWidget, QProgressBar,
        QSplitter, QMessageBox, QDialog, QGraphicsDropShadowEffect,
        QCheckBox, QFormLayout, QDoubleSpinBox, QSpinBox, QScrollArea,
        QToolButton, QSizePolicy, QFrame
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
        QToolButton, QSizePolicy, QFrame
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QSize
    )
    from PyQt5.QtGui import (
        QColor, QFont, QPixmap, QPainter, QLinearGradient, QBrush
    )
    PYQT_VERSION = 5


def exec_app(app_or_dialog):
    """
    Cross-compatible exec() for QApplication and QDialog.
    PyQt6 uses exec(), PyQt5 uses exec_()
    """
    if PYQT_VERSION == 6:
        return app_or_dialog.exec()
    else:
        return app_or_dialog.exec_()
