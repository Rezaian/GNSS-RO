# Building the Windows app (GNSS-RO 3.5.5)

Goal: one folder that runs on a freshly installed Windows 7 SP1, 8.1, 10 or 11 (64-bit).

## Build machine
1. Windows 7 SP1 (or 10) 64-bit. Building on the oldest target is safest.
2. Install Python 3.8.10 64-bit (python.org; the last version with a Windows 7 installer).
   Bare Windows 7 needs updates KB2533623 and KB2999226 (Universal C Runtime) first.
3. In a fresh virtual environment:
       py -3.8 -m venv venv
       venv\Scripts\activate
       pip install -r requirements-win7.txt
4. Put qt_compat.py (3.5.5) next to gnss_ro_gui.py. It uses PyQt5 when installed;
   Qt 6 (PySide6 / PyQt6) does not run on Windows 7 or Windows 10 before 1809.

## Build
    pyinstaller --noconfirm --windowed --name GNSS-RO ^
        --add-data "assets;assets" ^
        --hidden-import netCDF4 --hidden-import cftime ^
        gnss_ro_gui.py

Use the default one-folder build (not --onefile): it starts faster, needs no
unpacking to %TEMP%, and is far less likely to be blocked by antivirus.

## Ship
Zip `dist\GNSS-RO\` and run `GNSS-RO.exe` from the unzipped folder.
Test on a clean Windows 7 VM before release. Each run's `log.txt` now records the
app version, Python, Windows, NumPy and pandas versions, so bug reports show the setup.
