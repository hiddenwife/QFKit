"""
About:
  Modular Python toolkit for ETF, fund, and portfolio analytics. 
  Integrates market data (yfinance) with time-series analysis, 
  Monte Carlo (GBM) simulations, risk metrics, and Bayesian forecasting, 
  all accessible via an interactive GUI.

Developer: Christopher Andrews
GitHub: https://github.com/hiddenwife/QFKit

Read LICENCE for use of any of this code.

"""
import os
import platform
import multiprocessing as mp
import matplotlib
import subprocess, sys
import shutil
from subprocess import check_output

# Check the operating system
current_os = platform.system()

def force_x11_unless_impossible():
    # quick env checks
    has_display = bool(os.environ.get("DISPLAY"))
    has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))

    # If there's already an explicit platform, respect it
    if os.environ.get("QT_QPA_PLATFORM"):
        print(" - QT_QPA_PLATFORM already set; leaving display platform unchanged.")
        return

    # If no X11 display present, can't force xcb
    if not has_display:
        # allow wayland if present, otherwise leave unset
        if has_wayland:
            os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
            print(" - No X11 DISPLAY; keeping Wayland (QT_QPA_PLATFORM='wayland').")
        return

    # Try to force xcb; if plugin missing or blocked, detect by launching a tiny Qt process
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    try:
        # run a minimal Qt probe that exits quickly to verify plugin is usable
        # Using python -c to avoid importing app modules
        probe_cmd = [sys.executable, "-c",
                     "from PySide6 import QtWidgets; app=QtWidgets.QApplication([]); print('ok')"]
        subprocess.run(probe_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        print(" - X11 enforced (QT_QPA_PLATFORM='xcb').")
    except Exception:
        # xcb failed - fall back to wayland if available, otherwise remove the forced value
        if has_wayland:
            os.environ["QT_QPA_PLATFORM"] = "wayland"
            print(" - X11 probe failed; falling back to Wayland (QT_QPA_PLATFORM='wayland').")
        else:
            os.environ.pop("QT_QPA_PLATFORM", None)
            print(" - X11 probe failed and no Wayland detected; QT_QPA_PLATFORM unset - using Qt defaults.")

def is_clang_binary(bin_path: str) -> bool:
    try:
        out = subprocess.run([bin_path, "--version"], capture_output=True, text=True, check=True)
        return "clang" in out.stdout.lower() or "clang" in out.stderr.lower()
    except Exception:
        return False

def is_g___binary(bin_path: str) -> bool:
    try:
        out = subprocess.run([bin_path, "--version"], capture_output=True, text=True, check=True)
        return "free software foundation" in out.stdout.lower() or "free software foundation" in out.stderr.lower()
    except Exception:
        return False

# Apply settings based on the OS - edit these as you require.
if current_os == "Linux":
    print("\nAppplying Linux based settings:")

    # Try safer Qt settings for Linux
    # os.environ.setdefault("QT_QUICK_BACKEND", "software")      # avoid GPU QtQuick font/OpenGL issues
    # os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")    # avoid xcb GL integration bugs
    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")  # sometimes needed on Linux packaging

    # Force Matplotlib to use QtAgg and a known font family
    matplotlib.use("QtAgg") 
    #matplotlib.rcParams["font.family"] = "DejaVu Sans"
    
    # Forcing x11 if available.
    force_x11_unless_impossible()

    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx:
        if is_g___binary(cxx):
            print(f" - Using C++ compiler: {cxx} (g++) — OK for pytensor.")
        else:
            print(f" - Not using g++ compiler: {cxx} - pytensor may fail")
    else:
        print(" - No C++ compiler found in PATH. pytensor may fail; install g++ compiler.")


elif current_os == "Windows":
    print("Applying Windows based settings:")
    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx:
        if is_g___binary(cxx):
            print(f" - Using C++ compiler: {cxx} (g++) — OK for pytensor.")
        else:
            print(f" - Not using g++ compiler: {cxx} - pytensor may fail")
    else:
        print(" - No C++ compiler found in PATH. pytensor may fail; install g++ compiler.")
    pass

elif current_os == "Darwin":
    # macOS stuff here
    print("Applying macOS based settings:")

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx:
        if is_clang_binary(cxx):
            print(f" - Using C++ compiler: {cxx} (clang) — OK for pytensor.")
        else:
            print(f" - Using C++ compiler: {cxx} — not clang. pytensor may not work as well; consider installing/using clang (Apple Clang) or setting CXX to clang++.")
    else:
        print(" - No C++ compiler found in PATH. pytensor may fail; install Xcode command line tools (clang).")

else:
    print("Unknown OS - no settings applied.\n")

from gui import launch_gui 


if __name__ == "__main__":
    mp.set_start_method('spawn') # Recommended!!! Do not edit otherwise GUI might crash.
    print("\nThis code will compare funds, ETFs, trackers etc from Yahoo Finance.\n")
    print("Launching Financial Analysis & Simulation Toolkit...")
    launch_gui()