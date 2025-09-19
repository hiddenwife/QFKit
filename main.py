from gui import launch_gui
from tkinter import TclError


#stock_list = ["ACWI", "SPXL", "AAPL", "^GSPC", "^FTSE", "VWCE.DE"]
print("This code will compare funds, ETFs, trackers etc from Yahoo Finance.")
print("It will plot them and calculate growth probabilities and past returns.\n")

try:
    launch_gui()
    # exit after GUI closes
    exit(0)
except TclError as e:
    print(f"Unable to launch GUI due to Tkinter issue. Check your installation. Issue: {e}")
except ImportError as e:
    print(f"Tkinter is not installed. Please install it to use the GUI. Issue: {e}")
except Exception as e:
    print(f"An unexpected error occurred while launching the GUI: {e}")
