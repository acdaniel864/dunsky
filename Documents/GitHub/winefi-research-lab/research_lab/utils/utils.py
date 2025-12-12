"""
Utility functions for file operations, paths, and notebook helpers.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from PIL import Image

# Ensure code directory is in path
try:
    # Try to import from code package - works if code is in path
    from code.config import BASE_DIR, CODE_DIR, watermark_path

except ImportError:
    # If that fails, try to locate the project root directory
    current_path = Path.cwd()
    
    # If we're in the code directory, go up one level to project root
    if current_path.name == 'code':
        current_path = current_path.parent
    
    # If we're somewhere else, search for the project root
    while current_path != Path('/') and not (current_path / 'code' / 'config.py').exists():
        current_path = current_path.parent
        
    if (current_path / 'code').exists():
        # Found the project root
        sys.path.append(str(current_path))
        from code.config import BASE_DIR, CODE_DIR, watermark_path
    else:
        # If project root not found, make best guess
        current_path = Path.cwd()
        print(f"Warning: Could not import config. Using current directory: {current_path}")
        
        # If in code directory, set BASE_DIR to parent
        if current_path.name == 'code':
            BASE_DIR = current_path.parent
        else:
            BASE_DIR = current_path
            
        CODE_DIR = BASE_DIR / 'code'
        # DATA_DIR = BASE_DIR / 'data'
        IMAGE_DIR = BASE_DIR / 'images'
        
        watermark_candidates = [
            IMAGE_DIR / "watermark600x600.png",
            IMAGE_DIR / "plot_logs/watermark600x600.png",
            Path("/Users/AaranDaniel/Desktop/git_not/quarterly_reports/images/watermark600x600.png")
        ]
        watermark_path = next((path for path in watermark_candidates if path.exists()), None)

def get_project_root():
    """
    Returns the project root directory.
    """
    return BASE_DIR  # BASE_DIR is already the project root


# def get_data_path(filename=''):
#     """
#     Returns the path to the data directory or a file within it.
    
#     Args:
#         filename: Optional filename to append to the data path
    
#     Returns:
#         Path object for the data directory or file
#     """
#     data_dir = DATA_DIR
#     data_dir.mkdir(parents=True, exist_ok=True)
    
#     if filename:
#         return data_dir / filename
#     return data_dir

def get_plot_date():
    """
    Returns the current date formatted for plots.
    
    Returns:
        str: Current date formatted as DD/MM/YY
    """
    return datetime.now().strftime('%d/%m/%y')



def add_watermark_to_plot(ax=None, alpha=0.15):
    """
    Adds a watermark to the current matplotlib plot.
    
    Args:
        ax: Matplotlib axis (uses current axis if None)
        alpha: Opacity of the watermark
    """
    if watermark_path is None or not Path(watermark_path).exists():
        print("Warning: Watermark image not found.")
        return
        
    if ax is None:
        ax = plt.gca()
        
    try:
        img = Image.open(watermark_path)
        # Create a new axes for the watermark with automatic positioning
        watermark_ax = ax.inset_axes([0.8, 0.05, 0.15, 0.15], transform=ax.transAxes)
        # Turn off axis for the watermark
        watermark_ax.axis('off')
        # Display the watermark
        watermark_ax.imshow(img, alpha=alpha)
    except Exception as e:
        print(f"Could not add watermark: {e}")


def load_data_for_quarter(date_str=None):
    """
    Load standard dataframes for the specified quarter end date.
    
    Args:
        date_str: Quarter end date string (YYYY-MM-DD). If None, uses current quarter.
    
    Returns:
        tuple of (df_info, df_time) DataFrames
    """
    if date_str is None:
        date_str = get_quarter_end_date().strftime('%Y-%m-%d')
        
    data_dir = get_data_path()
    
    # Try to find the data files
    info_pattern = f"*{date_str}*df_info*.csv"
    time_pattern = f"*{date_str}*df_time*.csv"
    
    info_files = list(data_dir.glob(info_pattern))
    time_files = list(data_dir.glob(time_pattern))
    
    if not info_files:
        raise FileNotFoundError(f"No df_info file found matching pattern {info_pattern}")
    if not time_files:
        raise FileNotFoundError(f"No df_time file found matching pattern {time_pattern}")
    
    # Load the data
    df_info = pd.read_csv(info_files[0], dtype={'lwin11': str, 'lwin18': str})
    df_time = pd.read_csv(time_files[0], parse_dates=['date'])
    df_time.set_index(['date'], inplace=True)
    
    return df_info, df_time 

def quarter_end_date():
    """
    Returns the last day of the current month or last month as date object.
    If we're at the end of a quarter (months 3, 6, 9, 12), we use the current month.
    Args:
    Returns:
        date: The last day of the month
    """
    today = datetime.today()

    current_month = datetime.now().month

    if current_month % 3 == 0:
        use_current_month = True
    else:
        use_current_month = False
    
    if not use_current_month:
        # If we want last month, subtract one month from today
        if today.month == 1:
            # If January, go to December of previous year
            month = 12
            year = today.year - 1
        else:
            month = today.month - 1
            year = today.year
    else:
        # Use current month
        month = today.month
        year = today.year
    
    # First day of the next month
    next_month = month % 12 + 1
    next_month_year = year + (month // 12)
    first_day_next_month = date(next_month_year, next_month, 1)
    # Subtract one day to get the last day of the current/last month
    quarter_end_date = first_day_next_month - timedelta(days=1)
    
    # Check if the resulting month is a quarter end and print warning if not
    if quarter_end_date.month % 3 != 0:
        print(f"Warning: The returned date {quarter_end_date} is not a quarter end (month {quarter_end_date.month} is not divisible by 3)")
    
    return quarter_end_date


test_lwins = ['10286742007', '18076262002', '13169351995', '11020372004', '10140332000', '10996342008']