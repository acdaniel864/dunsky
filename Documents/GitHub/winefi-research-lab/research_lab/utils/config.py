"""
Configuration module for all notebook projects.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': os.getenv('DB_PORT')
}

# Create SQLAlchemy database URL
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Price configuration
LABEL_MEAN_PRICE_MIN = 720

VINTAGE_MIN = 1975

# Region to index mapping
REGION_INDEX_DICT = {
    'burgundy': 'Burgundy 150',
    'champagne': 'Champagne 50',
    'tuscany': 'Italy 100',
    'piedmont': 'Italy 100',
    'california': 'California 50',
    'bordeaux': 'Liv-ex Bordeaux 500',
    'rhone': 'Rhone 100',
    'port': 'Port 50',
    'row': 'Rest of the World 60'
}

# ROW (Rest of World) regions
ROW_REGIONS = [
    'Rhone', 'South Australia', 'Mosel', 'Loire', 'Rioja',
    'Castilla y Leon', 'Alsace', 'Rheinhessen', 'Mendoza',
    'Central Otago', 'Coastal Region'
]


COLOURS = {
    "main_purple": "#9437ff",
    "cool_purple": "#7c60f9",
    "mantis": "#83D483",
    "sunglow": "#FFD166",
    "coral": "#F78C6B",
    "blue": "#4D87D0",
    "red": "#EF476F",
    "emerald": "#06D6A0",
    "pink_purple": "#C23FB7",
    "slate": "#4A4A68"
}
