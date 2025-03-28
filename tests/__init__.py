import os
import sys
from pathlib import Path


ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
SOURCE_PATH = os.path.join(PROJECT_PATH, 'src')
sys.path.append(SOURCE_PATH)