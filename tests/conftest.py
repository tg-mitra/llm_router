import sys
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = MAIN_DIR / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
