"""Repository paths shared by the reproduction modules."""

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = CODE_DIR.parent
DATA_DIR = ROOT_DIR / "Data"
OUTPUTS_DIR = ROOT_DIR / "Outputs"
