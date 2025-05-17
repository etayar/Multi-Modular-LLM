from pathlib import Path
import re
from typing import List

def list_training_run_dates() -> List[str]:
    """
    Lists only the subdirectory names in 'training_runs/' that match the date format YYYY-MM-DD_HH-MM-SS.
    """
    base_dir = Path(__file__).resolve().parent.parent  # adjust if needed
    training_runs_dir = base_dir / "training_runs"
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

    return sorted([
        p.name for p in training_runs_dir.iterdir()
        if p.is_dir() and date_pattern.match(p.name)
    ])


if __name__ == "__main__":
    dates = list_training_run_dates()
    print("Training run directories:")
    for d in dates:
        print("-", d)
