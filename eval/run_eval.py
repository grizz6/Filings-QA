"""Entry point named in the project plan; the logic lives in src/evaluate.py.

python eval/run_eval.py --all      (same as: python -m src.evaluate --all)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
