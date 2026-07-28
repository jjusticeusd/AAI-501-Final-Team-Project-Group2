"""Fetch the PhiUSIIL Phishing URL Dataset (UCI id 967) and cache it
locally as data/phiusiil.csv so notebooks don't hit the network.

Run once after `uv sync`:

    uv run python code/fetch_data.py
"""

from pathlib import Path

from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "phiusiil.csv"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    dataset = fetch_ucirepo(id=967)
    features = dataset.data.features
    targets = dataset.data.targets
    df = features.join(targets)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} cols to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
