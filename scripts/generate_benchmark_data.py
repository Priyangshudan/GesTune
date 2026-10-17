import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gesturetune.synthetic_data import generate_synthetic_dataset

def main():
    out_path = Path("data/benchmark_landmarks.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_dataset(n_samples_per_gesture=35, no_hand_count=20, seed=42)
    df.to_csv(out_path, index=False)
    print(f"Generated benchmark dataset with {len(df)} records at {out_path}")

if __name__ == "__main__":
    main()
