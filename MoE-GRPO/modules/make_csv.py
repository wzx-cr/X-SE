import argparse
import csv
import random
from pathlib import Path
from collections import Counter


REQUIRED_COLUMNS = ["split", "speaker", "utt_id", "noisy_path", "clean_path"]


def get_speaker_id(wav_path: Path) -> str:
    """
    Example:
        p264_092.wav -> p264
    For datasets without this naming rule, it still gives a stable rough speaker field.
    """
    return wav_path.stem.split("_")[0]


def parse_dataset_arg(item: str):
    """
    Format:
        name=/path/to/noisy,/path/to/clean

    Example:
        wham=path/to/noisy_wavs,path/to/clean_wavs
    """
    if "=" not in item:
        raise ValueError(f"Invalid dataset spec: {item}. Expected name=noisy_dir,clean_dir")

    name, paths = item.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid dataset name in: {item}")

    parts = [p.strip() for p in paths.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid dataset spec: {item}. Expected name=noisy_dir,clean_dir")

    noisy_dir = Path(parts[0]).expanduser()
    clean_dir = Path(parts[1]).expanduser()

    if not noisy_dir.is_dir():
        raise FileNotFoundError(f"Noisy dir does not exist: {noisy_dir}")
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Clean dir does not exist: {clean_dir}")

    return name, noisy_dir, clean_dir


def collect_pairs(dataset_name: str, noisy_dir: Path, clean_dir: Path):
    rows = []
    missing = 0

    noisy_wavs = sorted(noisy_dir.glob("*.wav"))
    if len(noisy_wavs) == 0:
        raise RuntimeError(f"No wav files found in noisy dir: {noisy_dir}")

    for noisy_wav in noisy_wavs:
        clean_wav = clean_dir / noisy_wav.name
        if not clean_wav.exists():
            missing += 1
            continue

        speaker = f"{dataset_name}_{get_speaker_id(noisy_wav)}"
        utt_id = f"{dataset_name}_{noisy_wav.stem}"

        rows.append(
            {
                "split": "",  # filled later
                "speaker": speaker,
                "utt_id": utt_id,
                "noisy_path": str(noisy_wav.resolve()),
                "clean_path": str(clean_wav.resolve()),
            }
        )

    if missing > 0:
        print(f"[warn] {dataset_name}: {missing} noisy wavs skipped because clean file was not found")

    if len(rows) == 0:
        raise RuntimeError(f"No matched noisy/clean pairs found for dataset: {dataset_name}")

    return rows


def split_train_valid(rows, train_ratio: float, valid_ratio: float, seed: int):
    ratio_sum = train_ratio + valid_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + valid_ratio must be 1.0, got {ratio_sum}")

    rows = rows[:]
    random.Random(seed).shuffle(rows)

    n_total = len(rows)
    n_train = round(n_total * train_ratio)

    train_rows = rows[:n_train]
    valid_rows = rows[n_train:]

    for row in train_rows:
        row["split"] = "train"
    for row in valid_rows:
        row["split"] = "valid"

    return train_rows + valid_rows


def write_csv(out_path: Path, rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Mix multiple noisy/clean wav datasets and split them into train/valid CSV rows."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="One or more datasets in format: name=noisy_dir,clean_dir",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="mixed_split.csv")

    args = parser.parse_args()

    all_rows = []
    dataset_counts = Counter()

    for item in args.datasets:
        dataset_name, noisy_dir, clean_dir = parse_dataset_arg(item)
        rows = collect_pairs(dataset_name, noisy_dir, clean_dir)
        all_rows.extend(rows)
        dataset_counts[dataset_name] += len(rows)
        print(f"[ok] {dataset_name}: {len(rows)} matched pairs")

    mixed_rows = split_train_valid(
        all_rows,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    out_path = Path(args.out)
    write_csv(out_path, mixed_rows)

    split_counts = Counter(row["split"] for row in mixed_rows)
    print()
    print(f"Saved CSV to: {out_path.resolve()}")
    print(f"Total rows: {len(mixed_rows)}")
    print(f"Train rows: {split_counts['train']}")
    print(f"Valid rows: {split_counts['valid']}")
    print()
    print("Dataset rows:")
    for name, count in sorted(dataset_counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
