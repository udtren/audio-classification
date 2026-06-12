#!/usr/bin/env python3
"""
BPM analyzer using librosa beat tracking.
Writes detected tempo as a custom TXXX tag named "bpm".
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
from tags import write_tags  # noqa: E402

SAMPLE_RATE = 22050  # librosa default; sufficient for beat tracking


def detect_bpm(audio_path: str) -> tuple[float, float]:
    """Return (bpm, duration_seconds)."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    return bpm, duration


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect BPM with librosa and write a 'bpm' TXXX tag"
    )
    ap.add_argument("files", nargs="+", metavar="FILE", help="MP3 or WAV file(s)")
    ap.add_argument("--no-tag", action="store_true", help="Skip writing the bpm tag")
    args = ap.parse_args()

    for path in args.files:
        if not Path(path).exists():
            print(f"Not found: {path}", file=sys.stderr)
            continue

        print(f"{'='*60}")
        print(f"File: {path}")

        bpm, duration = detect_bpm(path)
        print(f"  Duration : {duration:.1f} s")
        print(f"  BPM      : {bpm:.1f}")

        if not args.no_tag:
            write_tags(path, {"bpm": f"{bpm:.1f}"})

        print()


if __name__ == "__main__":
    main()
