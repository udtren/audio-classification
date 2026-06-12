#!/usr/bin/env python3
"""
Batch music analyzer (CLI) — prompts for input and output folders, then for
every MP3/WAV it finds:
  1. Detects BPM with librosa
  2. Classifies genre with MuQ-MuLan (top-3)
  3. Predicts valence + arousal with music2emo
  4. Writes all tags (bpm, genre1/2/3, valence, arousal)
  5. Moves the tagged file to the output folder
"""

import os
import shutil
import sys
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")

from pathlib import Path

MUSIC_EXTENSIONS = {".mp3", ".wav"}


def find_music_files(folder: str) -> list[Path]:
    root = Path(folder)
    found: list[Path] = []
    for ext in MUSIC_EXTENSIONS:
        found.extend(root.rglob(f"*{ext}"))
    return sorted(found)


def move_to_output(src: Path, output_dir: Path) -> Path:
    """Move src to output_dir, appending a counter on name conflicts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / src.name
    if dst.exists() and dst.resolve() != src.resolve():
        stem, suffix = src.stem, src.suffix
        n = 1
        while dst.exists():
            dst = output_dir / f"{stem} ({n}){suffix}"
            n += 1
    shutil.move(str(src), dst)
    return dst


def main():
    folder = input("Input folder : ").strip().strip('"')
    if not Path(folder).is_dir():
        print(f"Not a directory: {folder}")
        sys.exit(1)

    output = input("Output folder: ").strip().strip('"')
    output_dir = Path(output)

    files = find_music_files(folder)
    if not files:
        print("No music files found.")
        sys.exit(0)

    print(f"\nFound {len(files)} file(s)  →  output: {output_dir}\n")

    import torch
    from muq import MuQMuLan
    from analyzers.muq_analyze import MULAN_MODEL_ID, load_segments, classify_genre
    from analyzers.bpm_analyze import detect_bpm
    from analyzers.emo_analyze import load_model as load_emo, analyze_emotion
    from analyzers.tags import write_tags, update_tags

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading MuQ-MuLan on {device} …")
    mulan = MuQMuLan.from_pretrained(MULAN_MODEL_ID).to(device).eval()
    print("Loading Music2emo …")
    emo_model, emo_dir = load_emo()
    print("All models ready.\n")

    ok = 0
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        try:
            # BPM
            bpm, duration = detect_bpm(str(path))
            print(f"  Duration : {duration:.1f}s  |  BPM: {bpm:.1f}")
            write_tags(str(path), {"bpm": f"{bpm:.1f}"})

            # Genre
            segments, _ = load_segments(str(path))
            genre_results = classify_genre(segments, mulan, device)
            genres = [g for g, _ in genre_results[:5]]
            print(f"  Genres   : {', '.join(genres)}")
            update_tags(str(path), genres)

            # Emotion
            valence, arousal, moods = analyze_emotion(str(path), emo_model, emo_dir)
            print(f"  Valence  : {valence:.2f} / 9")
            print(f"  Arousal  : {arousal:.2f} / 9")
            if moods:
                print(f"  Moods    : {', '.join(moods)}")
            write_tags(str(path), {"valence": f"{valence:.2f}", "arousal": f"{arousal:.2f}"})

            # Move
            dest = move_to_output(path, output_dir)
            print(f"  Moved  → {dest}")
            ok += 1
        except Exception as exc:
            print(f"  Error: {exc}")
        print()

    print(f"Done — {ok}/{len(files)} file(s) tagged and moved to {output_dir}")


if __name__ == "__main__":
    main()
