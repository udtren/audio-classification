#!/usr/bin/env python3
"""
Audio genre analyzer using MIT Audio Spectrogram Transformer (AST).
Model: MIT/ast-finetuned-audioset-10-10-0.4593

Splits audio into 10-second segments, averages predictions, then writes
the top-3 music genre labels as a TCON tag to the file.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*mel filter.*zero values.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification  # noqa: E402

MODEL_ID     = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE  = 16000
SEGMENT_SECS = 10
MAX_SEGMENTS = 6
TOP_GENRES   = 3    # how many genre labels to write into the TCON tag

# AudioSet labels that count as music genres
_GENRE_KEYWORDS = {
    "pop", "rock", "hip hop", "electronic", "classical", "jazz",
    "country", "reggae", "blues", "rhythm and blues", "techno",
    "dance music", "heavy metal", "punk", "soul", "folk", "ambient",
    "gospel", "disco", "funk", "opera", "rap", "house music",
    "dubstep", "drum and bass", "trance", "grunge", "ska",
    "latin", "bossanova", "samba", "flamenco", "indie", "alternative",
    "r&b", "new-age", "world music",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _is_genre(label: str) -> bool:
    low = label.lower()
    return any(kw in low for kw in _GENRE_KEYWORDS)


def _has_music(label: str) -> bool:
    return "music" in label.lower()


def _segments(y: np.ndarray) -> list[np.ndarray]:
    seg_len = SEGMENT_SECS * SAMPLE_RATE
    total   = len(y)
    if total <= seg_len:
        return [y]
    n = min(MAX_SEGMENTS, total // seg_len)
    starts = np.linspace(0, total - seg_len, n, dtype=int)
    return [y[s : s + seg_len] for s in starts]


def _pick_top_genres(
    results: list[tuple[str, float]],
    n: int = TOP_GENRES,
) -> list[tuple[str, float]]:
    """
    Return the top-n genre labels from the full ranked list.
    Priority order: genre keyword match > any "music" label > overall top-1.
    """
    seen: set[str] = set()
    picked: list[tuple[str, float]] = []

    # Pass 1 — genre keyword matches
    for lb, sc in results:
        if _is_genre(lb) and lb not in seen:
            picked.append((lb, sc))
            seen.add(lb)
            if len(picked) == n:
                return picked

    # Pass 2 — any label containing "music"
    for lb, sc in results:
        if _has_music(lb) and lb not in seen:
            picked.append((lb, sc))
            seen.add(lb)
            if len(picked) == n:
                return picked

    # Pass 3 — overall top results
    for lb, sc in results:
        if lb not in seen:
            picked.append((lb, sc))
            seen.add(lb)
            if len(picked) == n:
                return picked

    return picked


# ── core ───────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading model  {MODEL_ID} ...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model     = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"  Device: {device}\n")
    return extractor, model, device


def classify(audio_path: str, extractor, model, device) -> list[tuple[str, float]]:
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / SAMPLE_RATE
    print(f"  Duration : {duration:.1f} s")

    segs = _segments(y)
    print(f"  Segments : {len(segs)} x {SEGMENT_SECS}s")

    logit_sum = None
    for seg in segs:
        inputs = extractor(
            seg,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[0].cpu()
        logit_sum = logits if logit_sum is None else logit_sum + logits

    probs = torch.softmax(logit_sum / len(segs), dim=-1)
    top   = torch.topk(probs, k=len(probs))
    return [
        (model.config.id2label[i.item()], s.item())
        for s, i in zip(top.values, top.indices)
    ]


from tags import update_tags  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classify music genre with MIT AST and tag the file (top-3 genres)"
    )
    ap.add_argument("files", nargs="+", metavar="FILE",
                    help="MP3 or WAV file(s) to analyse")
    ap.add_argument("--top", type=int, default=15,
                    help="Top-N predictions to display (default: 15)")
    ap.add_argument("--no-tag", action="store_true",
                    help="Skip writing the TCON genre tag")
    args = ap.parse_args()

    extractor, model, device = load_model()

    for path in args.files:
        if not Path(path).exists():
            print(f"Not found: {path}", file=sys.stderr)
            continue

        print(f"{'='*60}")
        print(f"File: {path}")

        results = classify(path, extractor, model, device)
        top_genres = _pick_top_genres(results, n=TOP_GENRES)
        top_genre_labels = {lb for lb, _ in top_genres}

        # Display table — mark genre picks with *
        print(f"\n  {'#':<4} {'Label':<38} {'Conf':>6}  Bar")
        print(f"  {'-'*70}")
        for rank, (label, score) in enumerate(results[: args.top], 1):
            bar  = "#" * int(score * 30)
            flag = " *" if label in top_genre_labels else "  "
            print(f"  {rank:<4} {label:<38} {score:>6.3f}  {bar}{flag}")

        print(f"\n  Top-{TOP_GENRES} genres for tag:")
        for i, (label, score) in enumerate(top_genres, 1):
            print(f"    {i}. {label}  ({score:.1%})")

        if not args.no_tag:
            update_tags(path, [lb for lb, _ in top_genres])

        print()


if __name__ == "__main__":
    main()
