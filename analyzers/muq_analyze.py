#!/usr/bin/env python3
"""
Music analyzer using OpenMuQ models:
  - MuQ-large-msd-iter : deep audio feature extraction (SSL, 300M params)
  - MuQ-MuLan-large    : zero-shot genre/style matching via music-text similarity

Workflow:
  1. Load audio at 24 kHz (MuQ native sample rate)
  2. Split into 30-second segments, run MuQ for embeddings
  3. Use MuQ-MuLan to rank genre/style prompts by cosine similarity
  4. Write top-3 genres as TCON tag

MuQ alone is a feature extractor — it has no text decoder.
MuQ-MuLan brings audio and text into a shared embedding space,
enabling zero-shot genre classification.
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
warnings.filterwarnings("ignore", category=UserWarning,  module="huggingface_hub.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from muq import MuQ, MuQMuLan  # noqa: E402

MUQ_MODEL_ID   = "OpenMuQ/MuQ-large-msd-iter"
MULAN_MODEL_ID = "OpenMuQ/MuQ-MuLan-large"
SAMPLE_RATE    = 24000
SEGMENT_SECS   = 30
MAX_SEGMENTS   = 3

# ── Genre / style prompts ─────────────────────────────────────────────────────
# MuQ-MuLan was trained on descriptive music-text pairs, so natural-language
# descriptions outperform bare label names.
GENRES: dict[str, str] = {
    # ── Retro / chiptune ─────────────────────────────────────────────────────
    "Chiptune":          "8-bit chiptune video game music with square waves, triangle bass and noise percussion",
    "16-bit":            "16-bit SNES-era game music with FM synthesis, sampled instruments and melodic loops",
    "Synthwave":         "retro synthwave with 80s analogue synthesizers, arpeggios and nostalgic game feel",

    # ── Game function / mood ─────────────────────────────────────────────────
    "Boss Battle":       "intense dramatic boss battle music with heavy percussion, dissonance and building tension",
    "Dungeon":           "dark atmospheric dungeon or cave music, eerie pads and minor key melodies",
    "Exploration":       "adventurous open-world exploration music, sweeping and hopeful",
    "Platformer":        "bouncy energetic platformer game music, upbeat and rhythmically driven",
    "Puzzle":            "light playful puzzle game music, repetitive and calm with simple motifs",
    "Horror":            "horror game music, unsettling dissonance, silence and sudden stingers",
    "Victory / Fanfare": "short triumphant victory fanfare or jingle typical of game completion",
    "Title Screen":      "atmospheric game title or menu screen music, memorable and introductory",

    # ── Setting / aesthetic ──────────────────────────────────────────────────
    "Fantasy RPG":       "epic fantasy role-playing game orchestral score with strings, choir and brass",
    "Sci-Fi / Space":    "futuristic science fiction or space game music with electronic pads and cosmic atmosphere",
    "Cyberpunk":         "cyberpunk urban game music with industrial beats, distorted bass and glitchy electronics",
    "Tropical / Kawaii": "bright cheerful tropical or cute Japanese-style game music with steel drums or bells",
    "Fighting Game":     "aggressive fast-paced fighting game music with heavy guitar riffs and driving rhythm",
    "Street / Urban":    "urban street game music with hip hop beats, scratching and city atmosphere",

    # ── Universal genres heavily used in game soundtracks ───────────────────
    "Orchestral":        "cinematic orchestral game score, dynamic and emotionally expressive",
    "Electronic":        "electronic game music with synthesizers, arpeggios and programmed beats",
    "Rock":              "rock music in a game soundtrack with electric guitars and a strong drum beat",
    "Heavy Metal":       "heavy metal game music with distorted guitars, double-kick drums and aggressive energy",
    "Jazz":              "jazz-influenced game music with improvisation, swing and acoustic instruments",
    "Ambient":           "ambient atmospheric game music with slow evolving pads and minimal melody",
    "Lo-Fi":             "lo-fi relaxed warm beats, mellow and calm like background study music",
    "Acoustic":          "acoustic guitar-driven game music, intimate and natural sounding",
}


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_segments(audio_path: str) -> tuple[list[np.ndarray], float]:
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration  = len(y) / SAMPLE_RATE
    seg_len   = SEGMENT_SECS * SAMPLE_RATE

    if len(y) <= seg_len:
        return [y], duration

    n      = min(MAX_SEGMENTS, len(y) // seg_len)
    starts = np.linspace(0, len(y) - seg_len, n, dtype=int)
    return [y[s : s + seg_len] for s in starts], duration


# ── MuQ feature extraction ────────────────────────────────────────────────────

def extract_muq_embedding(
    segments: list[np.ndarray],
    model: MuQ,
    device: str,
) -> torch.Tensor:
    """Return mean-pooled MuQ embedding averaged across segments. Shape: (1024,)"""
    per_seg = []
    for seg in segments:
        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x, output_hidden_states=True)
        # last_hidden_state: (1, T, 1024) -> mean pool -> (1024,)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        per_seg.append(emb)
    return torch.stack(per_seg).mean(dim=0)  # (1024,)


# ── MuQ-MuLan zero-shot genre matching ───────────────────────────────────────

def classify_genre(
    segments: list[np.ndarray],
    mulan: MuQMuLan,
    device: str,
) -> list[tuple[str, float]]:
    """Rank genres by cosine similarity between audio and text embeddings."""
    # Audio embeddings (one per segment, then average)
    seg_embeds = []
    for seg in segments:
        wav = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = mulan(wavs=wav)   # (1, D)
        seg_embeds.append(emb.cpu())
    audio_emb = torch.stack(seg_embeds).mean(dim=0)   # (1, D)

    # Text embeddings for all genre prompts
    genre_names  = list(GENRES.keys())
    genre_texts  = list(GENRES.values())
    with torch.no_grad():
        text_embs = mulan(texts=genre_texts).cpu()     # (N, D)

    # Cosine similarity
    sims = mulan.calc_similarity(
        audio_emb.to(device),
        text_embs.to(device),
    ).cpu()                                            # (1, N)  or  (N,)

    if sims.dim() == 2:
        sims = sims[0]   # (N,)

    top_idx = torch.argsort(sims, descending=True)
    return [(genre_names[i], float(sims[i])) for i in top_idx]


from tags import update_tags  # noqa: E402


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze music with OpenMuQ (feature extraction + MuLan text matching)"
    )
    ap.add_argument("files", nargs="+", metavar="FILE", help="MP3 or WAV file(s)")
    ap.add_argument("--top", type=int, default=10,
                    help="Top-N genres to display (default: 10)")
    ap.add_argument("--no-tag", action="store_true",
                    help="Skip writing TCON genre tag")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading MuQ  ({MUQ_MODEL_ID}) ...")
    muq = MuQ.from_pretrained(MUQ_MODEL_ID).to(device).eval()

    print(f"Loading MuLan ({MULAN_MODEL_ID}) ...")
    mulan = MuQMuLan.from_pretrained(MULAN_MODEL_ID).to(device).eval()

    print(f"  Device: {device}\n")

    for path in args.files:
        if not Path(path).exists():
            print(f"Not found: {path}", file=sys.stderr)
            continue

        print(f"{'='*60}")
        print(f"File: {path}")

        segments, duration = load_segments(path)
        print(f"  Duration : {duration:.1f}s  |  Segments: {len(segments)} x {SEGMENT_SECS}s")

        # Step 1 — MuQ feature extraction
        print("  [MuQ]   Extracting audio embeddings ...")
        muq_emb = extract_muq_embedding(segments, muq, device)
        norm    = float(muq_emb.norm())
        print(f"          Embedding dim: {muq_emb.shape[0]}  |  L2 norm: {norm:.2f}")

        # Step 2 — Genre matching via MuQ-MuLan
        print("  [MuLan] Matching genre text prompts ...")
        genre_results = classify_genre(segments, mulan, device)

        # Display results
        top3_labels = {g for g, _ in genre_results[:3]}
        print(f"\n  {'#':<4} {'Genre':<18} {'Similarity':>10}  Bar")
        print(f"  {'-'*58}")
        for rank, (genre, score) in enumerate(genre_results[: args.top], 1):
            bar  = "#" * max(0, int(score * 30))
            flag = " *" if genre in top3_labels else "  "
            print(f"  {rank:<4} {genre:<18} {score:>10.4f}  {bar}{flag}")

        print("\n  Top-3 genres for tag:")
        for i, (genre, score) in enumerate(genre_results[:3], 1):
            print(f"    {i}. {genre}  (similarity: {score:.4f})")

        if not args.no_tag:
            update_tags(path, [g for g, _ in genre_results[:3]])

        print()


if __name__ == "__main__":
    main()
