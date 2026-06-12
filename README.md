# Audio Tagger

Music analyzer that tags MP3/WAV files with BPM, top-5 genres, and emotion (valence + arousal), then moves the tagged files to an output folder.

## Analyzers

| Script | Model | What it does |
|---|---|---|
| `analyzers/muq_analyze.py` | [OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | Zero-shot genre matching via audio-text cosine similarity |
| `analyzers/bpm_analyze.py` | librosa beat tracker | Detects tempo and writes a `bpm` tag |
| `analyzers/emo_analyze.py` | [amaai-lab/music2emo](https://huggingface.co/amaai-lab/music2emo) | Predicts valence and arousal on a 1–9 scale |
| `analyzers/ast_analyze.py` | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | Classifies against 527 AudioSet labels (standalone only) |
| `analyzers/tags.py` | — | Shared TXXX tag writer used by all analyzers |

## Requirements

- Python 3.13+
- Windows 11 (tested), macOS / Linux should work

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-emo.txt
pip install muq   # OpenMuQ package
```

**PyTorch — pick the right wheel for your machine:**

```powershell
# NVIDIA GPU (recommended) — check your CUDA version with nvidia-smi first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8

# CPU only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> Models download automatically from HuggingFace on first run and cache in `~/.cache/huggingface/`.

## Batch processing

### GUI (PyQt6)

```powershell
python app.py
```

1. Click **Browse …** next to *Input* and select the folder with your music files
2. Click **Browse …** next to *Output* and select where tagged files should be moved
3. Click **Scan** to list all MP3/WAV files found
4. Click **Start Analysis** — each file is tagged then moved to the output folder

The table updates live with BPM, top-5 genres, emotion (`V:x.xx  A:x.xx`), and status. A log panel streams progress.

### CLI

```powershell
python analyze_folder.py
```

```
Input folder : C:\Music\inbox
Output folder: C:\Music\tagged
```

Processes every MP3/WAV in the input folder (recursively), tags each file, and moves it to the output folder.

## Running individual analyzers

Each analyzer script in `analyzers/` can also be run directly on one or more files:

```powershell
python analyzers/muq_analyze.py track.mp3 --top 10
python analyzers/bpm_analyze.py track.wav --no-tag
python analyzers/emo_analyze.py track.mp3
python analyzers/ast_analyze.py track.mp3
```

| Flag | Description |
|---|---|
| `--top N` | Predictions to display (default: 15 for AST, 10 for MuQ) |
| `--no-tag` | Analyse only — skip writing tags |

## Output tags

All tags are written as custom `TXXX` ID3 frames, visible in Mp3tag, MusicBrainz Picard, foobar2000, etc.

| Tag | Value |
|---|---|
| `bpm` | Detected tempo (e.g. `128.5`) |
| `genre1` | Top predicted genre |
| `genre2` | Second predicted genre |
| `genre3` | Third predicted genre |
| `genre4` | Fourth predicted genre |
| `genre5` | Fifth predicted genre |
| `valence` | Emotional valence, 1 (negative) – 9 (positive) |
| `arousal` | Emotional energy, 1 (calm) – 9 (excited) |

## Genre prompts

`muq_analyze.py` uses descriptive natural-language prompts tuned for game music. Categories include:

- **Retro / chiptune** — Chiptune (8-bit), 16-bit/SNES, Synthwave
- **Game function** — Boss Battle, Dungeon, Exploration, Platformer, Puzzle, Horror, Victory Fanfare, Title Screen
- **Setting / aesthetic** — Fantasy RPG, Sci-Fi/Space, Cyberpunk, Tropical/Kawaii, Fighting Game, Street/Urban
- **Universal** — Orchestral, Electronic, Rock, Heavy Metal, Jazz, Ambient, Lo-Fi, Acoustic

## File structure

```
audio-classification/
├── app.py                # PyQt6 GUI — batch analyze + move
├── analyze_folder.py     # CLI equivalent of app.py
├── requirements.txt
├── requirements-emo.txt  # Extra deps for emo_analyze.py
├── analyzers/
│   ├── muq_analyze.py    # MuQ-MuLan zero-shot genre classifier
│   ├── bpm_analyze.py    # librosa BPM detector
│   ├── emo_analyze.py    # music2emo valence/arousal predictor
│   ├── ast_analyze.py    # AST 527-class classifier (standalone)
│   └── tags.py           # Shared TXXX tag writer
└── test/                 # Sample audio files
```
