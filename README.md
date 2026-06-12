# Audio Classification

Music analyzer that classifies MP3/WAV files and writes BPM + top-3 genre tags directly to each file, then moves the tagged files to an output folder.

## Analyzers

| Script | Model | What it does |
|---|---|---|
| `analyzers/ast_analyze.py` | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | Classifies against 527 AudioSet labels |
| `analyzers/muq_analyze.py` | [OpenMuQ/MuQ-large-msd-iter](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) + [MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | Zero-shot genre matching via audio-text cosine similarity |
| `analyzers/bpm_analyze.py` | librosa beat tracker | Detects tempo and writes a `bpm` tag |
| `analyzers/tags.py` | — | Shared TXXX tag writer used by all analyzers |

## Requirements

- Python 3.13+
- Windows / macOS / Linux

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install muq   # OpenMuQ package (not on PyPI index, install separately)
pip install torch --index-url https://download.pytorch.org/whl/cpu
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
4. Click **Start Analysis** — each file is tagged with BPM + top-3 genres then moved to the output folder

The table updates live with BPM, genres, and status (Analyzing → Moved / Error). A log panel streams progress.

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
python analyzers/ast_analyze.py track.mp3
python analyzers/muq_analyze.py track.mp3 --top 10
python analyzers/bpm_analyze.py track.wav --no-tag
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

## File structure

```
audio-classification/
├── app.py               # PyQt6 GUI — batch analyze + move
├── analyze_folder.py    # CLI equivalent of app.py
├── requirements.txt
├── analyzers/
│   ├── ast_analyze.py   # AST genre classifier
│   ├── muq_analyze.py   # MuQ + MuQ-MuLan genre classifier
│   ├── bpm_analyze.py   # librosa BPM detector
│   └── tags.py          # Shared TXXX tag writer
└── test/                # Sample audio files
```
