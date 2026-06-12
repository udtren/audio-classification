# Audio Classification

Music genre analyzer that classifies MP3/WAV files and writes the top-3 predicted genres as custom ID3 tags (`genre1`, `genre2`, `genre3`).

Two independent scripts, each using a different model:

| Script | Model | Approach |
|---|---|---|
| `ast_analyze.py` | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | Classifies directly against 527 AudioSet labels |
| `muq_analyze.py` | [OpenMuQ/MuQ-large-msd-iter](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) + [MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | Extracts audio embeddings, then ranks genre text prompts via cosine similarity |

Shared tag-writing logic lives in `tags.py`.

## Requirements

- Python 3.13+
- Windows / macOS / Linux

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Models are downloaded automatically from HuggingFace on the first run and cached in `~/.cache/huggingface/`.

## Usage

### AST analyzer

```powershell
python ast_analyze.py track.mp3
python ast_analyze.py *.mp3 --top 10
python ast_analyze.py track.wav --no-tag
```

### MuQ analyzer

```powershell
python muq_analyze.py track.mp3
python muq_analyze.py *.mp3 --top 10
python muq_analyze.py track.wav --no-tag
```

### Flags

| Flag | Description |
|---|---|
| `--top N` | Number of predictions to display (default: 15 for AST, 10 for MuQ) |
| `--no-tag` | Run analysis but skip writing tags to the file |

## Output tags

Each script writes three custom `TXXX` ID3 frames to the audio file:

| Tag | Value |
|---|---|
| `genre1` | Top predicted genre |
| `genre2` | Second predicted genre |
| `genre3` | Third predicted genre |

These are visible in any tagger that supports custom fields (Mp3tag, MusicBrainz Picard, foobar2000, etc.).

## File structure

```
audio-classification/
├── ast_analyze.py      # AST-based analyzer
├── muq_analyze.py      # MuQ + MuQ-MuLan analyzer
├── tags.py             # Shared TXXX tag writer
├── requirements.txt
└── test/               # Sample audio files
```
