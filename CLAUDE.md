# CLAUDE.md

## Project overview

Two standalone music genre classification scripts. Each script is self-contained — no shared state beyond `tags.py`.

## Environment

- Python 3.13, Windows 11
- Venv at `venv/` — always run scripts through `venv/Scripts/python.exe` or activate first
- `essentia` and `essentia-tensorflow` do **not** install on Python 3.13 / Windows; the project intentionally avoids them

## Running scripts

```powershell
.\venv\Scripts\Activate.ps1
python ast_analyze.py test/a_breath_of_air_2876.mp3
python muq_analyze.py test/a_breath_of_air_2876.mp3
```

## Key design decisions

**AST (`ast_analyze.py`)**
- Model: `MIT/ast-finetuned-audioset-10-10-0.4593` via `transformers`
- Audio: 16 kHz, split into 10-second segments (up to 6), logits averaged
- Genre selection: searches all 527 AudioSet labels ranked by confidence; prefers genre-keyword matches, then any label containing "music", then top-1 overall

**MuQ (`muq_analyze.py`)**
- Models: `OpenMuQ/MuQ-large-msd-iter` (audio features) + `OpenMuQ/MuQ-MuLan-large` (text-audio similarity)
- Audio: 24 kHz (MuQ native), split into 30-second segments (up to 3)
- Genre selection: cosine similarity between mean-pooled audio embedding and 24 descriptive genre text prompts
- MuQ alone cannot produce genre labels — MuQ-MuLan is required for the joint embedding space

**Tagging (`tags.py`)**
- Uses `TXXX` ID3 frames with `desc="genre1"`, `"genre2"`, `"genre3"` instead of a single `TCON` list
- Supports `.mp3` (via `mutagen.mp3.MP3`) and `.wav` (via `mutagen.wave.WAVE`)

## Adding a new analyzer

1. Create `<model_name>_analyze.py`
2. Import `update_tags` from `tags.py`
3. Call `update_tags(path, [genre1, genre2, genre3])`

## Models download location

HuggingFace cache: `~/.cache/huggingface/hub/`
- AST model: ~90 MB
- MuQ: ~300 MB
- MuQ-MuLan: ~700 MB
