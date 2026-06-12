# CLAUDE.md

## Project overview

Music folder analyzer with two entry points:
- `app.py` — PyQt6 GUI (browse input/output folders, scan, analyze, move)
- `analyze_folder.py` — CLI equivalent

Both use functions imported from `analyzers/` — no subprocess calls.

## Environment

- Python 3.13, Windows 11
- Venv at `venv/` — always run through `venv/Scripts/python.exe` or activate first
- `essentia` and `essentia-tensorflow` do **not** install on Python 3.13 / Windows; intentionally avoided
- PyTorch should be the CUDA wheel (`cu121` or `cu118`) for GPU acceleration; CPU wheel also works

## Running

```powershell
.\venv\Scripts\Activate.ps1
python app.py                                              # GUI
python analyze_folder.py                                   # CLI

# Individual analyzers (still runnable directly):
python analyzers/muq_analyze.py test/a_breath_of_air_2876.mp3 --top 10
python analyzers/bpm_analyze.py test/a_breath_of_air_2876.mp3
python analyzers/emo_analyze.py test/a_breath_of_air_2876.mp3
python analyzers/ast_analyze.py test/a_breath_of_air_2876.mp3
```

## Package layout

```
analyzers/
├── __init__.py          # empty — marks it as a package
├── tags.py              # write_tags(path, dict) and update_tags(path, genres)
├── bpm_analyze.py       # detect_bpm(path) → (bpm, duration)
├── muq_analyze.py       # load_segments(), classify_genre() — MuQ-MuLan only
├── emo_analyze.py       # load_model(), analyze_emotion() — music2emo valence/arousal
└── ast_analyze.py       # classify() — MIT AST, 527 AudioSet labels (standalone only)
```

Each analyzer script adds `sys.path.insert(0, str(Path(__file__).parent.resolve()))` so
`from tags import ...` resolves to `analyzers/tags.py` whether run directly or imported
as `analyzers.<module>`.

## Key design decisions

**MuQ (`analyzers/muq_analyze.py`)**
- Model: `OpenMuQ/MuQ-MuLan-large`
- Audio: 24 kHz (MuQ native), up to 3 × 30-second segments
- Genre list (`GENRES` dict) uses descriptive natural-language prompts tuned for game music:
  retro/chiptune, game-function moods, setting aesthetics, and universal genres
- Top-5 genres are written as `genre1`–`genre5` TXXX tags
- `classify_genre()` only needs MuLan — `app.py` and `analyze_folder.py` skip loading the
  heavier MuQ encoder to save ~300 MB RAM

**BPM (`analyzers/bpm_analyze.py`)**
- Uses `librosa.beat.beat_track()` at 22050 Hz
- Returns a scalar via `np.atleast_1d(tempo)[0]` (handles both librosa <0.10 scalar and ≥0.10 array)

**Emotion (`analyzers/emo_analyze.py`)**
- Model: `amaai-lab/music2emo` downloaded via `snapshot_download`; NOT a pip package
- Returns `valence` and `arousal` on a 1–9 scale, written as TXXX tags
- Three monkey-patches applied before importing `music2emo`:
  1. `_patch_torch_load()` — sets `map_location` to the right device and `weights_only=False`
     (checkpoint uses numpy globals, incompatible with PyTorch 2.6 new default)
  2. `_patch_torchaudio_load()` — replaces `torchaudio.load` with a librosa wrapper
     (torchaudio 2.5+ dropped soundfile/sox backends, requires torchcodec)
  3. `_stub_gradio()` — injects a fake `gradio` module (music2emo imports it at module level
     for its web demo; the inference path never calls it)
- Both `Music2emo.__init__()` and `predict()` load files with relative paths from the repo
  root; solved by wrapping both calls with `os.chdir(model_dir)` / restore
- `load_model()` returns `(model, model_dir)` and caches in a module-level `_cache` dict
- `analyze_emotion(audio_path, model, model_dir)` always passes an absolute path to `predict()`

**AST (`analyzers/ast_analyze.py`)**
- Model: `MIT/ast-finetuned-audioset-10-10-0.4593` via `transformers`
- Audio: 16 kHz, up to 6 × 10-second segments, logits averaged across segments
- Standalone only — not integrated into `app.py` / `analyze_folder.py`

**Tagging (`analyzers/tags.py`)**
- `write_tags(path, dict[str, str])` — generic TXXX writer for any key/value pair
- `update_tags(path, genres)` — convenience wrapper; writes `genre1`…`genreN` for any list length
- Supports `.mp3` (mutagen.mp3.MP3) and `.wav` (mutagen.wave.WAVE)

**File moving (`app.py`, `analyze_folder.py`)**
- `move_to_output(src, output_dir)` — creates output dir if needed, appends `(n)` counter on name conflict
- Files are moved **after** tagging, so a crash mid-move leaves the tagged original in the input folder

## Adding a new analyzer

1. Create `analyzers/<model>_analyze.py`
2. Add `sys.path.insert(0, str(Path(__file__).parent.resolve()))` after stdlib imports
3. Import `write_tags` or `update_tags` from `tags`
4. Export the detection/classification function so `app.py` / `analyze_folder.py` can import it

## Model sizes (HuggingFace cache: `~/.cache/huggingface/hub/`)

| Model | Size |
|---|---|
| MIT AST | ~90 MB |
| OpenMuQ/MuQ-large-msd-iter | ~300 MB |
| OpenMuQ/MuQ-MuLan-large | ~700 MB |
| amaai-lab/music2emo | ~400 MB |
