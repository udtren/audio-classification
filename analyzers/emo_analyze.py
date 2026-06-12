#!/usr/bin/env python3
"""
Music emotion analyzer using amaai-lab/music2emo.
Predicts valence and arousal on a 1–9 scale and writes them as TXXX tags.

Extra dependencies (see requirements-emo.txt):
    pip install -r requirements-emo.txt
"""

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_REPO = "amaai-lab/music2emo"

# Module-level cache so the model is loaded only once per process.
_cache: dict = {}


def _patch_torch_load() -> None:
    """
    music2emo checkpoints were saved on a CUDA device.  Without map_location,
    torch.load tries to restore to the original device and fails when CUDA is
    unavailable (CPU-only PyTorch) or when we want to redirect to a specific
    device.  Wrap torch.load so it always lands on the right device.
    """
    import torch
    _orig_load = torch.load
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _patched_load(f, map_location=None, weights_only=None, *args, **kwargs):
        if map_location is None:
            map_location = torch.device(device)
        if weights_only is None:
            weights_only = False
        return _orig_load(f, map_location=map_location, weights_only=weights_only, *args, **kwargs)

    torch.load = _patched_load


def _patch_torchaudio_load() -> None:
    """
    Newer torchaudio (2.5+) removed the soundfile/sox backends and requires
    torchcodec, a heavy optional package.  Replace torchaudio.load with a
    librosa-based loader — librosa is already installed and handles all common
    formats reliably on Windows without native codec dependencies.
    """
    import numpy as np
    import torch

    try:
        import torchaudio
    except ModuleNotFoundError:
        return

    def _load(filepath, frame_offset=0, num_frames=-1, normalize=True,
              channels_first=True, format=None, backend=None):
        import librosa
        y, sr = librosa.load(str(filepath), sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]       # ensure (channels, samples)
        if not channels_first:
            y = y.T
        tensor = torch.from_numpy(y.copy())
        if normalize and tensor.dtype != torch.float32:
            tensor = tensor.float() / 32768.0
        return tensor, sr

    torchaudio.load = _load


def _stub_gradio() -> None:
    """
    music2emo.py imports gradio at module level for its web demo, but the
    inference path never calls it.  Inject a lightweight stub so we don't
    have to install the full Gradio package.
    """
    import types
    if "gradio" in sys.modules:
        return
    gr = types.ModuleType("gradio")
    for _name in (
        "Blocks", "Interface", "Audio", "Textbox", "Button",
        "Row", "Column", "Label", "Markdown", "HTML",
        "Dropdown", "Checkbox", "Slider", "Number", "File",
    ):
        setattr(gr, _name, type(_name, (), {"__init__": lambda self, *a, **kw: None}))
    gr.themes = types.SimpleNamespace(Soft=object, Base=object, Default=object)
    sys.modules["gradio"] = gr
    sys.modules["gradio.themes"] = types.ModuleType("gradio.themes")


def load_model():
    """
    Download (first run only) and load Music2emo.
    Uses os.chdir briefly during __init__ because the upstream code
    resolves several local weight paths relative to the repo root.
    """
    if MODEL_REPO in _cache:
        return _cache[MODEL_REPO]

    from huggingface_hub import snapshot_download

    print(f"Downloading/verifying {MODEL_REPO} …")
    model_dir = Path(snapshot_download(MODEL_REPO))

    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    _patch_torch_load()
    _patch_torchaudio_load()
    _stub_gradio()

    old_cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        from music2emo import Music2emo  # noqa: E402
        print("  Initialising Music2emo …")
        model = Music2emo()
    finally:
        os.chdir(old_cwd)

    _cache[MODEL_REPO] = (model, model_dir)
    return model, model_dir


def analyze_emotion(audio_path: str, model, model_dir: Path) -> tuple[float, float, list[str]]:
    """
    Return (valence, arousal, predicted_moods). Valence/arousal on 1–9 scale.
    model_dir is required because predict() loads config files with relative paths.
    """
    old_cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        result = model.predict(str(Path(audio_path).resolve()))
    finally:
        os.chdir(old_cwd)
    return float(result["valence"]), float(result["arousal"]), list(result.get("predicted_moods", []))


from tags import write_tags  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Predict music valence/arousal with music2emo and write TXXX tags"
    )
    ap.add_argument("files", nargs="+", metavar="FILE", help="MP3 or WAV file(s)")
    ap.add_argument("--no-tag", action="store_true", help="Skip writing tags")
    args = ap.parse_args()

    model, model_dir = load_model()
    print()

    for path in args.files:
        if not Path(path).exists():
            print(f"Not found: {path}", file=sys.stderr)
            continue

        print(f"{'='*60}")
        print(f"File: {path}")

        valence, arousal, moods = analyze_emotion(path, model, model_dir)
        print(f"  Valence  : {valence:.2f} / 9")
        print(f"  Arousal  : {arousal:.2f} / 9")
        if moods:
            print(f"  Moods    : {', '.join(moods)}")

        if not args.no_tag:
            write_tags(path, {"valence": f"{valence:.2f}", "arousal": f"{arousal:.2f}"})

        print()


if __name__ == "__main__":
    main()
