"""Shared tag-writing utilities for audio genre analyzers."""

from pathlib import Path


def update_tags(audio_path: str, genres: list[str]) -> None:
    """Write genres into individual TXXX frames: genre1, genre2, genre3."""
    ext = Path(audio_path).suffix.lower()
    try:
        if ext == ".mp3":
            from mutagen.mp3 import MP3
            from mutagen.id3 import TXXX
            audio = MP3(audio_path)
            if audio.tags is None:
                audio.add_tags()
            for i, genre in enumerate(genres, 1):
                audio.tags.add(TXXX(encoding=3, desc=f"genre{i}", text=[genre]))
            audio.save()

        elif ext == ".wav":
            from mutagen.wave import WAVE
            from mutagen.id3 import TXXX
            audio = WAVE(audio_path)
            if audio.tags is None:
                audio.add_tags()
            for i, genre in enumerate(genres, 1):
                audio.tags.add(TXXX(encoding=3, desc=f"genre{i}", text=[genre]))
            audio.save()

        else:
            print(f"  Skipping tag — unsupported format: {ext}")
            return

        for i, genre in enumerate(genres, 1):
            print(f"  Tag written : genre{i} = {genre!r}")

    except Exception as exc:
        print(f"  Warning: could not write tag — {exc}")
