"""Shared tag-writing utilities for audio analyzers."""

from pathlib import Path


def write_tags(audio_path: str, tags: dict[str, str]) -> None:
    """Write arbitrary key/value pairs as TXXX ID3 frames to an MP3 or WAV file."""
    ext = Path(audio_path).suffix.lower()
    try:
        if ext == ".mp3":
            from mutagen.mp3 import MP3
            from mutagen.id3 import TXXX
            audio = MP3(audio_path)
            if audio.tags is None:
                audio.add_tags()
            for key, value in tags.items():
                audio.tags.add(TXXX(encoding=3, desc=key, text=[value]))
            audio.save()

        elif ext == ".wav":
            from mutagen.wave import WAVE
            from mutagen.id3 import TXXX
            audio = WAVE(audio_path)
            if audio.tags is None:
                audio.add_tags()
            for key, value in tags.items():
                audio.tags.add(TXXX(encoding=3, desc=key, text=[value]))
            audio.save()

        else:
            print(f"  Skipping tag — unsupported format: {ext}")
            return

        for key, value in tags.items():
            print(f"  Tag written : {key} = {value!r}")

    except Exception as exc:
        print(f"  Warning: could not write tag — {exc}")


def update_tags(audio_path: str, genres: list[str]) -> None:
    """Write top genres as TXXX frames genre1, genre2, genre3."""
    write_tags(audio_path, {f"genre{i}": g for i, g in enumerate(genres, 1)})
