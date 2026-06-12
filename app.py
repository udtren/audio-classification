#!/usr/bin/env python3
"""
Music Folder Analyzer — PyQt6 GUI
Browse to input/output folders, scan for MP3/WAV, then analyze each file:
  - librosa  → BPM
  - MuQ-MuLan → top-3 genres
  - music2emo → valence + arousal (1–9 scale)
Tags are written to the file, then the file is moved to the output folder.
"""

import os
import shutil
import sys
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

MUSIC_EXTENSIONS = {".mp3", ".wav"}


def find_music_files(folder: str) -> list[Path]:
    root = Path(folder)
    found: list[Path] = []
    for ext in MUSIC_EXTENSIONS:
        found.extend(root.rglob(f"*{ext}"))
    return sorted(found)


def move_to_output(src: Path, output_dir: Path) -> Path:
    """Move src to output_dir, appending a counter on name conflicts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / src.name
    if dst.exists() and dst.resolve() != src.resolve():
        stem, suffix = src.stem, src.suffix
        n = 1
        while dst.exists():
            dst = output_dir / f"{stem} ({n}){suffix}"
            n += 1
    shutil.move(str(src), dst)
    return dst


# ── Worker thread ─────────────────────────────────────────────────────────────


class AnalysisWorker(QThread):
    log          = pyqtSignal(str)
    file_started = pyqtSignal(int)                              # row
    file_done    = pyqtSignal(int, float, list, float, float)   # row, bpm, genres, valence, arousal
    file_error   = pyqtSignal(int, str)                         # row, message
    finished_all = pyqtSignal(int)                              # total successful

    def __init__(self, files: list[Path], output_dir: Path):
        super().__init__()
        self.files      = files
        self.output_dir = output_dir
        self._stop      = False

    def stop(self):
        self._stop = True

    def run(self):
        import torch
        from muq import MuQMuLan
        from analyzers.muq_analyze import MULAN_MODEL_ID, load_segments, classify_genre
        from analyzers.bpm_analyze import detect_bpm
        from analyzers.emo_analyze import load_model as load_emo, analyze_emotion
        from analyzers.tags import write_tags, update_tags

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.log.emit(f"Loading MuQ-MuLan on {device} …")
        mulan = MuQMuLan.from_pretrained(MULAN_MODEL_ID).to(device).eval()
        self.log.emit("MuQ-MuLan ready.")

        self.log.emit("Loading Music2emo …")
        emo_model, emo_dir = load_emo()
        self.log.emit("Music2emo ready.\n")

        count = 0
        for row, path in enumerate(self.files):
            if self._stop:
                self.log.emit("Stopped by user.")
                break

            self.file_started.emit(row)
            self.log.emit(f"[{row + 1}/{len(self.files)}] {path.name}")
            try:
                # BPM
                bpm, duration = detect_bpm(str(path))
                self.log.emit(f"  Duration : {duration:.1f}s  |  BPM: {bpm:.1f}")
                write_tags(str(path), {"bpm": f"{bpm:.1f}"})

                # Genre
                segments, _ = load_segments(str(path))
                genre_results = classify_genre(segments, mulan, device)
                genres = [g for g, _ in genre_results[:3]]
                self.log.emit(f"  Genres   : {', '.join(genres)}")
                update_tags(str(path), genres)

                # Emotion
                valence, arousal, moods = analyze_emotion(str(path), emo_model, emo_dir)
                self.log.emit(f"  Emotion  : valence={valence:.2f}  arousal={arousal:.2f}"
                              + (f"  moods={', '.join(moods)}" if moods else ""))
                write_tags(str(path), {"valence": f"{valence:.2f}", "arousal": f"{arousal:.2f}"})

                # Move
                dest = move_to_output(path, self.output_dir)
                self.log.emit(f"  Moved  → {dest.name}")

                self.file_done.emit(row, bpm, genres, valence, arousal)
                count += 1
            except Exception as exc:
                self.log.emit(f"  Error: {exc}")
                self.file_error.emit(row, str(exc))

        self.finished_all.emit(count)


# ── Main window ───────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Folder Analyzer")
        self.resize(1080, 700)
        self.files: list[Path] = []
        self.worker: AnalysisWorker | None = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Input folder row ─────────────────────────────────────────────────
        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Input :"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Folder containing music files …")
        self.folder_edit.setReadOnly(True)
        in_row.addWidget(self.folder_edit, 1)
        self.browse_btn = QPushButton("Browse …")
        self.browse_btn.clicked.connect(self._browse_input)
        in_row.addWidget(self.browse_btn)
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.setEnabled(False)
        self.scan_btn.clicked.connect(self._scan)
        in_row.addWidget(self.scan_btn)
        root.addLayout(in_row)

        # ── Output folder row ────────────────────────────────────────────────
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Folder to move tagged files into …")
        self.output_edit.setReadOnly(True)
        out_row.addWidget(self.output_edit, 1)
        self.output_browse_btn = QPushButton("Browse …")
        self.output_browse_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_browse_btn)
        root.addLayout(out_row)

        # ── Status + progress ────────────────────────────────────────────────
        status_row = QHBoxLayout()
        self.status_label = QLabel("Select input and output folders, then Scan.")
        status_row.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        status_row.addWidget(self.progress, 1)
        root.addLayout(status_row)

        # ── Splitter: table (top) + log (bottom) ─────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Columns: # | File | BPM | Genres | Emotion | Status
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "File", "BPM", "Genres", "Emotion", "Status"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 44)
        self.table.setColumnWidth(2, 70)
        self.table.setColumnWidth(4, 130)
        self.table.setColumnWidth(5, 100)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        splitter.addWidget(self.table)

        log_wrap = QWidget()
        log_layout = QVBoxLayout(log_wrap)
        log_layout.setContentsMargins(0, 4, 0, 0)
        log_layout.addWidget(QLabel("Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier New", 9))
        log_layout.addWidget(self.log_box)
        splitter.addWidget(log_wrap)
        splitter.setSizes([420, 200])
        root.addWidget(splitter, 1)

        # ── Bottom buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        btn_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_analysis)
        btn_row.addWidget(self.stop_btn)
        root.addLayout(btn_row)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_start_btn(self):
        self.start_btn.setEnabled(
            bool(self.files) and bool(self.output_edit.text().strip())
        )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.folder_edit.setText(folder)
            self.scan_btn.setEnabled(True)
            self.status_label.setText("Input folder set. Click Scan to find music files.")

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)
            self._update_start_btn()

    def _scan(self):
        folder = self.folder_edit.text()
        self.files = find_music_files(folder)
        self._populate_table()
        n = len(self.files)
        self.status_label.setText(
            f"Found {n} music file(s)." if n else "No music files found in this folder."
        )
        self._update_start_btn()

    def _populate_table(self):
        self.table.setRowCount(0)
        for i, path in enumerate(self.files):
            self.table.insertRow(i)
            self._set_cell(i, 0, str(i + 1), center=True)
            self._set_cell(i, 1, path.name)
            self._set_cell(i, 2, "")
            self._set_cell(i, 3, "")
            self._set_cell(i, 4, "")
            self.table.setItem(i, 5, self._status_item("Waiting"))

    def _set_cell(self, row: int, col: int, text: str, center: bool = False):
        item = QTableWidgetItem(text)
        if center:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, col, item)

    def _status_item(self, text: str, color: QColor | None = None) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if color:
            item.setForeground(color)
        return item

    def _start(self):
        output_dir = Path(self.output_edit.text().strip())
        self.log_box.clear()
        self.progress.setValue(0)
        self.progress.setMaximum(len(self.files))
        self.progress.setVisible(True)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)
        self.output_browse_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.log_box.append(f"Output folder: {output_dir}\n")

        self.worker = AnalysisWorker(self.files, output_dir)
        self.worker.log.connect(self._append_log)
        self.worker.file_started.connect(self._on_file_started)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.file_error.connect(self._on_file_error)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.start()

    def _stop_analysis(self):
        if self.worker:
            self.worker.stop()
        self.stop_btn.setEnabled(False)

    def _append_log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    def _on_file_started(self, row: int):
        self.table.setItem(row, 5, self._status_item("Analyzing…", QColor("#e0a020")))
        self.table.scrollToItem(self.table.item(row, 0))

    def _on_file_done(self, row: int, bpm: float, genres: list, valence: float, arousal: float):
        self._set_cell(row, 2, f"{bpm:.1f}", center=True)
        self._set_cell(row, 3, ", ".join(genres))
        self._set_cell(row, 4, f"V:{valence:.2f}  A:{arousal:.2f}", center=True)
        self.table.setItem(row, 5, self._status_item("Moved", QColor("#3cb371")))
        self.progress.setValue(self.progress.value() + 1)

    def _on_file_error(self, row: int, msg: str):
        self.table.setItem(row, 5, self._status_item("Error", QColor("#e05252")))
        self.progress.setValue(self.progress.value() + 1)

    def _on_finished(self, count: int):
        self.status_label.setText(
            f"Done — {count}/{len(self.files)} file(s) tagged and moved."
        )
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.output_browse_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self._append_log(f"\nFinished. {count}/{len(self.files)} file(s) tagged and moved.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
