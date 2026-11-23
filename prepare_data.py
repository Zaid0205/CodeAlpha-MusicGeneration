"""
Step 1: collect MIDI files and turn them into token sequences.

If data/midi/ is empty, the Bach chorales that ship with music21 are exported
there as MIDI. You can also drop your own .mid files into data/midi/.
"""

import json
from pathlib import Path

from music21 import corpus

from music_utils import midi_to_tokens

MIDI_DIR = Path("data/midi")
OUT_FILE = Path("data/notes.json")


def export_bach_chorales():
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    paths = corpus.getComposer("bach")
    print(f"Exporting {len(paths)} Bach chorales to {MIDI_DIR}/ ...")
    for p in paths:
        try:
            corpus.parse(p).write("midi", fp=str(MIDI_DIR / (Path(p).stem + ".mid")))
        except Exception as e:
            print("  skipped", p.name, e)


def main():
    files = sorted(MIDI_DIR.glob("*.mid")) + sorted(MIDI_DIR.glob("*.midi"))
    if not files:
        export_bach_chorales()
        files = sorted(MIDI_DIR.glob("*.mid"))

    pieces = []
    for i, f in enumerate(files, 1):
        try:
            tokens = midi_to_tokens(f)
        except Exception as e:
            print("  skipped", f.name, e)
            continue
        if len(tokens) > 10:
            pieces.append(tokens)
        if i % 50 == 0:
            print(f"  parsed {i}/{len(files)}")

    OUT_FILE.write_text(json.dumps(pieces))
    total = sum(len(p) for p in pieces)
    vocab = len({t for p in pieces for t in p})
    print(f"Saved {len(pieces)} pieces, {total} tokens, {vocab} unique tokens to {OUT_FILE}")


if __name__ == "__main__":
    main()
