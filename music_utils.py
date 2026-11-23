"""Converting between MIDI files and token sequences."""

from pathlib import Path

from music21 import chord, converter, instrument, interval, note, pitch, stream, tempo

# durations (in quarter notes) every note gets rounded to
DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def quantize(d):
    return min(DURATIONS, key=lambda x: abs(x - d))


def transpose_to_c(score):
    """Move a piece to C major / A minor so the model sees fewer distinct notes."""
    key = score.analyze("key")
    tonic = "C" if key.mode == "major" else "A"
    return score.transpose(interval.Interval(key.tonic, pitch.Pitch(tonic)))


def chord_name(pitches):
    """Pitch classes without octave, e.g. C.E.G. Keeps the vocabulary small."""
    return ".".join(sorted({p.name for p in pitches}))


def midi_to_tokens(path):
    """
    Parse a MIDI file into tokens like "C.E.G_1.0" (chord_duration) or
    "rest_0.5". All parts are merged into chords so the harmony is kept.
    """
    score = converter.parse(str(path))
    score = transpose_to_c(score)
    tokens = []
    for el in score.chordify().flatten().notesAndRests:
        d = quantize(float(el.quarterLength))
        if el.quarterLength == 0:
            continue
        if isinstance(el, note.Rest):
            tokens.append(f"rest_{d}")
        else:
            tokens.append(f"{chord_name(el.pitches)}_{d}")
    return tokens


def voice_chord(names, octave=4):
    """Turn pitch classes back into pitches: a bass note plus the chord in the given octave."""
    c = chord.Chord([n + str(octave) for n in names])
    bass = pitch.Pitch(c.root().name + str(max(octave - 2, 2)))
    return bass, list(c.pitches)


def tokens_to_midi(tokens, path, bpm=90, transpose=0, octave=4, style="block"):
    """
    Write tokens to a MIDI file.
    style="block" plays each chord as one hit, style="arpeggio" holds the bass
    and runs through the chord notes in eighth notes.
    """
    out = stream.Stream()
    out.insert(0, tempo.MetronomeMark(number=bpm))
    out.insert(0, instrument.Piano())
    offset = 0.0
    for tok in tokens:
        name, d = tok.rsplit("_", 1)
        d = float(d)
        if name != "rest":
            bass, upper = voice_chord(name.split("."), octave)
            if style == "arpeggio" and d >= 0.5:
                out.insert(offset, note.Note(bass, quarterLength=d))
                steps = int(d / 0.5)
                for i in range(steps):
                    out.insert(offset + i * 0.5, note.Note(upper[i % len(upper)], quarterLength=0.5))
            else:
                out.insert(offset, chord.Chord([bass, *upper], quarterLength=d))
        offset += d
    if transpose:
        out = out.transpose(transpose)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write("midi", fp=str(path))
    return path


def midi_to_wav(midi_path, wav_path, sample_rate=22050):
    """Render MIDI to a WAV file with a simple built-in synth (no soundfont needed)."""
    import numpy as np
    import pretty_midi
    from scipy.io import wavfile

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.synthesize(fs=sample_rate)
    audio = audio / (np.abs(audio).max() + 1e-9) * 0.9
    wavfile.write(str(wav_path), sample_rate, (audio * 32767).astype(np.int16))
    return wav_path
