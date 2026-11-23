"""
Make a piece of music from a short text description of a scene.

The text is turned into a mood (sentiment with NLTK's VADER plus a list of
energy words), and the mood controls how the LSTM generates:
  - major or minor: which training passages are used as the starting seed
  - energy: tempo, temperature and whether short or long chords are favoured
  - key, register and playing style (block chords or arpeggios)
"""

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nltk
import numpy as np

from generate import generate, load_model
from music_utils import midi_to_wav, tokens_to_midi
from train import SEQ_LEN

HIGH_ENERGY = {
    "party", "dance", "dancing", "run", "running", "chase", "battle", "fight", "war",
    "race", "storm", "festival", "celebrate", "celebration", "wedding", "adventure",
    "epic", "fast", "exciting", "victory", "jump", "concert", "game", "action", "hero",
}
LOW_ENERGY = {
    "sleep", "sleeping", "rain", "rainy", "night", "calm", "peaceful", "quiet", "sunset",
    "sunrise", "lake", "meditation", "lonely", "alone", "slow", "gentle", "lullaby",
    "soft", "dream", "dreaming", "snow", "morning", "reading", "tea", "memory", "memories",
}
SAD = {"rain", "rainy", "lonely", "alone", "goodbye", "loss", "lost", "funeral", "missing",
       "dark", "grief", "cry", "crying", "tears", "broken", "war", "empty", "cold"}
HAPPY = {"party", "birthday", "sunny", "wedding", "celebrate", "celebration", "love",
         "victory", "friends", "summer", "festival", "smile", "home", "spring"}

MAJOR_KEYS = {"C major": 0, "G major": -5, "D major": 2, "F major": 5}
MINOR_KEYS = {"A minor": 0, "E minor": -5, "D minor": 5, "G minor": -2}


@dataclass
class Mood:
    valence: float  # -1 sad .. 1 happy
    energy: float   # 0 calm .. 1 intense
    mode: str
    key: str
    transpose: int
    bpm: int
    temperature: float
    style: str
    octave: int

    def describe(self):
        feel = "happy" if self.valence > 0.3 else "sad" if self.valence < -0.3 else "neutral"
        pace = "energetic" if self.energy > 0.65 else "calm" if self.energy < 0.4 else "moderate"
        if feel == "sad" and pace == "energetic":
            feel = "dark"
        return f"{feel}, {pace} ({self.key}, {self.bpm} BPM, {self.style})"


def get_sentiment():
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def analyze(text):
    words = set(re.findall(r"[a-z]+", text.lower()))

    valence = get_sentiment().polarity_scores(text)["compound"]
    valence += 0.3 * (len(words & HAPPY) - len(words & SAD))
    valence = float(np.clip(valence, -1, 1))

    energy = 0.5 + 0.2 * (len(words & HIGH_ENERGY) - len(words & LOW_ENERGY))
    energy = float(np.clip(energy, 0, 1))

    mode = "major" if valence >= 0 else "minor"
    key = random.choice(list(MAJOR_KEYS if mode == "major" else MINOR_KEYS))
    transpose = (MAJOR_KEYS | MINOR_KEYS)[key]

    return Mood(
        valence=valence,
        energy=energy,
        mode=mode,
        key=key,
        transpose=transpose,
        bpm=int(60 + energy * 80),
        temperature=round(0.6 + energy * 0.4, 2),
        style="arpeggio" if energy < 0.45 else "block",
        octave=3 if mode == "minor" and energy < 0.5 else 4,
    )


def piece_mode(piece):
    """Guess major/minor from the final chord (everything was transposed to C major / A minor)."""
    for tok in reversed(piece):
        name = tok.rsplit("_", 1)[0]
        if name == "rest":
            continue
        notes = set(name.split("."))
        if {"A", "E"} <= notes:
            return "minor"
        if {"C", "G"} <= notes:
            return "major"
        return None
    return None


def seed_for(mode, data_path="data/notes.json"):
    pieces = json.loads(Path(data_path).read_text())
    matching = [p for p in pieces if len(p) > SEQ_LEN and piece_mode(p) == mode]
    piece = random.choice(matching)
    start = random.randint(0, len(piece) - SEQ_LEN)
    return piece[start:start + SEQ_LEN]


def duration_bias(vocab, energy):
    """Favour short chords for energetic scenes and long ones for calm scenes."""
    target = 2 ** (1 - 2.5 * energy)  # energy 0 -> 2 beats, energy 1 -> ~0.35 beats
    durations = np.array([float(t.rsplit("_", 1)[1]) for t in vocab])
    return np.exp(-0.8 * np.abs(np.log2(durations) - np.log2(target)))


def compose(text, model, vocab, length=150, out_dir="output"):
    mood = analyze(text)
    tokens = generate(
        model, vocab, seed_for(mood.mode), length, mood.temperature,
        bias=duration_bias(vocab, mood.energy),
    )
    name = datetime.now().strftime("scene_%Y%m%d_%H%M%S")
    midi_path = tokens_to_midi(
        tokens, Path(out_dir) / f"{name}.mid", bpm=mood.bpm,
        transpose=mood.transpose, octave=mood.octave, style=mood.style,
    )
    wav_path = midi_to_wav(midi_path, Path(out_dir) / f"{name}.wav")
    return mood, midi_path, wav_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help='e.g. "a rainy night, walking home alone"')
    parser.add_argument("--length", type=int, default=150)
    args = parser.parse_args()

    model, vocab = load_model()
    mood, midi_path, wav_path = compose(args.text, model, vocab, args.length)
    print("Mood:", mood.describe())
    print("Saved", midi_path, "and", wav_path)


if __name__ == "__main__":
    main()
