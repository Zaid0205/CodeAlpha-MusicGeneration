"""Step 3: generate new music with the trained model and save it as MIDI and WAV."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from music_utils import midi_to_wav, tokens_to_midi
from train import SEQ_LEN


def load_model(model_dir="models"):
    model = tf.keras.models.load_model(Path(model_dir) / "model.keras")
    vocab = json.loads((Path(model_dir) / "vocab.json").read_text())
    return model, vocab


def random_seed(data_path="data/notes.json"):
    """Pick a random 32-token passage from the training data to start from."""
    pieces = [p for p in json.loads(Path(data_path).read_text()) if len(p) > SEQ_LEN]
    piece = random.choice(pieces)
    start = random.randint(0, len(piece) - SEQ_LEN)
    return piece[start:start + SEQ_LEN]


def sample(probs, temperature):
    # higher temperature = more random, lower = safer and more repetitive
    logits = np.log(probs + 1e-9) / temperature
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return np.random.choice(len(p), p=p)


def generate(model, vocab, seed, length=200, temperature=0.8):
    index = {t: i for i, t in enumerate(vocab)}
    window = [index[t] for t in seed if t in index]
    window = window[-SEQ_LEN:]
    out = []
    for _ in range(length):
        x = np.array([window[-SEQ_LEN:]])
        probs = model(x, training=False).numpy()[0]
        nxt = sample(probs, temperature)
        out.append(vocab[nxt])
        window.append(nxt)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=200, help="number of chords/rests to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--bpm", type=int, default=90)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    model, vocab = load_model()
    tokens = generate(model, vocab, random_seed(), args.length, args.temperature)

    name = datetime.now().strftime("generated_%Y%m%d_%H%M%S")
    midi_path = tokens_to_midi(tokens, Path(args.out) / f"{name}.mid", bpm=args.bpm)
    wav_path = midi_to_wav(midi_path, Path(args.out) / f"{name}.wav")
    print("Saved", midi_path, "and", wav_path)


if __name__ == "__main__":
    main()
