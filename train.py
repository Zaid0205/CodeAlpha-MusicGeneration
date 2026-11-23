"""Step 2: train an LSTM to predict the next chord from the previous ones."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

SEQ_LEN = 32
MODEL_DIR = Path("models")


def load_data(path="data/notes.json"):
    pieces = json.loads(Path(path).read_text())
    vocab = sorted({t for p in pieces for t in p})
    return pieces, vocab


def make_sequences(pieces, vocab):
    index = {t: i for i, t in enumerate(vocab)}
    X, y = [], []
    for piece in pieces:
        ids = [index[t] for t in piece]
        for i in range(len(ids) - SEQ_LEN):
            X.append(ids[i:i + SEQ_LEN])
            y.append(ids[i + SEQ_LEN])
    return np.array(X), np.array(y)


def build_model(vocab_size):
    model = tf.keras.Sequential([
        layers.Input(shape=(SEQ_LEN,)),
        layers.Embedding(vocab_size, 96),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(256),
        layers.Dropout(0.3),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy"],
    )
    return model


def save_plot(history, path="docs/training.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, metric in enumerate(["loss", "accuracy"]):
        ax[i].plot(history.history[metric], label="train")
        ax[i].plot(history.history["val_" + metric], label="validation")
        ax[i].set_title(metric)
        ax[i].set_xlabel("epoch")
        ax[i].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    pieces, vocab = load_data()
    X, y = make_sequences(pieces, vocab)
    print(f"{len(X)} training sequences, vocabulary of {len(vocab)} tokens")

    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "vocab.json").write_text(json.dumps(vocab))

    model = build_model(len(vocab))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "model.keras", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
    ]
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        shuffle=True,
        callbacks=callbacks,
    )
    save_plot(history)
    print("Saved model to", MODEL_DIR / "model.keras")


if __name__ == "__main__":
    main()
