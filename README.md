# Music Generation with an LSTM

CodeAlpha AI Internship, Task 3.

An LSTM network trained on 430 Bach chorales that generates new chord progressions and saves them as MIDI and WAV. You can also describe a scene in words ("a rainy night, walking home alone") and it composes music to match the mood.

![screenshot](docs/screenshot.png)

Examples:
- "kids dancing at a birthday party": [birthday_party.wav](samples/birthday_party.wav)
- "a rainy night, walking home alone": [rainy_night.wav](samples/rainy_night.wav)
- no scene, default settings: [sample.wav](samples/sample.wav)

## How it works

1. **Data** (`prepare_data.py`): the Bach chorales that come with music21 are exported to `data/midi/` as MIDI files. Each file is parsed with music21, transposed to C major / A minor, and its parts are merged into chords. Every chord becomes a token like `A.C.E_1.0` (pitch classes + duration in beats). This gives 43k tokens with a vocabulary of 913.
2. **Model** (`train.py`): an Embedding layer, two LSTM layers (256 units) with dropout, and a softmax over the vocabulary. It sees 32 chords and predicts the next one. Training stops early when validation loss stops improving.
3. **Generation** (`generate.py`): start from a random 32-chord passage, predict the next chord, sample it with a temperature, append it and repeat. The tokens are converted back to a MIDI file with music21 and rendered to WAV with pretty_midi.

4. **Scene to music** (`scenario.py`): the description is scored for sentiment with NLTK's VADER and checked against lists of happy/sad and high/low energy words. The mood then steers the model:
   - positive scenes start from a major-key chorale passage, negative ones from a minor-key one
   - energy sets the tempo (60 to 140 BPM) and temperature, and re-weights the model's predictions toward short chords (energetic) or long chords (calm)
   - calm scenes are played as arpeggios in a lower register, energetic ones as block chords
   - the result is transposed to a random key that fits the mode

![training](docs/training.png)

The best validation accuracy is around 20%. That's expected, since many chords are valid continuations at any point, and the model overfits after about 7 epochs on this small dataset.

## Running it

```
pip install -r requirements.txt

python prepare_data.py   # build data/notes.json (already included)
python train.py          # train the model (a trained one is already in models/)
python generate.py       # write a new piece to output/
python scenario.py "an epic battle in a dark castle"
streamlit run app.py     # web demo
```

`generate.py` options: `--length 200 --temperature 0.8 --bpm 90`. Lower temperature is safer, higher is more adventurous.

Training takes about 10 minutes on a CPU. To train on your own music, put `.mid` files in `data/midi/` (delete the Bach ones first) and rerun `prepare_data.py` and `train.py`.

## Files

- `music_utils.py`: MIDI to tokens and back, WAV rendering
- `prepare_data.py`: builds the dataset
- `train.py`: model and training
- `generate.py`: sampling and output
- `scenario.py`: text description to mood to music
- `app.py`: Streamlit demo

Built with Python, music21, TensorFlow/Keras, pretty_midi and Streamlit.
