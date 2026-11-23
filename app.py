import tempfile
from pathlib import Path

import streamlit as st

from generate import generate, load_model, random_seed
from music_utils import midi_to_wav, tokens_to_midi
from scenario import compose

st.set_page_config(page_title="Music Generator")


@st.cache_resource
def get_model():
    return load_model()


def show_result():
    if "result" not in st.session_state:
        return
    midi_bytes, wav_bytes, info = st.session_state.result
    if info:
        st.info(info)
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("Download MIDI", midi_bytes, "generated.mid", "audio/midi")


st.title("Music Generation with an LSTM")
st.caption("Trained on 430 Bach chorales.")

model, vocab = get_model()
scene_tab, manual_tab = st.tabs(["Describe a scene", "Manual"])

with scene_tab:
    text = st.text_area(
        "What's the scene?",
        placeholder="a rainy night, walking home alone",
    )
    st.caption("Examples: kids dancing at a birthday party · a peaceful sunrise over the mountains · an epic battle in a dark castle")
    length = st.slider("Length (chords)", 50, 300, 150, 25, key="scene_len")
    if st.button("Compose", type="primary", disabled=not text.strip()):
        with st.spinner("Composing..."):
            mood, midi_path, wav_path = compose(text, model, vocab, length, out_dir=tempfile.mkdtemp())
        st.session_state.result = (Path(midi_path).read_bytes(), Path(wav_path).read_bytes(),
                                   f"Mood: {mood.describe()}")

with manual_tab:
    length = st.slider("Length (chords)", 50, 400, 200, 25)
    temperature = st.slider("Temperature", 0.2, 1.5, 0.8, 0.1,
                            help="Low = safe and repetitive, high = more surprising")
    bpm = st.slider("Tempo (BPM)", 60, 160, 90, 5)
    if st.button("Generate"):
        with st.spinner("Composing..."):
            tokens = generate(model, vocab, random_seed(), length, temperature)
            tmp = Path(tempfile.mkdtemp())
            midi_path = tokens_to_midi(tokens, tmp / "generated.mid", bpm=bpm)
            wav_path = midi_to_wav(midi_path, tmp / "generated.wav")
        st.session_state.result = (midi_path.read_bytes(), wav_path.read_bytes(), None)

show_result()
