import tempfile
from pathlib import Path

import streamlit as st

from generate import generate, load_model, random_seed
from music_utils import midi_to_wav, tokens_to_midi

st.set_page_config(page_title="Music Generator")


@st.cache_resource
def get_model():
    return load_model()


st.title("Music Generation with an LSTM")
st.caption("Trained on Bach chorales. Each run continues a random 32-chord passage from the dataset.")

model, vocab = get_model()

length = st.slider("Length (chords)", 50, 400, 200, 25)
temperature = st.slider("Temperature", 0.2, 1.5, 0.8, 0.1,
                        help="Low = safe and repetitive, high = more surprising")
bpm = st.slider("Tempo (BPM)", 60, 160, 90, 5)

if st.button("Generate", type="primary"):
    with st.spinner("Composing..."):
        tokens = generate(model, vocab, random_seed(), length, temperature)
        tmp = Path(tempfile.mkdtemp())
        midi_path = tokens_to_midi(tokens, tmp / "generated.mid", bpm=bpm)
        wav_path = midi_to_wav(midi_path, tmp / "generated.wav")
    st.session_state.result = (midi_path.read_bytes(), wav_path.read_bytes(), tokens)

if "result" in st.session_state:
    midi_bytes, wav_bytes, tokens = st.session_state.result
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("Download MIDI", midi_bytes, "generated.mid", "audio/midi")
    with st.expander("Generated tokens"):
        st.write(" ".join(tokens[:100]) + (" ..." if len(tokens) > 100 else ""))
