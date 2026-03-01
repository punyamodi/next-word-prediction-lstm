from __future__ import annotations

import os

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.data import DataProcessor

MODEL_PATH = "models/model.keras"
PROCESSOR_PATH = "models/processor.pkl"


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PROCESSOR_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    processor = DataProcessor.from_file(PROCESSOR_PATH)
    return model, processor


def get_top_suggestions(
    model,
    processor: DataProcessor,
    seed_text: str,
    top_k: int,
    temperature: float,
) -> list[tuple[str, float]]:
    tokens = processor.tokenizer.texts_to_sequences([seed_text.lower().strip()])[0]
    if not tokens:
        return []

    padded = pad_sequences(
        [tokens],
        maxlen=processor.max_sequence_len - 1,
        padding="pre",
    )
    probs = model.predict(padded, verbose=0)[0]

    if temperature != 1.0:
        log_probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(log_probs)
        probs /= probs.sum()

    top_indices = np.argsort(probs)[-top_k:][::-1]
    index_to_word = {v: k for k, v in processor.tokenizer.word_index.items()}

    suggestions = []
    for idx in top_indices:
        word = index_to_word.get(int(idx), "")
        if word and word != "<OOV>":
            suggestions.append((seed_text.strip() + " " + word, float(probs[idx])))
    return suggestions


def predict_continuation(
    model,
    processor: DataProcessor,
    seed_text: str,
    num_words: int,
    temperature: float,
) -> str:
    index_to_word = {v: k for k, v in processor.tokenizer.word_index.items()}
    text = seed_text.lower().strip()

    for _ in range(num_words):
        tokens = processor.tokenizer.texts_to_sequences([text])[0]
        if not tokens:
            break

        padded = pad_sequences(
            [tokens],
            maxlen=processor.max_sequence_len - 1,
            padding="pre",
        )
        probs = model.predict(padded, verbose=0)[0]

        if temperature != 1.0:
            log_probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(log_probs)
            probs /= probs.sum()

        next_idx = int(np.argmax(probs))
        next_word = index_to_word.get(next_idx, "")
        if not next_word or next_word == "<OOV>":
            break
        text = text + " " + next_word

    return text


def main():
    st.set_page_config(
        page_title="ML Autofill",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ML-Powered Text Autofill")
    st.markdown(
        "An LSTM-based next-word prediction engine. "
        "Type a phrase and get real-time completions powered by a neural network."
    )

    model, processor = load_artifacts()

    if model is None:
        st.error(
            "No trained model found. Run `python train.py` to train the model first."
        )
        st.code("python train.py --epochs 200")
        return

    with st.sidebar:
        st.header("Settings")
        num_words = st.slider("Words to generate", min_value=1, max_value=20, value=5)
        top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Lower values produce more predictable output; higher values add diversity.",
        )
        st.markdown("---")
        st.caption(f"Vocabulary size: {processor.vocab_size}")
        st.caption(f"Max sequence length: {processor.max_sequence_len}")

    if "text_value" not in st.session_state:
        st.session_state.text_value = ""

    col1, col2 = st.columns([3, 2])

    with col1:
        seed_text = st.text_input(
            "Start typing...",
            key="text_value",
            placeholder="e.g. what is machine learning",
        )

    if seed_text.strip():
        with col2:
            st.subheader("Next-word suggestions")
            suggestions = get_top_suggestions(
                model, processor, seed_text, top_k=top_k, temperature=temperature
            )
            if suggestions:
                for suggestion, confidence in suggestions:
                    next_word = suggestion[len(seed_text.strip()):].strip()
                    label = f"{next_word}  —  {confidence:.1%}"
                    if st.button(label, use_container_width=True, key=suggestion):
                        st.session_state.text_value = suggestion
                        st.rerun()
            else:
                st.info("No suggestions found for this input.")

        st.subheader("Predicted continuation")
        with st.spinner("Generating..."):
            continuation = predict_continuation(
                model, processor, seed_text, num_words=num_words, temperature=temperature
            )
        st.markdown(
            f'<div style="background:#f0f2f6;padding:16px;border-radius:8px;'
            f'font-size:1.1rem;line-height:1.6">{continuation}</div>',
            unsafe_allow_html=True,
        )

        st.subheader("All suggestions at a glance")
        if suggestions:
            cols = st.columns(min(len(suggestions), 5))
            for i, (suggestion, confidence) in enumerate(suggestions[:5]):
                next_word = suggestion[len(seed_text.strip()):].strip()
                with cols[i]:
                    st.metric(label=f"Option {i+1}", value=next_word, delta=f"{confidence:.1%}")


if __name__ == "__main__":
    main()
