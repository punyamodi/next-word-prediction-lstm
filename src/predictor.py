from __future__ import annotations

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.data import DataProcessor


class Predictor:
    def __init__(self, model_path: str, processor_path: str):
        self.model = load_model(model_path)
        self.processor = DataProcessor.from_file(processor_path)
        self._index_to_word = {
            v: k for k, v in self.processor.tokenizer.word_index.items()
        }

    def _predict_next_token(self, token_list: list, temperature: float = 1.0) -> tuple[str, float]:
        padded = pad_sequences(
            [token_list],
            maxlen=self.processor.max_sequence_len - 1,
            padding="pre",
        )
        probs = self.model.predict(padded, verbose=0)[0]

        if temperature != 1.0:
            log_probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(log_probs)
            probs /= probs.sum()

        best_idx = int(np.argmax(probs))
        word = self._index_to_word.get(best_idx, "")
        return word, float(probs[best_idx])

    def predict_continuation(
        self, seed_text: str, num_words: int = 5, temperature: float = 1.0
    ) -> str:
        text = seed_text.lower().strip()
        for _ in range(num_words):
            tokens = self.processor.tokenizer.texts_to_sequences([text])[0]
            if not tokens:
                break
            next_word, _ = self._predict_next_token(tokens, temperature)
            if not next_word or next_word == "<OOV>":
                break
            text = text + " " + next_word
        return text

    def get_top_suggestions(
        self, seed_text: str, top_k: int = 5, temperature: float = 1.0
    ) -> list[tuple[str, float]]:
        tokens = self.processor.tokenizer.texts_to_sequences([seed_text.lower().strip()])[0]
        if not tokens:
            return []

        padded = pad_sequences(
            [tokens],
            maxlen=self.processor.max_sequence_len - 1,
            padding="pre",
        )
        probs = self.model.predict(padded, verbose=0)[0]

        if temperature != 1.0:
            log_probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(log_probs)
            probs /= probs.sum()

        top_indices = np.argsort(probs)[-top_k:][::-1]
        suggestions = []
        for idx in top_indices:
            word = self._index_to_word.get(int(idx), "")
            if word and word != "<OOV>":
                suggestions.append((seed_text.strip() + " " + word, float(probs[idx])))
        return suggestions
