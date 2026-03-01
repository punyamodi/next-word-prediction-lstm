import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class DataProcessor:
    def __init__(self):
        self.tokenizer = Tokenizer(oov_token="<OOV>")
        self.max_sequence_len = 0
        self.vocab_size = 0

    def load_text(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def fit(self, text: str) -> list:
        self.tokenizer.fit_on_texts([text])
        self.vocab_size = len(self.tokenizer.word_index) + 1

        input_sequences = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                input_sequences.append(token_list[: i + 1])

        if input_sequences:
            self.max_sequence_len = max(len(seq) for seq in input_sequences)
        return input_sequences

    def prepare(self, input_sequences: list) -> tuple:
        padded = pad_sequences(
            input_sequences, maxlen=self.max_sequence_len, padding="pre"
        )
        X = padded[:, :-1]
        y = to_categorical(padded[:, -1], num_classes=self.vocab_size)
        return X, y

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "tokenizer": self.tokenizer,
                    "max_sequence_len": self.max_sequence_len,
                    "vocab_size": self.vocab_size,
                },
                f,
            )

    @classmethod
    def from_file(cls, path: str) -> "DataProcessor":
        processor = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        processor.tokenizer = data["tokenizer"]
        processor.max_sequence_len = data["max_sequence_len"]
        processor.vocab_size = data["vocab_size"]
        return processor
