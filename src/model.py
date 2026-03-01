from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


def build_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int = 100,
    lstm_units: int = 150,
) -> Sequential:
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, input_length=sequence_length - 1),
            LSTM(lstm_units, return_sequences=True),
            Dropout(0.2),
            LSTM(lstm_units),
            Dropout(0.2),
            Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model
