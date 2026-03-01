import argparse
import os

import matplotlib.pyplot as plt

from src.data import DataProcessor
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LSTM autofill model")
    parser.add_argument("--data", type=str, default="data/faqs.txt", help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--lstm-units", type=int, default=150, help="LSTM hidden units")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor = DataProcessor()
    text = processor.load_text(args.data)
    input_sequences = processor.fit(text)
    X, y = processor.prepare(input_sequences)

    print(f"Vocabulary size: {processor.vocab_size}")
    print(f"Max sequence length: {processor.max_sequence_len}")
    print(f"Training samples: {len(input_sequences)}")

    model = build_model(
        vocab_size=processor.vocab_size,
        sequence_length=processor.max_sequence_len,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
    )
    model.summary()

    history = model.fit(X, y, epochs=args.epochs, verbose=1)

    model_path = os.path.join(args.output_dir, "model.keras")
    processor_path = os.path.join(args.output_dir, "processor.pkl")

    model.save(model_path)
    processor.save(processor_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"])
    axes[0].set_title("Accuracy vs Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")

    axes[1].plot(history.history["loss"])
    axes[1].set_title("Loss vs Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    plt.tight_layout()
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()

    print(f"Model saved to {model_path}")
    print(f"Processor saved to {processor_path}")
    print(f"Training curves saved to {curves_path}")


if __name__ == "__main__":
    main()
