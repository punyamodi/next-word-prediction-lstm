import argparse
import os
import sys


DEFAULT_MODEL = "models/model.keras"
DEFAULT_PROCESSOR = "models/processor.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text completions using the trained LSTM model")
    parser.add_argument("seed", type=str, help="Seed text to complete")
    parser.add_argument("--words", type=int, default=5, help="Number of words to generate")
    parser.add_argument("--top-k", type=int, default=5, help="Number of suggestions to show")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to trained model")
    parser.add_argument("--processor", type=str, default=DEFAULT_PROCESSOR, help="Path to saved processor")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model) or not os.path.exists(args.processor):
        print("Error: trained model not found. Run 'python train.py' first.")
        sys.exit(1)

    from src.predictor import Predictor

    predictor = Predictor(args.model, args.processor)

    print(f"\nSeed: {args.seed}")
    print("-" * 50)

    continuation = predictor.predict_continuation(
        args.seed, num_words=args.words, temperature=args.temperature
    )
    print(f"Continuation: {continuation}")

    print(f"\nTop {args.top_k} next-word suggestions:")
    suggestions = predictor.get_top_suggestions(
        args.seed, top_k=args.top_k, temperature=args.temperature
    )
    for i, (text, confidence) in enumerate(suggestions, 1):
        next_word = text[len(args.seed):].strip()
        print(f"  {i}. {next_word:20s}  ({confidence:.2%})")


if __name__ == "__main__":
    main()
