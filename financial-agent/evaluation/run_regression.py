"""Regression test runner — evaluates agent on a fixed set of FinQA questions."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=0.70)
    args = parser.parse_args()

    # TODO: load test cases, run agent, compute exact-match accuracy
    accuracy = 0.0
    print(f"Accuracy: {accuracy:.2%} (threshold: {args.min_accuracy:.2%})")

    if accuracy < args.min_accuracy:
        raise SystemExit(f"FAIL: accuracy {accuracy:.2%} below threshold {args.min_accuracy:.2%}")


if __name__ == "__main__":
    main()
