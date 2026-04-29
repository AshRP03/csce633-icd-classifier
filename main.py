"""
Main entry point — runs the full pipeline end-to-end.

Usage:
    python main.py              # full run: pipeline → train → predict
    python main.py --skip-data  # skip data pipeline (reuse existing pseudo_labeled.csv)
    python main.py --predict-only  # run inference only (requires saved checkpoint)
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="ICD codability classifier")
    parser.add_argument("--skip-data",    action="store_true",
                        help="Skip data pipeline and reuse existing pseudo_labeled.csv")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training and run inference from saved checkpoint")
    args = parser.parse_args()

    if not args.predict_only:
        if not args.skip_data:
            print("=" * 60)
            print("STAGE 1–5: Data pipeline")
            print("=" * 60)
            from data_pipeline import run_pipeline
            run_pipeline()
        else:
            print("[main] Skipping data pipeline — using existing pseudo_labeled.csv")

        print("\n" + "=" * 60)
        print("STAGE 6: Training")
        print("=" * 60)
        from train import train_model
        train_model()

    print("\n" + "=" * 60)
    print("STAGE 7: Inference")
    print("=" * 60)
    from predict import run_inference
    run_inference()

    print("\nDone. Prediction CSVs are in the predictions/ folder.")


if __name__ == "__main__":
    main()
