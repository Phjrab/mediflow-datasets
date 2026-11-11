from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation import evaluate_dataset
from .models import MODEL_SPECS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MediFlow 모델 평가 재현")
    parser.add_argument("model_type", choices=MODEL_SPECS, help="평가할 분류 모델")
    parser.add_argument("dataset", type=Path, help="클래스별 하위 폴더가 있는 test 폴더")
    parser.add_argument(
        "--variant", choices=("original", "augmented"), default="augmented", help="모델 버전"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, help="평가 JSON 저장 경로")
    parser.add_argument("--overwrite", action="store_true", help="기존 출력 JSON 덮어쓰기")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.output and args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output}. Use --overwrite to replace it."
        )

    result = evaluate_dataset(args.model_type, args.dataset, args.variant, args.batch_size)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
