from __future__ import annotations

import argparse
import json
import sys

from .inference import predict_image
from .models import MODEL_SPECS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MediFlow 단일 이미지 분류")
    parser.add_argument("model_type", choices=MODEL_SPECS, help="사용할 분류 모델")
    parser.add_argument("image", help="분류할 이미지 경로")
    parser.add_argument(
        "--variant", choices=("original", "augmented"), default="augmented", help="모델 버전"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = predict_image(args.image, args.model_type, args.variant)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
