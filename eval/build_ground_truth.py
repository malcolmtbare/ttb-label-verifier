"""Build the evaluation ground-truth set from the ColaCloud sample pack.

Lays out eval/ground_truth/<TTB_ID>.webp + <TTB_ID>.json in the format the eval
harness expects.

IMPORTANT — what counts as ground truth:
  We use only the genuine TTB *application* fields as truth: BRAND_NAME, CLASS_NAME,
  and origin (DOMESTIC_OR_IMPORTED + ORIGIN_NAME). We deliberately DO NOT use
  OCR_ABV / OCR_VOLUME or any LLM_* column as truth — those are themselves extracted
  by AI (Google Vision / GPT-4o), so scoring our extraction against them would be
  grading AI with AI. ABV and net contents are therefore not in the scored set.

Images are downloaded from the ColaCloud CDN. This must run on a machine with
internet access (the documented URL is https://dyuie4zgfxmt6.cloudfront.net/<id>.webp).

Usage:
  python -m eval.build_ground_truth --src /path/to/sample_pack --limit 150
  python -m eval.build_ground_truth --src . --limit 20 --skip-images   # JSON only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CDN = "https://dyuie4zgfxmt6.cloudfront.net/{image_id}.webp"
OUT = Path(__file__).parent / "ground_truth"


def _origin(row: dict) -> str | None:
    """Genuine TTB origin: domestic -> United States; imported -> the country."""
    di = (row.get("DOMESTIC_OR_IMPORTED") or "").strip().lower()
    origin = (row.get("ORIGIN_NAME") or "").strip()
    if di == "domestic":
        return "United States"
    if di == "imported" and origin:
        return origin.title()
    return None


def build(src: Path, limit: int, skip_images: bool) -> None:
    cola = src / "cola.csv"
    if not cola.exists():
        sys.exit(f"cola.csv not found in {src}")
    OUT.mkdir(parents=True, exist_ok=True)

    with cola.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    written = downloaded = 0
    session = None
    if not skip_images:
        import requests  # only needed when downloading
        session = requests.Session()
        session.headers["User-Agent"] = "ttb-label-verifier-eval/1.0"

    for row in rows:
        if written >= limit:
            break
        if (row.get("APPLICATION_STATUS") or "").strip().lower() != "approved":
            continue
        ttb_id = (row.get("TTB_ID") or "").strip()
        image_id = (row.get("MAIN_TTB_IMAGE_ID") or "").strip()
        brand = (row.get("BRAND_NAME") or "").strip()
        if not (ttb_id and image_id and brand):
            continue

        application = {
            "brand_name": brand,
            "class_type": (row.get("CLASS_NAME") or "").strip() or None,
            "country_of_origin": _origin(row),
        }
        meta = {
            "ttb_id": ttb_id,
            "application": application,
            "product_name": (row.get("PRODUCT_NAME") or "").strip() or None,
            "domestic_or_imported": (row.get("DOMESTIC_OR_IMPORTED") or "").strip() or None,
            "_scored_fields": ["brand_name", "class_type", "country_of_origin", "government_warning"],
            "_excluded": "alcohol_content & net_contents — only AI-OCR values in the source (circular).",
        }

        img_path = OUT / f"{ttb_id}.webp"
        if not skip_images and not (img_path.exists() and img_path.stat().st_size > 0):
            try:
                resp = session.get(CDN.format(image_id=image_id), timeout=20)
                if resp.status_code == 200 and resp.content:
                    img_path.write_bytes(resp.content)
                    downloaded += 1
                else:
                    print(f"  ! {ttb_id}: image HTTP {resp.status_code} — skipping")
                    continue
            except Exception as e:
                print(f"  ! {ttb_id}: download failed ({e}) — skipping")
                continue
        elif not skip_images:
            downloaded += 1  # already present

        (OUT / f"{ttb_id}.json").write_text(json.dumps(meta, indent=2))
        written += 1

    print(f"\nWrote {written} ground-truth records to {OUT}"
          + (f" ({downloaded} images downloaded)" if not skip_images
             else " (JSON only — re-run without --skip-images to fetch images)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("."), help="folder containing cola.csv")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()
    build(args.src, args.limit, args.skip_images)