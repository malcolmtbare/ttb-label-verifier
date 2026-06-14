"""Evaluation harness.

Runs a chosen model over a folder of ground-truth records and reports per-field
accuracy and latency. This is the "senior move" the project leans on: instead of
eyeballing two labels, you pressure-test against real TTB COLA records and report
*measured* numbers — never invented ones.

Ground-truth layout (eval/ground_truth/):
  <ttb_id>.jpg              the label image
  <ttb_id>.json            {"application": {...}, "expected_warning_status": "pass"|"fail"}

The "application" block is the real COLA form data (the ground truth). Use the
form fields as truth, NOT another model's OCR output, or you are grading AI with AI.

Usage:
  python -m eval.run_eval --model azure-openai            # all records
  python -m eval.run_eval --model gemini --limit 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Allow running this file directly (python run_eval.py) from inside eval/, not just
# as a module from the repo root — put the repo root on the path either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

# Load .env the same way the web app does, so model credentials are available when
# running the harness from the command line.
load_dotenv()

from app.adapters.base import get_adapter  # noqa: E402
from app.schema import ApplicationData, Status  # noqa: E402
from app.verifier import verify  # noqa: E402
from app import matching  # noqa: E402

GT_DIR = Path(__file__).parent / "ground_truth"
_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _records(limit: int | None):
    metas = sorted(GT_DIR.glob("*.json"))
    if limit:
        metas = metas[:limit]
    for meta in metas:
        img = next((meta.with_suffix(ext) for ext in _MEDIA if meta.with_suffix(ext).exists()), None)
        if img:
            yield img, json.loads(meta.read_text())


def run(model_key: str, limit: int | None, show_misses: int = 0) -> None:
    adapter = get_adapter(model_key)
    # per field: counts of pass / review / fail
    stats: dict[str, dict[str, int]] = {}
    misses: dict[str, list[tuple[str, str]]] = {}
    latencies: list[int] = []
    n = 0

    def tally(field: str, status: Status, expected: str = "", detected: str = "") -> None:
        s = stats.setdefault(field, {"pass": 0, "review": 0, "fail": 0})
        if status is Status.PASS:
            s["pass"] += 1
        elif status is Status.REVIEW:
            s["review"] += 1
        else:
            s["fail"] += 1
            misses.setdefault(field, []).append((expected, detected))

    for img, meta in _records(limit):
        application = meta.get("application", {}) or {}
        expected = ApplicationData.model_validate(application)
        # Only score fields that actually have a ground-truth value for this record.
        scored = {k for k, v in application.items() if v}
        warn_expected = meta.get("expected_warning_status")
        product_name = meta.get("product_name")  # acceptable alternative for brand

        data = img.read_bytes()
        t0 = time.perf_counter()
        try:
            extraction = adapter.extract(data, _MEDIA[img.suffix.lower()])
        except Exception as e:
            print(f"  ! {img.name}: extraction failed ({e})")
            continue
        latencies.append(int((time.perf_counter() - t0) * 1000))
        report = verify(extraction, expected, adapter.info.label, latencies[-1])
        n += 1

        for f in report.fields:
            if f.field == "government_warning":
                if warn_expected is None:
                    continue
                got_pass = f.status in (Status.PASS, Status.REVIEW)
                ok = (warn_expected == "pass" and got_pass) or (warn_expected == "fail" and f.status is Status.FAIL)
                tally("government_warning", Status.PASS if ok else Status.FAIL, warn_expected, f.detected or "")
                continue
            if f.field not in scored:
                continue
            status = f.status
            # A label's "brand" may be the company name or the fanciful product name.
            # ColaCloud's BRAND_NAME is often the company; accept the product name too.
            if f.field == "brand_name" and status is Status.FAIL and product_name:
                alt = matching.compare_fuzzy("brand_name", "Brand", product_name, f.detected)
                if alt.status in (Status.PASS, Status.REVIEW):
                    status = alt.status
            tally(f.field, status, f.expected or "", f.detected or "")

    if n == 0:
        print("No records found. Populate eval/ground_truth/ with <id>.jpg + <id>.json.")
        return

    print(f"\nModel: {adapter.info.label}   ({adapter.info.boundary})")
    print(f"Records evaluated: {n}\n")
    print(f"{'Field':<20}{'Pass':>7}{'Review':>8}{'Fail':>7}{'n':>6}{'Match%':>9}")
    print("-" * 57)
    for field in sorted(stats):
        s = stats[field]
        tot = s["pass"] + s["review"] + s["fail"]
        match = (s["pass"] + s["review"]) / tot if tot else 0
        print(f"{field:<20}{s['pass']:>7}{s['review']:>8}{s['fail']:>7}{tot:>6}{match:>8.1%}")
    print("-" * 57)
    print(f"{'Latency p50 (ms)':<20}{statistics.median(latencies):>10.0f}")
    print(f"{'Latency p95 (ms)':<20}{_pctl(latencies, 95):>10.0f}")
    print("\nMatch% counts Pass+Review. Only fields with real ground truth are scored.")
    print("Report these measured numbers in the README. Do not round up.")

    if show_misses:
        print("\n--- sample mismatches (expected   vs  detected) ---")
        for field in sorted(misses):
            print(f"\n{field}:")
            for exp, det in misses[field][:show_misses]:
                print(f"   expected: {exp!r}")
                print(f"   detected: {det!r}")


def _pctl(xs: list[int], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="adapter key, e.g. azure-openai / claude-foundry / gemini")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show-misses", type=int, default=0,
                    help="print up to N expected-vs-detected examples per field that failed")
    args = ap.parse_args()
    run(args.model, args.limit, args.show_misses)