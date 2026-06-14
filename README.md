# TTB Label Verifier

A prototype that checks an alcohol-beverage label image against its application
data and against the federally required government health warning. It is built to
be run by a reviewer with a single API key, and to model the agency's real Azure
environment rather than a convenient demo setup.

> Scope note: this is a standalone proof-of-concept, not a COLA integration. It is
> intended to inform a future procurement decision, which is why it deliberately
> lets you compare several models and is explicit about which ones fit the agency's
> compliance boundary.

---

## Deployed application

Live prototype: **`<paste your https://....azurecontainerapps.io URL here>`**

- `/` — single-label review (extract → correct → approve/reject)
- `/batch` — batch triage against an application manifest

Deployed on Azure Container Apps; build and configuration steps are in
[DEPLOY.md](DEPLOY.md).

## Approach, tools, and assumptions

**Approach.** A multimodal LLM reads the label into a structured schema; all pass/fail
matching is then done in **deterministic Python**, never by the model — so every verdict
is auditable and reproducible. Extraction sits behind one adapter interface so models are
swappable, tiered by where inference runs relative to the agency's Azure compliance
boundary. A separate, independent OCR pass (Azure AI Vision) corroborates and locates
each field on the image; it never decides pass/fail.

**Tools.** Python 3.12, FastAPI + Uvicorn, Pydantic v2 (the extraction schema), Azure
OpenAI (in-boundary extraction), Azure AI Vision Read (the locate/zoom overlay), with
optional Gemini (external benchmark) behind the same
interface; Pillow + pillow-heif (HEIC support); pytest. The frontend is dependency-free
HTML/JS. Containerized and deployed on Azure Container Apps.

**Key assumptions.** (1) A COLA application covers one product, so a multi-SKU image is an
input to fix, not a verdict to synthesize. (2) The model extracts; Python decides — the
LLM is never the source of a compliance pass. (3) The eval is scored only against genuine
TTB form fields, never against AI-derived values, which would be circular. (4) External
models are blocked by the production firewall and exist as a local benchmark only; Azure
OpenAI and Vision stay in-boundary. The fuller list lives under
[How it maps to the stakeholders](#how-it-maps-to-the-stakeholders) and
[Known limitations](#known-limitations-prototype-scope).

---

## What it does

Drop a label image. The selected model reads the label and fills in the detected
fields; you correct anything against the image, then **approve or reject** the label.
Switching the model re-reads the same label, so you can compare how two models
extract it without re-uploading.

The single-label screen is an extraction-review-and-sign-off tool: it surfaces what
the model read (one editable box per field), runs the one check that needs no
application data (the government warning), flags each field with an independent OCR
corroboration signal, and records the agent's determination. The label-vs-**application**
comparison (fuzzy brand, numeric ABV, unit-aware volume, country) runs in the
**batch** view against an uploaded manifest and in the eval harness — that's where
expected application values come from. The deterministic matching engine behind that
comparison:

- **Brand name, class/type, name & address** — fuzzy match (case- and
  punctuation-insensitive), so `STONE'S THROW` and `Stone's Throw` are equal but
  real differences still surface.
- **Alcohol content** — numeric comparison (`45% Alc./Vol.` vs `45% ALC/VOL`).
- **Net contents** — unit-aware (`750 mL` vs `0.75 L`).
- **Country of origin** — inferred from a U.S. address when the label doesn't
  state it (see below).
- **Government warning** — checked against the exact legal text (see below); this one
  runs in the single view too, since it needs no application value.
- **Multiple SKUs in one image** (a shelf shot of several different bottles) is
  detected and flagged. The extracted values aren't a single coherent label — across
  several bottles the model fills each field with the clearest value it can read, which
  may come from *different* bottles — so rather than presenting that as one verdict, the
  app shows a clear "multiple products (N)" warning stating the values may span SKUs and
  that the reviewer should upload one label per image. A single COLA application covers
  one product, so this is treated as an input to fix, not a thing to guess at.
- **Junk uploads** (a photo of a dog, a blank page) are detected and rejected with
  a plain message instead of a fabricated comparison.
- **iPhone HEIC/HEIF** photos are accepted; they're converted to JPEG on upload
  (with EXIF orientation baked in) so the models, OCR, preview, and overlay all work.

In comparison contexts each field returns one of **Match**, **Mismatch**, **Review**,
or **N/A**, the overall verdict being the most cautious of these.

### Review workflow & export

The single view deliberately shows **one editable box per field** — the text read
off the label — because there's no way to pre-load application values here (that's
the batch view's job). The agent corrects any field against the image, then clicks
**Finish item review**, which reveals **Approve** / **Reject** to record a
determination. Each finished item — determination, every field's value, the warning
status, which fields the OCR couldn't corroborate, and the model used — is saved to
the session. **Finish session** downloads all of those records as a CSV (one row per
label, Excel-friendly for the team), which is the review log / audit trail.

Consistent with the prototype's no-server-persistence scope, the session lives in the
browser (cleared when the tab closes) and the export happens client-side; a
production build would post each determination to a system of record instead.

### Model/OCR corroboration (a second, independent reader)

Because the app already runs a dedicated OCR pass for highlighting, each field also
gets a cheap confidence signal: did the OCR independently find the model's claimed
text on the label? Agreement ("OCR ✓") is corroboration from a second reader;
absence ("OCR ?") flags the field for a closer look. This is deliberately *not* a
pass/fail gate — OCR routinely misses stylized label fonts the model reads fine, so
"unconfirmed" means "couldn't independently verify," not "the model is wrong." The
per-label export lists any unconfirmed fields. (Note: the signal is corroboration,
not string-equality — the highlighter locates OCR words *by* matching the model's
value, so a naive text-vs-text comparison would be circular and falsely reassuring.)

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in at least one model's credentials
uvicorn app.main:app --reload # http://localhost:8000
pytest                        # 26 tests
```

Or with Docker:

```bash
docker build -t ttb-verifier .
docker run --rm -p 8000:8000 --env-file .env ttb-verifier
```

The model picker shows only models whose credentials are present, so one set of
credentials is enough to run the whole app.

---

## Architecture

```
app/
  schema.py        shared data contract (LabelExtraction, ApplicationData, VerificationReport)
  prompts.py       the one extraction prompt every model receives
  adapters/        one file per model, all behind a single interface
  matching.py      deterministic field comparison (fuzzy / numeric / volume / country)
  warning.py       government-warning verification against the canonical text
  verifier.py      extraction + application data -> report
  main.py          FastAPI endpoints
static/index.html  the review UI (single file, no build step)
eval/run_eval.py   measured per-field accuracy + latency over real records
tests/             unit tests for the matching and warning logic
```

Three decisions drive everything:

**1. The model is behind an adapter.** Every model implements one method,
`extract(image) -> LabelExtraction`. The UI picker and the eval harness only see
that interface, so adding or swapping a model is a local change and the comparison
across models is apples-to-apples (identical prompt and schema). Each adapter also
declares its *boundary* — whether it runs inside Azure or reaches out — which the
UI surfaces.

**2. Matching is deterministic code, not the model.** In a compliance setting an
agent who rejects a label needs a defensible, reproducible reason. The model
extracts text; `matching.py` and `warning.py` decide pass/fail and explain exactly
why ("expected 'birth defects' but found 'health issues'"). That logic is
unit-tested and runs without any API.

**3. Extraction and comparison are separate calls.** Uploading runs the model once
(`/api/verify`). Editing a field or entering application data re-runs only the
matcher (`/api/compare`) — no extra model calls, so correction is instant and free.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | The review UI |
| GET | `/api/models` | Models that are actually configured |
| POST | `/api/verify` | One image + model (+ optional expected) → full report |
| POST | `/api/compare` | Re-run matching only, no model call |
| POST | `/api/ocr` | Locate detected values on the image (async, optional) |
| POST | `/api/batch` | Many images, bounded concurrency (see Batch) |

---

## How it maps to the stakeholders

The interviews are full of constraints disguised as anecdotes. Each one shaped a
decision:

- **Interviewer 1 — "results back in about 5 seconds or nobody uses it."** Single-pass
  extraction with a fast model (GPT-5.4-mini / Gemini 3 Flash); no chained
  multi-agent workflow. The label is read in one call on upload.
- **Interviewer 1 / Interviewer 2 — importers dump 200–300 applications at once.** A dedicated
  batch triage view (`/batch`) processes a drop concurrently and surfaces only the
  exceptions — see Batch processing.
- **Interviewer 3 — "you need judgment; STONE'S THROW vs Stone's Throw is the same thing."**
  Brand and class use normalization + similarity; trivial case/punctuation
  differences pass, genuinely ambiguous ones return **Review** rather than a hard
  fail, and every detected field is editable so the human stays in the loop.
- **Interviewer 5 — the warning must be exact, and "GOVERNMENT WARNING" must be caps/bold.**
  The warning is checked against the canonical text with a caps check; a title-case
  "Government Warning" fails with a reason. (See the warning section for an
  important refinement.)
- **Jenny — labels shot at angles, with glare.** Multimodal models read skewed and
  low-light photos far better than classic OCR; the prompt tells the model to read
  rotated images.
- **Interviewer 3  /Interviewer 5  — wildly varying tech comfort.** One screen: pick a model, the
  fields auto-fill, correct against the image beside them, read a verdict. Status is
  shown with color **and** an icon **and** a word — never color alone.
- **Interviewer 4 — Azure since 2019, a firewall that blocks outbound traffic, FedRAMP.**
  See Models & the Azure boundary.

---

## Models & the Azure boundary

The picker offers up to three models, each labeled by how it relates to the
agency's Azure boundary — because for a tool meant to inform a procurement decision,
the compliance posture of each option *is* part of the answer.

| Model | Boundary | Notes |
|---|---|---|
| **Azure OpenAI** (e.g. GPT-5.4-mini) | In-boundary (Azure) | First-party, runs in your Azure region via the `/openai/v1` endpoint, FedRAMP High in Azure Government. No external egress — directly answers Marcus's firewall concern. |
| **Gemini** | External | Not hosted in Azure; an outbound call to Google. Included as a performance **benchmark**, and labeled external — it would be blocked by the agency's firewall in production. |

Which model is most accurate per field is left to the eval harness, not assumed.
Early observation worth measuring: on a label with no printed bottler address,
GPT-5.4 correctly left the field blank while Gemini tended to infer a city — i.e.
GPT was less prone to hallucinating a value. That is exactly the kind of difference
the harness quantifies.

**Production note.** The prototype runs on these models directly for portability.
In the agency's FedRAMP'd, firewalled environment the in-boundary choice is Azure
OpenAI (no egress);Gemini would require a cross-cloud Vertex
arrangement. The prompt and schema port across models, but behavior differs, so the
eval harness should be re-run against whatever production model is chosen.

---

## The government warning check

The warning is verified against the fixed text required by the Alcoholic Beverage
Labeling Act of 1988 (27 CFR part 16) — not against the application form. The
canonical text lives in `app/warning.py`.

The check enforces what the regulation actually governs:
- **Wording** — compared case- and punctuation-insensitively. An all-caps body is
  permitted, and OCR/vision routinely varies punctuation, so neither should fail.
- **Prefix caps** — the words `GOVERNMENT WARNING` must appear in capitals. This is
  the case rule that exists, and it catches a title-case "Government Warning".
- **Bold** — `GOVERNMENT WARNING` must be bold. A vision model can only give a
  visual heuristic here, not a font-weight measurement, so this is **advisory** and
  never hard-fails on its own.

### What testing changed (a real false positive)

The first version compared the warning to the canonical text case-sensitively.
Tested against a real can — Half Acre Daisy Cutter — it failed the label because the
can prints the whole warning in ALL CAPS while the canonical text is mixed-case.
Checking the regulation showed the body's letter-case isn't governed at all; the
check was enforcing a rule that doesn't exist. It now enforces only the real
requirement, and the exact Half Acre extraction is pinned as a regression test. The
lesson generalizes: validate compliance rules against real labels, and encode the
actual requirement, not a stricter-looking proxy.

---

## Country-of-origin inference

Country of origin is required on labels only for imports, so a domestic product
legitimately won't print it. When the label doesn't state a country but the
name/address is clearly U.S. (a state code in a "City, ST" pattern, or an explicit
USA mention), the tool infers "United States."

This is done in deterministic code (`matching.infer_us_country`), not by the model,
so the derivation is auditable, and the result is labeled "inferred from U.S.
address" — it never claims the label *stated* a country it didn't. USA / U.S. /
United States are treated as equivalent when comparing to the application.

---

## Batch processing

The brief calls for handling large drops (Sarah/Janet: importers submit 200–300 at
once). This is **implemented** — a dedicated triage view at `/batch`, backed by
`POST /api/batch`, which processes the drop concurrently (bounded semaphore) so it
doesn't serialize.

The batch UX is deliberately different from the single-image review, because the
correction form does not scale to 200 labels:

- **Triage table, not a form.** Results render as a table — filename, overall
  verdict, and the specific fields that failed. Filter pills (All / Mismatch /
  Review / Pass) let an agent jump straight to the exceptions. This is the point:
  the tool clears the obvious passes (the "drowning in routine stuff" Sarah
  described) and routes only the problems to a human.
- **Drill-in.** Clicking any row expands it to show the full per-field comparison
  inline, so the agent can see *why* without leaving the table.
- **Application data at scale.** Each label needs its expected values to compare
  against. The batch view accepts an optional **manifest CSV** — a `filename`
  column plus any application fields — and matches each image to its row. The page
  can generate a pre-filled template for the selected files. Without a manifest,
  every label is checked **warning-only** (the warning is mandatory regardless), so
  a drop is still useful with zero setup. In production this data comes from the
  COLA application records; the CSV is the standalone stand-in.

- **Review queue.** After a run, **Review N one-by-one** opens the items in the
  current filter as a queue — each one shown with its actual image and the full
  single-label tooling (editable fields with live re-compare, OCR locate+zoom,
  corroboration badges) plus Approve / Reject and Prev / Next. Because the uploaded
  images are still in the browser from the batch run, the queue carries them without
  any server-side storage. **Finish & download determinations** exports the whole
  batch — comparison results plus the agent's Approve/Reject calls — as one CSV. This
  is the intended compliance loop: triage clears the obvious passes, the human walks
  only the exceptions.

---

## Evaluation

`eval/run_eval.py` runs a chosen model over real TTB COLA records and prints
**measured** per-field accuracy and p50/p95 latency:

```bash
python -m eval.run_eval --model azure-openai --limit 100
```

Ground-truth records go in `eval/ground_truth/` as `<id>.jpg` + `<id>.json` (the
`application` block = the COLA form fields). The free TTB Public COLA Registry and
ColaCloud's free sample pack are good sources.

> Report only numbers this harness actually produces — do not paste aspirational
> figures. And use the **application form fields** as ground truth, not a third
> party's OCR-extracted fields: grading AI extraction against AI extraction is
> circular.

---

## Deploying to Azure

This prototype is deployed on **Azure Container Apps**. The image is built in the cloud
from the `Dockerfile` (no local Docker, and the build stays inside the Azure tenant —
relevant given the outbound firewall). The fastest path is:

```bash
az login                      # complete MFA if prompted
./deploy.sh                   # edit the variables at the top first
```

`deploy.sh` creates the resources, builds + deploys the image, pushes credentials from
your `.env` as Container Apps secrets, and prints the live URL. Full step-by-step (and an
App Service alternative) is in [DEPLOY.md](DEPLOY.md). The default
`*.azurecontainerapps.io` hostname satisfies the "deployed URL we can test" deliverable;
a custom domain is optional and goes last.

### A note on Azure OpenAI via Foundry

GPT models browsed and deployed through the Foundry portal are still served on the
Azure OpenAI endpoint (`https://your-resource.openai.azure.com/openai/v1`) and called
with the standard `OpenAI` client — the v1 API needs no dated `api-version`. 
Set `AZURE_OPENAI_DEPLOYMENT` to the deployment **name** you assigned (case-sensitive),
which may differ from the model's catalog name.

---

## Known limitations (prototype scope)

- **Bold detection** is a visual heuristic from the model, not a font-weight
  measurement. A not-bold warning is flagged **Review**, not auto-failed.
- **Fuzzy matching** handles surface differences, not semantic equivalence
  (abbreviations, alternate spellings); those return Review for a human.
- **On-image field highlighting** is implemented via a dedicated OCR pass (Azure AI
  Vision Read), not the extraction model: the model gives the text, OCR gives true
  word coordinates, and each field value is fuzzy-aligned to the word run that spells
  it. It runs **asynchronously** — the fast extraction renders first, then each field
  gets a "locate" button that highlights it on the image a moment later. An earlier
  attempt using model-estimated boxes was scrapped as too imprecise; this is the
  reliable version. It's **optional**: if `AZURE_VISION_*` is unset the app behaves
  identically minus the overlay. Vision is a first-party Azure service, so it stays
  in-boundary. "Locate" does two things at once: it highlights the region on the full
  image *and* renders a **magnified, auto-straightened crop** of it underneath, so
  small text on a full-bottle shot is actually readable. The crop is rotation-aware —
  it reads the text angle from the OCR word polygons and rotates the region upright, so
  a warning printed vertically (common on cans) reads normally. The warning itself is
  located by a rotation-robust method (it unions every OCR word belonging to the
  warning rather than matching a long string in order). Directly serves Jenny's
  "labels shot at bad angles" concern.
- **Type-size and "contrasting background"** rules from 27 CFR part 16 are out of
  scope; they need physical dimensions, not just pixels.
- **No persistence / auth / PII handling** — stateless proof-of-concept, per the brief.

---

## Testing

`pytest` runs the unit suite (20 tests) covering the deterministic logic: fuzzy
matching (incl. Dave's case), ABV and volume normalization, the warning rules (incl.
the all-caps regression and the title-case failure), and U.S.-origin inference.
These run without any API key.

## TTB reference

The required health warning text and formatting rules come from the Alcoholic
Beverage Labeling Act of 1988 (27 CFR part 16); see ttb.gov. The canonical text and
the rules actually enforced live in `app/warning.py`.
