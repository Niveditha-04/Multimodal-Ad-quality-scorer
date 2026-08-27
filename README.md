# Multimodal Ad Quality & Policy Violation Scorer

A portfolio project scoring pet-product advertising creatives (image + text pairs)
for policy violations and quality issues, using frozen CLIP embeddings, a small
trained classifier head, a SQL-backed data layer, a RAG-grounded LLM explanation
layer, and a simulated offline comparison against a rule-based baseline.

**This is a portfolio demonstration, not a production system.** The dataset is
synthetic (real photos, generated ad copy), fine-tuning refers only to a small
classifier head on top of a frozen CLIP backbone, and the "A/B test" is an
offline comparison against cached predictions on a held-out set -- not a live
test with real traffic. Every number in this document comes from a checked-in
results file; none are estimated.

## Architecture

```
data/           Oxford-IIIT Pet photos + synthetic ad copy -> data/dataset.csv
db/             SQLAlchemy schema (ads, scores, policy_categories) + queries.sql
model/          CLIP embedding extraction + classifier head training/eval
api/            FastAPI service: POST /score, GET /stats
rag/            Chroma vector store + Claude-generated explanations
ab_test/        Rule-based baseline + statistical comparison vs. the classifier
mcp_server/     (optional) MCP tool wrapper around the same scoring pipeline
```

Pipeline for one `/score` request: image + text in -> frozen CLIP encodes both
-> L2-normalize and concatenate the two 512-dim embeddings -> trained MLP head
predicts one of 4 labels -> if flagged, the ad text (and image) go to a RAG
retrieval step, which pulls the relevant policy rule and similar past examples
-> that context plus the ad get sent to Claude for a grounded explanation ->
the prediction and explanation are persisted to SQLite and returned.

## Tech stack

Python 3.11 &middot; PyTorch + Hugging Face `transformers` (`openai/clip-vit-base-patch32`)
&middot; FastAPI &middot; SQLAlchemy (SQLite) &middot; ChromaDB &middot;
Anthropic API (`claude-sonnet-4-6`) &middot; scikit-learn &middot; pandas/numpy

## Setup

All commands below (setup, pipeline, and the API) assume you're running from
the project root -- every script uses paths relative to it (`data/`, `db/`,
`model/`), not absolute paths.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Running the full pipeline from scratch

Order matters -- each step reads output from the one before it. All are
idempotent (re-running is safe; scripts either skip or fail loudly on
existing data rather than silently duplicating it).

`data/dataset.csv` and all 480 `data/images/*.jpg` are already committed to
this repo, so the actual minimal path starts at step 2 below --
`build_dataset.py` is only needed if you want to regenerate the dataset from
scratch (e.g. to verify the generation process yourself), and doing so
re-downloads the ~800MB Oxford-IIIT Pet archive even though the built output
is already present.

```bash
# 1. (optional) Regenerate the dataset from scratch instead of using the
#    committed data/dataset.csv + data/images/ -- same fixed seed (42), so
#    this reproduces the identical 480 rows, just slower and bandwidth-heavy
python data/build_dataset.py       # downloads Oxford-IIIT Pet (~800MB) + rebuilds data/dataset.csv

# 2. Data layer
python -m db.load_dataset          # loads 480 ads + 7 policy rules into db/ads.db
python data/check_leakage.py       # verifies no image is reused across rows (informational)
python data/make_split.py          # writes data/split.json: the one split every model below reuses

# 3. Model layer
python -m model.extract_embeddings         # frozen CLIP embeddings -> model/embeddings.npz
python -m model.verify_embedding_space     # confirms embeddings are in CLIP's projected space, not raw pooler output
python -m model.train_classifier --modality image   # ablation
python -m model.train_classifier --modality text    # ablation
python -m model.train_classifier --modality joint    # the actual classifier -> model/classifier_head.pt
python -m model.compute_confidence_intervals

# 4. Populate the DB with real eval predictions (so db/queries.sql has real data)
python -m db.populate_eval_scores

# 5. RAG layer
python -m rag.build_vector_store
python -m rag.verify_retrieval     # sanity-checks retrieval quality (informational)

# 6. API (needs ANTHROPIC_API_KEY in .env for the explanation path)
uvicorn api.main:app --reload

# 7. Simulated offline A/B comparison
python -m ab_test.run_comparison

# 8. (optional) MCP server -- separate entry point, not required for anything above
python -m mcp_server.server
```

`sqlite3 db/ads.db < db/queries.sql` runs the 5 reference queries directly.

## Results

### Classifier (frozen CLIP + trained MLP head)

**CLIP's backbone (`openai/clip-vit-base-patch32`) is never fine-tuned.**
Only a 2-layer MLP head (1024&rarr;128&rarr;4, dropout 0.3) is trained on top
of frozen, L2-normalized CLIP embeddings. This is stated explicitly because
"multimodal fine-tuning" can easily be misread as backbone fine-tuning, which
this project does not do.

384/96 train/test split, stratified by label, grouped by source image (verified
zero image reuse across the 480 rows, both by filename and by content hash), fixed
seed. Full metrics: [`model/results/metrics.json`](model/results/metrics.json).

| | accuracy | approved | policy_violation | misleading | low_quality |
|---|---|---|---|---|---|
| **Test accuracy / recall** | **89.6%** | 0.875 | 0.917 | 0.792 | 1.000 |
| 95% Wilson CI (accuracy) | [81.9%, 94.2%] | | | | |

With only 24 test examples per class, per-class recall numbers carry real
sampling uncertainty -- see [`model/results/confidence_intervals.json`](model/results/confidence_intervals.json)
for the full interval table before treating any single decimal as precise.

### Ablation: does the model actually use both modalities?

Before trusting the joint model, three variants were trained on the *identical*
split: image-only, text-only, and joint (concatenated). The concern going in
was real -- CLIP's zero-shot cat/dog alignment might make `misleading` trivial
for image alone, and a text-only model might nail `policy_violation` without
ever needing the image, in which case "multimodal" would just be marketing.

| Modality | Accuracy | approved | policy_violation | misleading | low_quality |
|---|---|---|---|---|---|
| Image-only | 36.5% | 0.208 | 0.458 | 0.292 | 0.500 |
| Text-only | 68.8% | 0.542 | **1.000** | 0.625 | 0.583 |
| **Joint** | **89.6%** | 0.875 | 0.917 | **0.792** | **1.000** |

Full writeup with statistical caveats: [`model/results/ablation_summary.json`](model/results/ablation_summary.json).

- **`misleading` needs both modalities**: joint (0.792) clearly beats both
  single-modality ablations (image 0.292, text 0.625). Detecting a
  species mismatch is inherently relational -- neither modality alone can
  know what the other one claims.
- **`policy_violation` does NOT need the image**: text-only (1.000) actually
  edges out the joint model (0.917). Reported as-is, not smoothed over --
  concatenating irrelevant image features cost a small amount of recall at
  the margin. A text-only path would likely outperform the joint model on
  this specific category in a real system.
- The three modalities' overall-accuracy 95% CIs do not overlap (image
  [27.5%, 46.4%], text [58.9%, 77.1%], joint [81.9%, 94.2%]), so the ranking
  joint > text > image is statistically solid despite the small test set.
  Individual per-class numbers are not all this robust -- image-only's
  `misleading` recall (0.292) has a CI of [14.9%, 49.2%] that *includes* the
  0.25 random-chance baseline, so that specific point estimate is not
  statistically distinguishable from guessing, even though the underlying
  reasoning (image alone can't see the text) is correct.

### Simulated offline A/B comparison: joint classifier vs. rule-based baseline

**This is an offline comparison against a fixed, cached, labeled test set --
not a live A/B test with real production traffic.** Arm B's predictions were
computed once and reused; Arm A was scored fresh but against the same
never-changes-again 96 ads. Full results: [`ab_test/results/comparison.json`](ab_test/results/comparison.json),
baseline calibration: [`ab_test/calibration_log.md`](ab_test/calibration_log.md).

| | Arm A (rule-based) | Arm B (joint classifier) |
|---|---|---|
| Accuracy | 75.0% | **89.6%** |
| Binary flagged precision | 1.000 | 0.957 |
| Binary flagged recall | 0.667 | 0.917 |
| False positive rate | 0.0% | 12.5% |
| False negative rate | 33.3% | 8.3% |

**Headline finding: Arm A has a hard, structural 0% recall on `misleading`.**
The rule-based baseline is keyword blocklists plus basic image heuristics
(blur detection) -- it has zero mechanism to compare what's pictured against
what the text claims, so every single one of the 24 `misleading` test ads
gets predicted `approved`. This isn't a tuning gap; it's the category a
text/basic-image rule engine structurally cannot address without some form
of image understanding, which is exactly what the multimodal classifier adds.
On the two categories the baseline *can* address (keyword-detectable
`policy_violation`, heuristically-detectable `low_quality`), it hits 100%
recall -- a real rule engine, not a strawman.

Two independent statistical tests agree the difference is real:
- **McNemar's test** (the statistically correct paired test here, not a plain
  chi-square of independence -- both arms score the *same* 96 ads, so their
  outcomes aren't independent samples): &chi;&sup2;=7.04, **p=0.008**.
- **Bootstrap** (10,000 resamples): mean accuracy difference +14.6 points in
  the joint model's favor, 95% CI [5.2%, 24.0%], excludes zero.

### API

`POST /score` (multipart: `image` file + `text` field) returns predicted
label, confidence, model version, and (if flagged) an LLM explanation.
`GET /stats` runs the violation-rate-by-category query against whatever has
actually been scored. Every scored ad is persisted to `db/ads.db`.

### MCP server (Phase 6, optional)

`mcp_server/server.py` exposes the same scoring + explanation pipeline as an
MCP tool (`check_ad_compliance(image_path, ad_text)`), for an MCP-capable
agent to call directly rather than going through HTTP. It's a genuinely
separate entry point -- doesn't import `api/main.py` -- but reuses the same
underlying `model/clip_features.py`, `model/classifier_architecture.py`, and
`rag/explain.py` modules, so both entry points score identically. Run with
`python -m mcp_server.server` (stdio transport).

Note: this directory is named `mcp_server/`, not `mcp/`, deliberately -- an
earlier version used `mcp/` and it silently collided with the installed
`mcp` PyPI package's own name (`import mcp` resolving ambiguously depending
on `sys.path` order, made unambiguous only by accident). Caught and fixed
before it became a real bug.

Tested through the SDK's actual tool-dispatch layer (`server.call_tool(...)`,
not just calling the underlying Python function directly) with both a real
flagged prediction (correct label, real Claude explanation call) and both
error paths (missing image file, empty text) -- confirmed each returns a
clean structured error rather than a raised exception, which was verified
empirically to produce a worse result at this layer (a wrapped
`UnexpectedToolError` with a full traceback) than a clean JSON payload does.

## RAG + LLM explanation layer

7 policy rules (written for this project, not lifted from any real platform)
plus 288 past flagged examples (training-split only -- the 96 test ads are
never in the retrieval corpus, same reasoning as keeping them out of model
training) are embedded in a Chroma vector store.

Retrieval is **hybrid, not pure semantic search**. Pure semantic search over
7 short rule documents was tested first and found unreliable at this corpus
size -- a garbled-spam query ranked the correct rule (`low_quality_creative`,
whose text literally contains the phrase "keyword-stuffed spam") 5th of 7.
Since the classifier's predicted label is already known at request time,
`misleading` and `low_quality` map 1:1 onto a specific rule, which is now
guaranteed-included rather than left to embedding luck; `policy_violation`
(no 1:1 mapping -- it covers 4 different sub-rules) stays on pure semantic
search, which tested correctly for exactly that disambiguation task.

### Known limitation: explanations do not catch classifier errors

This was tested explicitly, not assumed. The classifier is 89.6% accurate --
worth asking what happens on the other 10.4%. Since `predicted_label` (not
ground truth) drives which policy rule gets retrieved, a misclassified ad
retrieves the same confident-looking grounding context as a genuine
violation, and the LLM writes a fluent explanation for it regardless.

Concretely: an ad with an accurate photo of a Birman cat and accurate text
("*A dependable sisal scratching post for cat owners, tested with breeds
including the Birman.*") was misclassified by the model as `misleading`
(confidence 0.73). The explanation layer did not hedge or flag uncertainty.
It produced:

> "This ad was flagged under the **mismatched_creative** policy rule because
> the image shows only a **Birman cat outdoors on grass** -- no scratching
> post or any pet product is visible anywhere in the image... giving the
> buyer no visual representation of what they are actually purchasing."

This is a specific, confident, and **entirely fabricated** claim -- no such
"product must be visible in frame" requirement exists in the actual policy
rule, and the ad is genuinely accurate. The explanation is more convincing
than the underlying prediction deserves. **Explanations are grounded in the
model's predicted category and therefore inherit its ~10% error rate -- a
misclassified ad receives a fluent explanation for the wrong reason, not a
signal of uncertainty a downstream reviewer could act on.** Full evidence:
[`rag/results/misclassification_explanation_test.json`](rag/results/misclassification_explanation_test.json).
Building an independent verification pass (e.g. asking the LLM to also judge
whether the retrieved rule actually applies, rather than assuming it does)
was out of scope for this project but is the natural next step.

## Database

Three tables (`ads`, `scores`, `policy_categories`) via SQLAlchemy. `scores`
accumulates rows from multiple sources across this project -- every query
must filter by `model_version`, and this bit once (see [`db/README.md`](db/README.md)
for the full story, including a bug where an unfiltered query silently
blended two different model arms into a meaningless composite number). Both
the joint classifier and the rule-based baseline use a `model_version`
string tied to a content hash of the file that actually determines their
behavior (`model/model_version.py`, `ab_test/baseline_version.py`), so a
future retrain/retune can't silently collide with the current one.

5 reference queries with real, verified output: [`db/queries.sql`](db/queries.sql).

### Postgres (v2 extension) vs. SQLite (v1 default)

The app defaults to SQLite (`db/ads.db`) with no setup required -- this is
still the easiest path and remains fully supported; nothing about the v1
setup instructions changed. Setting `DATABASE_URL` switches to Postgres:
`db/session.py`'s models are pure SQLAlchemy with no SQLite-specific types
(verified: the identical `db/models.py`, unmodified, produces a correct
Postgres schema), so this is a config change, not a code migration.

```bash
brew install postgresql@16          # or Docker: docker run -e POSTGRES_DB=ad_quality_scorer -p 5432:5432 postgres:16
LC_ALL="en_US.UTF-8" $(brew --prefix postgresql@16)/bin/pg_ctl -D $(brew --prefix)/var/postgresql@16 -l logfile start
createdb ad_quality_scorer
# in .env: DATABASE_URL=postgresql://your-user@localhost:5432/ad_quality_scorer

python -c "from db.session import init_db; init_db()"   # creates the schema
python -m db.migrate_to_postgres                         # copies existing SQLite rows over, if any
```

Used a local Homebrew install here rather than Docker (neither was present
on this machine) -- Docker Desktop would mean installing a large GUI app and
its daemon, whereas Homebrew gives a lightweight CLI-only local service.
Either works; `db/migrate_to_postgres.py` only needs `DATABASE_URL` pointed
at a running Postgres, it doesn't care how that Postgres got there.

**Verified, not assumed**: after migrating, ran all 5 `db/queries.sql`
queries against both databases and diffed the output directly. All 15
result rows across the 5 queries are numerically identical. The raw diff
shows two cosmetic differences, both understood and harmless: (1) tied rows
in queries with no secondary `ORDER BY` key can print in either order
depending on the engine's internal storage order, and (2) SQLite's `ROUND()`
returns a float that drops trailing zeros in text output (`0.962`) while
Postgres's `NUMERIC` preserves fixed decimal places (`0.9620`) -- same value,
different formatting. One real portability bug was caught and fixed along
the way: Postgres has no `round(double precision, integer)` overload (SQLite
accepts it via loose typing; Postgres doesn't), so query 2 needed an
explicit `CAST(... AS NUMERIC)` -- see the comment directly above that query
in `db/queries.sql`. Row counts and values matched exactly after migration
too (490 ads, 202 scores, 7 policy categories on both sides; spot-checked
individual rows including the misclassification-test and MCP-test ads
byte-for-byte).

## Limitations

- **Dataset size**: 480 examples total, 96 per class in the test set. This is
  enough to detect the modality-comparison findings above (their confidence
  intervals don't overlap), but individual per-class recall numbers carry
  real sampling noise -- see the CI tables before treating any single decimal
  as precise.
- **Single train/test split, no cross-validation**: one fixed 384/96 split
  (seed 42), reused by every model for comparability. A different split would
  likely move individual numbers by a few points; the qualitative findings
  (joint > text > image; Arm B > Arm A; explanations inherit classifier
  error) are unlikely to flip, but weren't verified across multiple splits.
- **Dataset is synthetic and narrowly scoped**: real photos (Oxford-IIIT Pet)
  paired with generated ad copy, not real advertising platform data. Scoped
  deliberately to pet-product advertising (not a general "any ad" dataset) to
  keep the policy rules concrete and the labeling logic coherent.
- **`misleading` is species-level, not breed-level**: mismatches are
  dog-vs-cat, not e.g. "toy breed text on a giant breed photo." Fine-grained
  breed mismatch would need a stronger visual discriminator than off-the-shelf
  CLIP embeddings reliably provide.
- **Explanations don't verify predictions** -- see the dedicated section
  above. This is the most important limitation in the project.
- **Rule-based baseline is intentionally simple**: keyword blocklist + two
  lightweight image/text heuristics, calibrated only on the training split.
  It's a fair baseline for what it's designed to catch, not a strawman, but
  it's also not what a real ad-policy team's production rule engine would
  look like (no learned thresholds, no per-market variation, no update cadence).
- **English only, one product vertical**: no multilingual support, no
  evaluation outside pet-product advertising.
- **No live traffic**: the A/B comparison is offline against a fixed labeled
  set, not real user-facing traffic with real business outcomes (click-through,
  appeals, revenue impact) to measure against.
- **Confidence isn't calibrated**: the classifier's softmax output is used as
  "confidence" throughout, but no calibration step (e.g. temperature scaling,
  reliability diagrams) was run to check it's actually well-calibrated.
