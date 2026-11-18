# MinGram demonstration: decision report

**Recommendation:** {decision}

## What was measured

A single shared multilingual vocabulary trained on **16,000 synthetic records**, evaluated on **3,200 different records**: 400 per language. BPE, Unigram, and MinGram each have **512 total pieces**, including special tokens. MinGram starts from a 589-piece BPE seed and uses two hard-EM passes.

Lower token counts are better. Savings mean `100 × (1 − MinGram tokens / baseline tokens)`; negative values are regressions.

## Overall result

{summary}

## Breakdown

| Synthetic language group | BPE | Unigram | MinGram | Savings vs BPE | Savings vs Unigram |
|---|---:|---:|---:|---:|---:|
{rows}

## Decision risks

{regressions}

{changes}

Equal-size and matching-normalization checks: **{comparable}**. Zero unknown IDs and exact raw ID round trips for all three models: **{lossless}**.

## Where the text came from

The word inventories and templates were authored for this demo; no text was downloaded. A generator with random seed **20261118** selects different subject/verb/object/ending/number combinations. Each group contributes 2,000 training and 400 held-out records. The actual text, generator, and SHA-256 manifest are committed under `demos/mingram/`. Language names are used to report results; MinGram receives no labels or weights.

Train/test records do not overlap, but use the same small vocabularies and templates. Some combinations are unnatural. This is a controlled compositional test, not a benchmark of natural-language generalization, dialect coverage, or all supported languages.

## Reproduce from a fresh clone

From the repository root:

```sh
uv run --locked demos/mingram/run.py
```

The script uses locked Python dependencies, verifies committed corpus checksums, builds the Rust executable, trains the baselines/seed, and invokes the Rust MinGram trainer and evaluator. Each execution creates a fresh directory under `target/mingram-demo/` and prints its `REPORT.md` location. Models, full JSON reports, and ID mappings remain in that directory. No corpus download or prior model is needed. Rust/Cargo, a C++ compiler, and uv are prerequisites; CMake and Python packages are supplied automatically.

All runs use identity normalization, preserved whitespace, full character coverage, no byte fallback, a maximum piece length of 16 code points, one native baseline-training thread, and no input sampling. The generator can be rerun with `uv run demos/mingram/generate.py`; it recreates the committed text.

## What this does not establish

No downstream model was trained. Token savings do not establish improved accuracy, linguistic boundaries, or end-to-end latency. This is one corpus and one initialization; no confidence interval or robustness sweep was performed. The production vocabulary size of 8,192 was not tested here. The next adoption decision should use representative, untouched text and inspect source/language regressions as well as aggregate savings.

## Evidence

- [Full comparison against BPE](results/vs-bpe.json)
- [Full comparison against Unigram](results/vs-unigram.json)

MinGram implementation: `4211799`. Fresh-run artifacts are authoritative if library/platform behavior differs.
