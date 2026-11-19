# MinGram demonstration: decision report

**Recommendation:** Worth evaluating against BPE on representative text; this synthetic result does not establish production quality.

## What was measured

A single shared multilingual vocabulary trained on **16,000 synthetic records**, evaluated on **3,200 different records**: 400 per language. BPE, Unigram, and MinGram each have **512 total pieces**, including special tokens. MinGram starts from a 589-piece BPE seed and uses two hard-EM passes.

Lower token counts are better. Savings mean `100 × (1 − MinGram tokens / baseline tokens)`; negative values are regressions.

## Overall result

| Tokenizer | Held-out tokens | MinGram saves |
|---|---:|---:|
| BPE | 39,206 | 19.84% |
| Unigram | 56,344 | 44.22% |
| MinGram | 31,429 | — |

## Breakdown

| Synthetic language group | BPE | Unigram | MinGram | Savings vs BPE | Savings vs Unigram |
|---|---:|---:|---:|---:|---:|
| Arabic | 5,376 | 7,741 | 4,040 | 24.85% | 47.81% |
| Chinese | 3,166 | 1,636 | 2,754 | 13.01% | -68.34% |
| English | 5,159 | 9,535 | 4,306 | 16.53% | 54.84% |
| French | 6,061 | 8,963 | 5,474 | 9.68% | 38.93% |
| Hindi | 5,925 | 10,585 | 4,316 | 27.16% | 59.23% |
| Japanese | 3,676 | 3,931 | 3,182 | 13.44% | 19.05% |
| Korean | 4,630 | 6,094 | 3,168 | 31.58% | 48.01% |
| Thai | 5,213 | 7,859 | 4,189 | 19.64% | 46.70% |

## Decision risks

**Do not replace Unigram unconditionally.** Aggregate gains hide regressions: Chinese: 68.34% more tokens than Unigram.

Against BPE: 2,292 records improved, 908 were unchanged, and 0 became longer.

Against Unigram: 2,677 records improved, 6 were unchanged, and 517 became longer.

Equal-size and matching-normalization checks: **PASS**. Zero unknown IDs and exact raw ID round trips for all three models: **PASS**.

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
