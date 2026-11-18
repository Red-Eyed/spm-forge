# MinGram demonstration log

Implementation evaluated: `4211799` (standard SentencePiece export and Rust hard EM).

The motivating question was whether the new trainer can demonstrate useful compression,
and whether an end user can reproduce that evidence from a fresh clone.

The experiment fixes a shared vocabulary budget of 512 pieces, one authored synthetic
eight-language corpus, and disjoint held-out records. It compares standard BPE and Unigram
against the Rust MinGram adaptation, using actual standard-runtime token IDs and exact
reconstruction checks. The committed generator, input text, SHA-256 manifest, script lock,
Markdown template, decision report, and JSON snapshots make the experiment reviewable.

The locked SentencePiece 0.2.3 wheel run produced 39,206 BPE tokens, 56,344 Unigram tokens,
and 31,429 MinGram tokens on 3,200 held-out records. MinGram saved 19.84% versus BPE and
44.22% versus Unigram overall, with zero unknown IDs or raw reconstruction failures.
Chinese regressed by 68.34% versus Unigram despite improving by 13.01% versus BPE.
The decision report surfaces this regression instead of recommending unconditional adoption.

An earlier exploratory run used the project's Git-built Python SentencePiece rather than
the locked wheel and emitted five additional Unigram tokens in the Japanese group.
The committed report uses the locked wheel result; the exploratory run is not mixed into it.

The corpus has a small, repetitive template grammar. Natural-text compression, dialect
coverage, behavior at 8,192 pieces, downstream model quality, latency, and statistical
robustness remain unproven. The next evaluation should use representative untouched data.

Fresh-clone verification used a separate local checkout of `392aea8`, with no existing
model artifacts or build directory. `uv run --locked demos/mingram/run.py --quiet` built
the binary, verified the committed text, trained the models, and generated the report.

Configuration integration adds `sentencepiece.model_type: "mingram"` to the ordinary
training workflow, with optional positive `mingram.iterations` defaulting to 2. A tagged
algorithm value keeps MinGram options out of the BPE configuration. The Python bridge
receives BPE seed requests only. Synthetic end-to-end testing verifies the config-driven
seed, Rust optimizer, final model, ID mapping, and training reports.

## NFC versus NFD

The normalization experiment uses NFC and NFD copies of the same committed text, with
512-piece final models and 589-piece BPE seeds for MinGram. Normalization is external
Python preprocessing; exported SentencePiece models retain identity normalization.
The original source text and training algorithm are unchanged.

MinGram emits 31,429 tokens under NFC and 34,797 under NFD: 10.72% more with NFD.
BPE also emits more tokens under NFD. The training alphabet shrinks from 240 to 227
code points, showing that alphabet size alone does not predict output token count.
Repeating at a 64-code-point piece limit yields the same MinGram totals as the default
16-code-point limit. Thus this particular difference is not removed by that control.

All tested models exactly reconstruct their normalized input and emit no unknown IDs.
NFC changes no original held-out records; NFD changes 1,098. Original text preservation
and normalized reconstruction are reported separately. Some groups, including Thai,
are unchanged by NFC/NFD but acquire different token counts as the shared vocabulary
changes. The report and structured evidence are committed beside this log.

Small Unigram count variations can occur between runs, including under the locked wheel;
the report records one actual run rather than promising bit-identical learned vocabularies.
Natural-text effects, deployment normalization support, and downstream quality remain
unproven. Both the original demo and normalization runner were executed after extracting
their shared training/evaluation helpers.


## Default profile and automatic held-out reports

The default profile now selects general multilingual MinGram at 8,192 total pieces, byte
fallback, NFC corpus preparation, and natural corpus frequencies without balancing. An
explicit soft-hyphen preservation policy avoids inheriting OCR-specific character mappings.
Native seed and baseline training sample at most one million training chunks by default.

Canonical original records are content-hashed into a deterministic approximately 95/5 split
before chunking. Exact canonical duplicates and all chunks of a parent remain together.
Only the training side enters balancing, BPE seed training, MinGram, or matched-budget BPE
training. Checksummed manifests are verified for train-only reuse. Similar records, documents,
and common substrings are not globally deduplicated. The disabled-balancing copy now streams
records rather than retaining entire shards.

Automatic Markdown and JSON reports include actual token savings, code points/token,
nearest-rank P50/P95 and mean tokens/record, byte fallback, unknowns, reconstruction failures,
vocabulary usage, per-shard results, hashes, and effective settings. Evaluation uses bounded
histograms and ID sets, merging global statistics before finalizing per-file summaries.
Round-trip checks refer to prepared test chunks; full Unicode NFC is not embedded in the
exported identity-normalized model.

The full CLI completed with 8,192-piece models on the committed synthetic corpus augmented
with 20,000 deterministically generated random Latin records to provide enough seed candidates.
It evaluated 1,770 held-out records with zero unknown IDs or reconstruction failures.
Train-only with a fresh model prefix succeeded and preserved the saved split byte-for-byte.
The unaugmented repetitive demo cannot fill an 8K vocabulary: native BPE explicitly rejects
the requested seed size. This is a workflow check, not natural-language compression evidence.
The small Python-backed integration test also verifies automatic matched-size evaluation and
Markdown generation. Natural-language quality, downstream accuracy, and latency remain unproven.
