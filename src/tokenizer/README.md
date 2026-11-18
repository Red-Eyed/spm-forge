# Tokenizer optimization and evaluation

These modules operate on standard SentencePiece models and UTF-8 corpus records, independently
of corpus repair, language identification, balancing, and OCR configuration.

- `model.rs` validates models and loads the standard native processor. Rewriting changes
  scores, piece order, model type, and vocabulary size. It preserves raw normalization rules,
  denormalization rules, unknown protobuf extensions, and retained piece metadata. Stale
  segmentation self-tests are removed.
- `wire.rs` implements selective protobuf field replacement using prost's wire decoder, so
  importing a model from a newer SentencePiece schema does not discard unknown fields.
- `records.rs` replays one file at a time. Only LF/CRLF delimiters are removed. Invalid UTF-8,
  excessive record size, encoding errors, and cancellation are explicit failures.
- `evaluate.rs` accumulates actual emitted-ID counts and reconstruction diagnostics. It uses
  no model-piece likelihood as a proxy for compression or downstream quality.
- `mingram.rs` runs hard EM, ranks removable pieces using full-precision integer usage counts,
  pins non-normal IDs, preserves character coverage, and exports a standard Unigram model.
  The CLI supplies paths and output policy; existing seed generation remains in the trainer bridge.

Training follows [MinGram](https://arxiv.org/abs/2606.27019): count tokens under a
token-count-dominated score, update log frequencies, then perform flat score pruning. The
reference uses SCRIPT atoms and its own encoder; this implementation retains SentencePiece's
normalizer and runtime. Initial probabilities are measured by encoding the supplied seed model.
All corpus occurrences have their natural frequency; no language-specific weights are applied.

The effective normal-piece score is `-1 + 0.00001 * log_probability`. This single scalar is
compatible with stock Unigram inference, but its probability term is not an exact
lexicographic tie-break for unbounded input lengths. Float32 rounding can also merge distinct
scores. Pruning uses usage counts rather than rounded exported scores to avoid that loss of
ranking precision. Unused characters keep a finite floor so coverage is not dropped.

Tests use self-contained multilingual protobuf fixtures, enumerate tiny fixed-vocabulary
segmentations, and exercise unknowns, byte fallback, metadata retention, reserved IDs, and
bounded input. They demonstrate correctness and controlled compression gains, not a general
quality improvement. `just check` also runs `python_runtime_compatibility`: it trains a small
BPE seed with the unmodified Python runtime, optimizes it in Rust, and verifies the exported
model's token counts and exact reconstruction through Python. This interoperability test is
ignored by default in plain `cargo test` because it requires the uv-managed Python environment.

For evaluation, the primary ratio is `candidate_tokens / baseline_tokens`; savings are
`100 * (1 - ratio)`. Empty baselines have explicit undefined results. Raw reconstruction compares
against original records; normalized reconstruction compares both sides after applying the
model normalizer. Different normalization bytes, denormalization bytes, whitespace-suffix
behavior, or user-defined symbol sets are rejected conservatively. Model hashes identify the
exact artifacts. Equal vocabulary size and zero information loss are necessary before treating
lower held-out counts as an algorithm improvement. Downstream model quality requires separate
evaluation.


The configured training workflow now splits canonical records before chunking and evaluates
on saved held-out shards. `report.rs` renders its paired result as Markdown and JSON with split
and configuration provenance. The evaluator keeps a histogram bounded by the configured
record limit and a set bounded by vocabulary size, rather than a token count per input record.
Global percentiles merge histograms; global utilization takes the union of IDs, not the sum of
per-shard counts. Code points/token uses prepared source code points, and P50/P95 use nearest
rank. Automatic training compares MinGram with a separately fitted BPE model at the same total
vocabulary budget; a BPE configuration compares its model with itself.
