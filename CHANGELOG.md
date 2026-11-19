# Changelog

## 0.5.0

### Highlights

- Train multilingual MinGram tokenizers through the normal configuration workflow and
  export standard SentencePiece models.
- Compare token compression and reconstruction on a saved held-out split, with automatic
  Markdown and JSON decision reports.
- Resume interrupted MinGram counting from completed input files and reuse verified seeds,
  models, and completed evaluation.

### Backwards Incompatible Changes

#### Default configuration and prepared data

The default profile changes from `ocr_multilingual` to `multilingual_mingram`: an 8,192-piece
MinGram vocabulary with byte fallback, NFC preparation, and no corpus balancing. Select
`ocr_multilingual` explicitly to retain the legacy BPE/OCR preset settings. A different
tokenizer can change token IDs and segmentation; adapt downstream models before using it.

All configured training runs now reserve approximately 5% of canonical original records for
testing before chunking. Exact canonical duplicates share a partition, and all chunks of a
parent record stay together. Balancing and model fitting use only training data. A split
with an empty partition fails explicitly. Configure `evaluation.test_fraction` and
`evaluation.split_seed` when necessary; this split also applies to legacy BPE presets.

`train-only` requires `reports/split.json` and verifies corpus checksums and preparation
settings. Older prepared runs without this manifest need one preparation run in a new work
directory. Compatible runs with a saved split can adopt an existing completed BPE seed;
MinGram work performed by versions without count checkpoints cannot be recovered.

#### Preset catalog filename

Rename custom references to `cfg.json.ocr` to `cfg.presets.json`. The CLI defaults and bundled
references use the new filename. `cfg.json` and `cfg.mingram.json` retain their names.

#### Python installation

The Python trainer now uses the published SentencePiece `0.2.3` wheel instead of a moving
upstream Git source. Source-build fallback is disabled for that package, so installation
requires a compatible wheel. The Rust binding retains its existing native runtime; this
release does not align the two native versions or introduce another native build.

### New Features

- Add standalone `mingram` and `evaluate` commands, including parallel execution through
  `--num-threads` and optional quiet or JSON output.
- Report paired token counts and savings, code points per token, mean/P50/P95 tokens per
  prepared record, byte fallback, unknown IDs, round-trip failures, vocabulary usage, and
  test-shard results. Token compression does not establish downstream model quality.
- Save versioned, checksummed MinGram counters after each completed corpus file. Atomic
  artifact writes and completion markers distinguish interrupted exports from completed
  models. Changed semantic settings or corpus/model fingerprints reject stale checkpoints.
  Thread count may change when resuming.

### Improvements

- Show byte progress for copying, hashing, MinGram passes, and held-out evaluation. Unknown
  totals use status/spinner output, and non-interactive HPC logs receive periodic stderr
  progress without contaminating JSON stdout.
- Resume with `just train-only cfg.mingram.json` using the same work directory and model
  prefix. An unfinished MinGram file is repeated; unfinished native fitting or evaluation
  restarts that stage. Fixing and balancing must already have completed.
- Clear inherited Rust and C/C++ flags in `just` recipes and use an isolated
  `target/hpc-portable/` build cache to avoid reusing earlier CPU-specific build helpers.

### Performance

- Move script classification into repair workers, cache shard filenames, and keep counters
  beside buffered writers instead of searching shard summaries for every record.
- Parallelize MinGram counting and paired evaluation with bounded batches and worker-local
  integer reductions. Retain all training records and per-record reconstruction checks.
- Stream unbalanced corpus assembly instead of retaining entire shards in memory. Legacy
  explicitly enabled balancing retains its existing in-memory sampling path.

### Documentation

- Present `spm-forge` as a general multilingual tokenizer toolkit, with OCR as one use case.
- Add reproducible multilingual compression and NFC/NFD demos with committed synthetic
  inputs, generated Markdown reports, and documented limitations.
- Record local throughput measurements and byte-equivalence/resume checks in the README
  and MinGram experiment log. HPC throughput and downstream accuracy remain unmeasured.

### Developers

- Forbid unsafe Rust in the application crate and add parallel, checkpoint, interrupted
  export, byte-fallback, and stock Python runtime interoperability checks.
