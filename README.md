# spm-ocr

Train a SentencePiece tokenizer for messy multilingual OCR corpora.

This project is for OCR datasets that arrive as real-world dumps: directories, single files,
archives inside archives, compressed text, mixed scripts, weak path hints, extractor artifacts,
and uneven source sizes. The goal is to produce a tokenizer training corpus that is repaired,
logged, lightly balanced, and easy to inspect before training.

## Why Use It

- Pass one corpus path instead of pre-sorting files by extension or language.
- Detect text, archives, and compressed files by content rather than filename.
- Fix OCR-relevant Unicode issues while preserving characters that matter for labels.
- Split oversized OCR lines into safe chunks, skip only unrepairable lines, and log every action.
- Repair line chunks in parallel while writing deterministic corpus outputs.
- Build several meaningful training corpus files instead of one anonymous bag.
- Train SentencePiece with OCR-safe defaults from `cfg.json.ocr`.
- Keep progress visible through long discovery, repair, balancing, and training stages.

## Pipeline

1. Read settings from `cfg.json` and apply the selected OCR preset.
2. Find text from one file or a recursive directory scan, unpacking archives and compressed inputs.
3. Fix the corpus, preserving the original files and logging every repair, chunk, or skipped line.
4. Build several lightly balanced training corpus files under `work_dir/train_corpus/`.
5. Train SentencePiece and write the model, vocabulary, and reports under `work_dir/`.

## Usage

Minimal `cfg.json`:

```json
{
  "preset": "ocr_multilingual",
  "num_threads": 16,
  "corpus": {
    "path": "data/raw-corpus"
  },
  "output": {
    "work_dir": "runs/ocr-spm-v1",
    "model_prefix": "ocr_tokenizer"
  }
}
```

`num_threads` applies to both corpus repair and SentencePiece training.

To use a specialized preset, change only the preset name:

```json
{
  "preset": "ocr_cjk_heavy",
  "num_threads": 16,
  "corpus": {
    "path": "data/raw-corpus"
  },
  "output": {
    "work_dir": "runs/ocr-spm-v1",
    "model_prefix": "ocr_tokenizer"
  }
}
```

Run:

```sh
just train
```

Or directly:

```sh
RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" CFLAGS="${CFLAGS:-} -march=native" uv run cargo run --release -- train --config cfg.json
```

Press Ctrl+C to stop a run. During SentencePiece training, `spm-ocr` also stops the Python
trainer subprocess before exiting.

If the corpus has already been fixed and balanced in `work_dir`, retrain only SentencePiece:

```sh
just train-only
```

The Python trainer's explicit named options and fallback defaults are in
[`spm_ocr_train_bridge.py`](spm_ocr_train_bridge.py), in `_train`. Request values override
those defaults, so the Rust OCR presets still select their configured vocabulary sizes.
For a Python-only experiment, call `_train` with `input` and `model_prefix`, or replay an
edited `trainer_request.json` with `uv run python -m spm_ocr_train_bridge <request.json>`.
The Python fallback vocabulary size is 8192; other fallback choices use OCR settings
where applicable and SentencePiece 0.2.2 defaults otherwise.
`auto_character_coverage` is not supported by the currently available 0.2.2 wheel.

## Outputs

- `work_dir/train_corpus/*.txt`: meaningful corpus parts used for training.
- `work_dir/fixed_corpus.txt`: repaired audit stream.
- `work_dir/ocr_tokenizer.model`: trained SentencePiece model.
- `work_dir/ocr_tokenizer.vocab`: trained vocabulary.
- `work_dir/reports/corpus_issues.jsonl`: fixes, skips, and source issues.
- `work_dir/reports/balance.json`: balancing summary.
- `work_dir/trainer_request.json`: replayable trainer request.
- `work_dir/trainer_output.json`: trainer status and artifact paths.

## MinGram training

Reduce an existing BPE or Unigram candidate vocabulary using Rust MinGram training:

```sh
just cargo-run mingram --seed-model seed.model --input train.txt --output runs/mingram.model --vocab-size 8192
```

Or automatically generate a BPE seed through the existing trainer bridge:

```sh
just cargo-run mingram --config cfg.json --input train.txt --output runs/mingram.model --vocab-size 8192
```

Automatic seeding uses the configuration's SentencePiece settings and a seed vocabulary of
`ceil(1.15 * target_size)`, saving its artifacts under `runs/mingram.seed/`. Both modes use the
specified UTF-8 files directly, without OCR repair or language balancing. A BPE seed follows
the paper's initialization; importing an existing Unigram model is an additional adaptation.
The initial native BPE trainer has its own memory and sentence-sampling behavior. Rust MinGram
passes stream the files, keeping vocabulary counts and one bounded record in memory.

MinGram initializes probabilities from a seed-encoding pass, then runs two hard-EM passes
(`--iterations` overrides this). It uses the reference score `-1 + 0.00001 * log_probability`,
then prunes by usage counts, which rank the same as unrounded probabilities. Zero-use normal
pieces retain a finite log-probability floor of -100; they may fill otherwise unused slots to
preserve an exact vocabulary budget. Every single-code-point piece and non-normal piece is
protected. Reserved IDs stay fixed; other IDs may change. An impossible coverage budget or
reserved ID outside the target size is an error. Large reductions such as 100K to 8K are
supported, but single-step pruning may be worse than starting from a modestly oversized seed.

The command writes `mingram.model`, `mingram.vocab`, and `mingram.mingram.json`, refusing to
overwrite existing outputs. The report contains model hashes, pass statistics, and the complete
seed-to-output ID mapping. Training rejects unknown IDs and normalized reconstruction loss.
Use `--json` for the report on stdout and `--quiet` to disable progress. Check compression on
separate held-out files with `evaluate`; training-set savings do not establish generalization.

This is a SentencePiece-compatible adaptation of
[MinGram](https://arxiv.org/abs/2606.27019), not the reference SCRIPT tokenizer. Native
SentencePiece normalization, special-symbol scoring, byte fallback, float32 arithmetic, and
tie-breaking are retained. Scores favor fewer tokens but do not guarantee lexicographic
minimum-token segmentation on arbitrarily long inputs or minimize the expanded byte-fallback
ID count. No custom inference code is needed. See [module details](src/tokenizer/README.md).

## Tokenizer evaluation

For a self-contained demonstration with committed synthetic data and a generated Markdown
decision report, run:

```sh
uv run --locked demos/mingram/run.py
```

Read the [demo decision report](demos/mingram/REPORT.md) for the measured gains, regressions,
text provenance, and limits. The demo builds the Rust executable and trains all models;
it requires no existing tokenizer or corpus download.

Compare two standard SentencePiece models on the same held-out text:

```sh
just cargo-run evaluate --baseline baseline.model --candidate candidate.model --input heldout.txt
just cargo-run --json evaluate --baseline baseline.model --candidate candidate.model --input heldout.txt --quiet
```

Multiple files may follow `--input`. Evaluation streams UTF-8 records without corpus repair
or language balancing. It counts emitted IDs, including byte fallback, without adding BOS/EOS.
Only LF/CRLF record delimiters are removed; empty records and a final unterminated record
are preserved. Oversized records fail explicitly (`--max-record-bytes`, default 1 MiB).

The primary metric is candidate total tokens divided by baseline total tokens; below 1 is
better. Percentage saved is `100 * (1 - ratio)`. Reports include per-file and aggregate counts,
improved/unchanged/worsened records, unknown IDs, byte tokens, and reconstruction failures.
Zero baseline tokens produce an explicit undefined metric. Models with different normalization,
denormalization, or user-defined symbols are rejected conservatively. Vocabulary sizes are
reported separately: compression comparisons at different sizes do not establish an algorithm gain.

Raw round-trip failures compare decoded IDs with the original record. Normalized failures
compare the model-normalized decoded text with the model-normalized input, distinguishing
intentional normalization from information loss. Check both counters and unknown IDs before
accepting lower counts; lossy encoding is not a compression improvement. These are tokenizer
metrics, not language-model perplexity or evidence of improved downstream accuracy.

Native encoding uses the standard SentencePiece C++ runtime through Rust bindings. Cargo builds
the pinned bundled runtime statically. `just` commands use the project's existing uv environment
to supply CMake automatically; they do not require changes to SentencePiece's C++ source.

## Development

```sh
just check
just fmt
```

Design details live in [docs/10-rust-corpus-fixer-trainer-design.md](docs/10-rust-corpus-fixer-trainer-design.md).
