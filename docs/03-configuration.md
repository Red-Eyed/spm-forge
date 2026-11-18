# Recommended configuration

## Default multilingual profile

`cfg.json` selects `multilingual_mingram`: 8192 total pieces, 256 byte-fallback pieces,
NFC corpus preparation, identity model normalization, no user-defined symbols, no language
weights, and no corpus balancing. Seed pieces are capped at 16 Unicode code points.
`split_digits` is false. `character_coverage` is 0.9995; an alphabet that exceeds the protected
budget causes an explicit error. Native BPE input sampling is capped at 1,000,000 records.

Preparation preserves soft hyphens, NBSP, zero-width spaces, joiners, compatibility characters,
and Arabic presentation forms. NFC can change canonically equivalent sequences. Other presets
retain their OCR-specific mapping policies.

`evaluation.test_fraction` defaults to 0.05 and must be strictly between 0 and 1;
`evaluation.split_seed` defaults to 42. Content hashing assigns canonical original records
before chunking. Balancing, BPE seed training, MinGram, and the equal-size BPE baseline consume
training files only. The saved split and checksums are reused by `train-only`; a changed split
or preparation config requires a fresh preparation run. Choose a fresh model prefix to retrain.

Training writes `reports/tokenizer.md` and `reports/tokenizer.json` automatically, with
compression, distribution, coverage, reconstruction, and per-shard results. Byte tokens are
counted individually. This adds a BPE baseline fit for MinGram; BPE configurations evaluate
against themselves. See the [usage and interpretation guide](../README.md#held-out-evaluation).

## MinGram in cfg.json

Set `sentencepiece.model_type` to `"mingram"` to use the Rust optimizer through the normal
`just train` or `just train-only` workflow. `sentencepiece.vocab_size` is the final vocabulary
budget, including protected characters and reserved pieces. Optional
`sentencepiece.mingram.iterations` is a positive integer, defaulting to 2.

See [cfg.mingram.json](../cfg.mingram.json) for a complete example. The configured Python
runner generates a BPE seed at `ceil(1.15 * final_vocab_size)`. It also trains the equal-budget BPE baseline after MinGram. Other
SentencePiece options apply to both native training runs, and its normalization and coverage are retained
in the final model. No language weights are introduced. Corpus repair and any explicitly
enabled balancing still belong to the existing preparation pipeline.

The final model, vocabulary, MinGram report/ID mapping, and `trainer_output.json` are written
under the configured work directory. BPE seed requests and outputs are retained under
`<model_prefix>.seed/`. The top-level MinGram `trainer_request.json` records optimizer inputs;
it is not a request for direct replay by the Python bridge. Existing model artifacts are not
overwritten. MinGram-specific options with `model_type: "bpe"` are rejected.

## Existing BPE preset

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='balanced_corpus.txt',
    model_prefix='ocr_tokenizer',
    model_type='bpe',
    vocab_size=40000,
    character_coverage=0.9998,
    byte_fallback=True,
    normalization_rule_name='identity',
    add_dummy_prefix=False,
    remove_extra_whitespaces=False,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_digits=True,
    max_sentencepiece_length=8,
    max_sentence_length=16384,
    input_sentence_size=20000000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True,
    user_defined_symbols=[
        '\\frac', '\\sqrt', '\\sum', '\\int', '\\prod', '\\lim',
        '\\alpha', '\\beta', '\\gamma', '\\theta', '\\pi', '\\sigma',
        '\\times', '\\cdot', '\\leq', '\\geq', '\\neq', '\\approx',
        '\\left', '\\right', '\\begin', '\\end', '\\mathbf', '\\mathrm',
        '^', '_', '{', '}',
    ],
    num_threads=16,
)
```

## Per-parameter reasoning

**`model_type='bpe'`** — deterministic segmentation, nothing to disable. See
[BPE vs Unigram](02-bpe-vs-unigram.md).

**`normalization_rule_name='identity'`** — the single most consequential setting. The default
`nmt_nfkc` silently folds fullwidth↔halfwidth, ligatures and compatibility forms, changing what
"correct" means at eval time with no error and no warning. There is an important exception when
your ground truth is *already* normalized — see [normalization](07-normalization.md) before
committing to this.

**`character_coverage=0.9998` + `byte_fallback=True`** — used as a pair. High coverage keeps
nearly every character directly in the vocab; byte fallback ensures whatever's left decomposes
to UTF-8 bytes rather than an unrecoverable `<unk>`. In OCR an `<unk>` is a permanent error for
that character — the model has no path to ever emit it.

**`add_dummy_prefix=False`, `remove_extra_whitespaces=False`** — prevents phantom leading spaces
in non-whitespace scripts (CJK, Thai) and preserves whitespace that carries meaning (tables,
indentation, code blocks).

**`split_by_unicode_script=True`** — blocks cross-script merges, e.g. a Latin digit fusing with
a CJK character. Those merges waste vocab budget and increase confusability.

**`split_digits=True`** (default `False`) — forces every digit to its own piece. Without it BPE
merges frequent numbers into single tokens and the model learns to reproduce *those* numbers
rather than read the digits on the page. Frequency alone is enough to trigger it: a corpus
where `100` recurs produces a `100` piece. See [math and LaTeX](06-math-latex.md).

**`max_sentencepiece_length=8`** (default 16) — caps piece length, keeping CJK near
character-level. Improves error locality: one wrong token is one wrong character, not several
lost at once. Note the interaction with `user_defined_symbols`: this cap applies to *learned*
merges, so any LaTeX command longer than 8 characters (`\operatorname`, `\displaystyle`) can
never become atomic through frequency — it must be declared explicitly or it will fragment
regardless of how common it is.

**`vocab_size=40000`** — much smaller than LM vocabs (250K+). More examples per class means
sharper softmax separation and better per-class accuracy. The cost is longer sequences, which
is a latency cost, not an accuracy cost.

**`max_sentence_length=16384`** (default 4192 bytes) — SentencePiece **silently drops** lines
longer than this. `spm-ocr` chunks oversized OCR lines before training, and skips only lines
that cannot be chunked safely. Raise it for long LaTeX (matrices, multi-line derivations) and
verify against your actual line-length distribution rather than assuming.

**`user_defined_symbols`** — forces frequent LaTeX commands to survive as single unsplittable
tokens. Curate the list from real command-frequency counts on your corpus, not from the example
above. See [math and LaTeX](06-math-latex.md).

**`input_sentence_size` + `shuffle_input_sentence=True` + `train_extremely_large_corpus=True`** —
RAM-bounded training controls. See [corpus engineering](05-corpus-engineering.md); note in
particular that these do **not** solve corpus imbalance.
