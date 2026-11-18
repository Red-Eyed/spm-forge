# /// script
# requires-python = ">=3.11"
# dependencies = ["sentencepiece==0.2.3", "pydantic-settings>=2,<3", "cmake>=4.4.4,<5"]
# ///
"""Compare NFC and NFD using the committed corpus and actual Rust tokenizer commands.

Unicode normalization is external preprocessing here. Exported models retain identity
normalization; this experiment does not add NFC or NFD deployment support.
"""

import tempfile
import unicodedata
from pathlib import Path
from typing import Literal

import run as demo
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag

Form = Literal["NFC", "NFD"]
FORMS: tuple[Form, Form] = ("NFC", "NFD")


class CorpusStats(BaseModel):
    """Separate representation changes from lossless normalized reconstruction."""

    training_alphabet_size: int
    changed_heldout_records: int
    heldout_codepoints: int


class Result(BaseModel):
    """Describe one fixed normalization and piece-limit experiment."""

    form: Form
    max_piece_length: int
    corpus: CorpusStats
    bpe: demo.Evaluation
    unigram: demo.Evaluation


class Comparison(BaseModel):
    """Record experiment metadata and all four variant results."""

    unicode_database_version: str
    results: list[Result]


class Settings(BaseSettings):
    """Run the normalization comparison and the predeclared piece-limit control."""

    model_config = ConfigDict(extra="forbid")
    quiet: CliImplicitFlag[bool] = False

    def cli_cmd(self) -> None:
        """Verify inputs, run both forms at two limits, and write evidence and Markdown."""
        manifest = demo.Manifest.model_validate_json(
            (demo.DEMO / "data/manifest.json").read_text()
        )
        demo.verify_inputs(manifest)
        parent = demo.REPO / "target/mingram-normalization"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
        binary = demo.build_binary()
        results = [
            run_variant(binary, output, manifest, form, limit, self.quiet)
            for limit in (16, 64)
            for form in FORMS
        ]
        comparison = Comparison(
            unicode_database_version=unicodedata.unidata_version, results=results
        )
        (output / "normalization-results.json").write_text(
            comparison.model_dump_json(indent=2), encoding="utf-8"
        )
        (output / "REPORT.md").write_text(report(comparison), encoding="utf-8")
        print(
            f"\nReport: {output / 'REPORT.md'}\nEvidence: {output / 'normalization-results.json'}"
        )


def normalized_inputs(output: Path, manifest: demo.Manifest, form: Form) -> CorpusStats:
    """Normalize copies of verified inputs; leave committed source records untouched."""
    output.mkdir(parents=True)
    changed = 0
    codepoints = 0
    for entry in manifest.files:
        original = (demo.DEMO / "data" / entry.path).read_text(encoding="utf-8")
        normalized = unicodedata.normalize(form, original)
        (output / entry.path).write_text(normalized, encoding="utf-8")
        if entry.path.name != "train.txt":
            changed += sum(
                a != b
                for a, b in zip(
                    original.splitlines(), normalized.splitlines(), strict=True
                )
            )
            codepoints += sum(map(len, normalized.splitlines()))
    training = (output / "train.txt").read_text(encoding="utf-8")
    alphabet = len(set(training.replace("\n", "")))
    train_records = set(training.splitlines())
    for entry in manifest.files:
        if entry.path.name != "train.txt":
            heldout = (output / entry.path).read_text(encoding="utf-8").splitlines()
            if not train_records.isdisjoint(heldout):
                raise ValueError("normalization introduced train/test overlap")
    return CorpusStats(
        training_alphabet_size=alphabet,
        changed_heldout_records=changed,
        heldout_codepoints=codepoints,
    )


def run_variant(
    binary: Path,
    output: Path,
    manifest: demo.Manifest,
    form: Form,
    limit: int,
    quiet: bool,
) -> Result:
    """Train equal-budget models and evaluate them against the corresponding normalized text."""
    folder = output / f"{form}-limit{limit}"
    data = folder / "data"
    stats = normalized_inputs(data, manifest, form)
    print(f"Comparing {form}, maximum piece length {limit}", flush=True)
    demo.train_baselines(folder, data, limit)
    demo.train_mingram(binary, folder, quiet, data)
    bpe = demo.evaluate(binary, folder, "bpe", manifest, quiet, data)
    unigram = demo.evaluate(binary, folder, "unigram", manifest, quiet, data)
    return Result(
        form=form, max_piece_length=limit, corpus=stats, bpe=bpe, unigram=unigram
    )


def token_totals(result: Result) -> tuple[int, int, int]:
    """Return BPE, Unigram, and MinGram emitted-ID totals for one representation."""
    return (
        result.bpe.total.counts.baseline.tokens,
        result.unigram.total.counts.baseline.tokens,
        result.bpe.total.counts.candidate.tokens,
    )


def report(comparison: Comparison) -> str:
    """Render paired normalization results without confusing normalized and original text."""
    nfc, nfd, nfc64, nfd64 = comparison.results
    rows = []
    for name, before, after in zip(
        ("BPE", "Unigram", "MinGram"), token_totals(nfc), token_totals(nfd), strict=True
    ):
        rows.append(
            f"| {name} | {before:,} | {after:,} | {100 * (after / before - 1):.2f}% |"
        )
    sources = []
    for before, after in zip(nfc.bpe.files, nfd.bpe.files, strict=True):
        if before.input.name != after.input.name:
            raise ValueError("normalization results use different input orders")
        a, b = (
            before.metrics.counts.candidate.tokens,
            after.metrics.counts.candidate.tokens,
        )
        name = before.input.stem.removeprefix("heldout-").capitalize()
        sources.append(f"| {name} | {a:,} | {b:,} | {100 * (b / a - 1):.2f}% |")
    lossless = all(
        count.unknown_tokens == 0 and count.raw_round_trip_failures == 0
        for result in comparison.results
        for evaluation in (result.bpe, result.unigram)
        for count in (
            evaluation.total.counts.baseline,
            evaluation.total.counts.candidate,
        )
    )
    increase = 100 * (token_totals(nfd)[2] / token_totals(nfc)[2] - 1)
    decision = (
        "NFC emits fewer MinGram tokens on this synthetic corpus."
        if increase > 0
        else "NFD does not increase MinGram token count in this run."
    )
    comparable = all(
        e.same_vocab_size and e.normalization_matches
        for r in comparison.results
        for e in (r.bpe, r.unigram)
    )
    if not lossless or not comparable:
        decision = "Do not adopt either representation from this run: correctness or comparability checks failed."
    control = f"At a 64-code-point limit, NFC MinGram emits {token_totals(nfc64)[2]:,} tokens and NFD emits {token_totals(nfd64)[2]:,}."
    return (
        (demo.DEMO / "normalization-report-template.md")
        .read_text(encoding="utf-8")
        .format(
            decision=decision,
            totals="\n".join(rows),
            sources="\n".join(sources),
            control=control,
            alphabet_nfc=nfc.corpus.training_alphabet_size,
            alphabet_nfd=nfd.corpus.training_alphabet_size,
            changed_nfc=nfc.corpus.changed_heldout_records,
            changed_nfd=nfd.corpus.changed_heldout_records,
            lossless="PASS" if lossless else "FAIL",
            unicode_version=comparison.unicode_database_version,
        )
    )


if __name__ == "__main__":
    CliApp.run(Settings)
