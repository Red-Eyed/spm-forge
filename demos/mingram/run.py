# /// script
# requires-python = ">=3.11"
# dependencies = ["sentencepiece==0.2.3", "pydantic-settings>=2,<3", "cmake>=4.4.4,<5"]
# ///
"""Build and run the Rust MinGram demo, then write a decision-oriented Markdown report.

The committed corpus is synthetic and needs no download. This script uses upstream
SentencePiece only for baseline/seed training; MinGram and evaluation run in Rust.
"""

import os
import subprocess
import tempfile
from hashlib import sha256
from pathlib import Path

import sentencepiece as sp
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parent.parent


class CorpusFile(BaseModel):
    """Identify and verify one committed corpus input."""

    path: Path
    sha256: str
    records: int


class Manifest(BaseModel):
    """Validate the committed list of corpus files at the input boundary."""

    seed: int
    files: list[CorpusFile]


class TokenCounts(BaseModel):
    """Read exact native-runtime counts and loss diagnostics."""

    tokens: int
    unknown_tokens: int
    raw_round_trip_failures: int


class Counts(BaseModel):
    """Read paired counts without interpreting tokenizer scores as quality."""

    records: int
    improved_records: int
    unchanged_records: int
    worsened_records: int
    baseline: TokenCounts
    candidate: TokenCounts


class Metrics(BaseModel):
    """Derive compression savings from integer corpus counts."""

    counts: Counts

    @property
    def saved(self) -> float:
        """Return percentage savings; this demo requires a nonempty baseline."""
        return 100 * (1 - self.counts.candidate.tokens / self.counts.baseline.tokens)


class FileReport(BaseModel):
    """Read one source-specific result from the Rust evaluator."""

    input: Path
    metrics: Metrics


class Evaluation(BaseModel):
    """Validate the evaluator fields used to make the decision report."""

    baseline_vocab_size: int
    candidate_vocab_size: int
    same_vocab_size: bool
    normalization_matches: bool
    files: list[FileReport]
    total: Metrics


class Settings(BaseSettings):
    """Run the demo with locked dependencies and fresh output artifacts."""

    model_config = ConfigDict(extra="forbid")
    quiet: CliImplicitFlag[bool] = False

    def cli_cmd(self) -> None:
        """Execute the complete build, training, evaluation, and reporting workflow."""
        manifest = Manifest.model_validate_json(
            (DEMO / "data/manifest.json").read_text()
        )
        verify_inputs(manifest)
        artifacts = REPO / "target/mingram-demo"
        artifacts.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="run-", dir=artifacts))
        binary = build_binary()
        data = DEMO / "data"
        train_baselines(output, data)
        train_mingram(binary, output, self.quiet, data)
        bpe = evaluate(binary, output, "bpe", manifest, self.quiet, data)
        unigram = evaluate(binary, output, "unigram", manifest, self.quiet, data)
        report = decision_report(bpe, unigram)
        path = output / "REPORT.md"
        path.write_text(report, encoding="utf-8")
        print(f"\nReport: {path}\nArtifacts: {output}")


def verify_inputs(manifest: Manifest) -> None:
    """Reject changed inputs and accidental train/test overlap before training."""
    training = set((DEMO / "data/train.txt").read_text(encoding="utf-8").splitlines())
    for file in manifest.files:
        path = DEMO / "data" / file.path
        data = path.read_bytes()
        if sha256(data).hexdigest() != file.sha256:
            raise ValueError(f"corpus checksum mismatch: {path}")
        lines = data.decode("utf-8").splitlines()
        if len(lines) != file.records:
            raise ValueError(f"corpus record count mismatch: {path}")
        if file.path.name != "train.txt" and not training.isdisjoint(lines):
            raise ValueError(f"training and held-out records overlap: {path}")


def build_binary() -> Path:
    """Build the locked Rust implementation and return its executable path."""
    subprocess.run(["cargo", "build", "--release", "--locked"], cwd=REPO, check=True)
    return REPO / "target/release" / ("spm-ocr.exe" if os.name == "nt" else "spm-ocr")


def train_baselines(output: Path, data: Path, max_piece_length: int = 16) -> None:
    """Train equal-budget baselines and the modestly oversized BPE seed."""
    for name, kind, size in (
        ("bpe", "bpe", 512),
        ("unigram", "unigram", 512),
        ("seed", "bpe", 589),
    ):
        print(f"Training {name}: {size} pieces", flush=True)
        sp.SentencePieceTrainer.train(
            input=str(data / "train.txt"),
            model_prefix=str(output / name),
            model_type=kind,
            vocab_size=size,
            character_coverage=1.0,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            byte_fallback=False,
            max_sentencepiece_length=max_piece_length,
            input_sentence_size=0,
            shuffle_input_sentence=False,
            num_threads=1,
            minloglevel=2,
        )


def train_mingram(binary: Path, output: Path, quiet: bool, data: Path) -> None:
    """Invoke the actual Rust MinGram command and retain its training report."""
    command = [
        str(binary),
        "--json",
        "mingram",
        "--seed-model",
        str(output / "seed.model"),
        "--input",
        str(data / "train.txt"),
        "--output",
        str(output / "mingram.model"),
        "--vocab-size",
        "512",
    ]
    if quiet:
        command.append("--quiet")
    with (output / "training.json").open("w", encoding="utf-8") as report:
        subprocess.run(command, cwd=REPO, stdout=report, check=True)


def evaluate(
    binary: Path,
    output: Path,
    baseline: str,
    manifest: Manifest,
    quiet: bool,
    data: Path,
) -> Evaluation:
    """Run the Rust evaluator, preserving the full JSON evidence."""
    tests = [
        str(data / item.path)
        for item in manifest.files
        if item.path.name != "train.txt"
    ]
    results = output / "results"
    results.mkdir(exist_ok=True)
    path = results / f"vs-{baseline}.json"
    command = [
        str(binary),
        "--json",
        "evaluate",
        "--baseline",
        str(output / f"{baseline}.model"),
        "--candidate",
        str(output / "mingram.model"),
        "--input",
        *tests,
    ]
    if quiet:
        command.append("--quiet")
    with path.open("w", encoding="utf-8") as report:
        subprocess.run(command, cwd=REPO, stdout=report, check=True)
    return Evaluation.model_validate_json(path.read_text(encoding="utf-8"))


def decision_report(bpe: Evaluation, unigram: Evaluation) -> str:
    """Generate a recommendation tied to measured savings, regressions, and correctness."""
    comparable = (
        bpe.same_vocab_size
        and unigram.same_vocab_size
        and bpe.normalization_matches
        and unigram.normalization_matches
    )
    lossless = all(
        count.unknown_tokens == 0 and count.raw_round_trip_failures == 0
        for report in (bpe, unigram)
        for count in (report.total.counts.baseline, report.total.counts.candidate)
    )
    if not comparable or not lossless:
        decision = "Do not adopt: comparability or exact reconstruction checks failed."
    elif bpe.total.saved > 0:
        decision = "Worth evaluating against BPE on representative text; this synthetic result does not establish production quality."
    else:
        decision = (
            "This run does not demonstrate an aggregate compression benefit over BPE."
        )
    summary = "\n".join(
        [
            "| Tokenizer | Held-out tokens | MinGram saves |",
            "|---|---:|---:|",
            f"| BPE | {bpe.total.counts.baseline.tokens:,} | {bpe.total.saved:.2f}% |",
            f"| Unigram | {unigram.total.counts.baseline.tokens:,} | {unigram.total.saved:.2f}% |",
            f"| MinGram | {bpe.total.counts.candidate.tokens:,} | — |",
        ]
    )
    rows, regressions = language_rows(bpe, unigram)
    changes = "\n\n".join(
        record_changes(label, report)
        for label, report in (("BPE", bpe), ("Unigram", unigram))
    )
    return (
        (DEMO / "report-template.md")
        .read_text(encoding="utf-8")
        .format(
            decision=decision,
            summary=summary,
            rows=rows,
            regressions=regressions,
            changes=changes,
            comparable="PASS" if comparable else "FAIL",
            lossless="PASS" if lossless else "FAIL",
        )
    )


def language_rows(bpe: Evaluation, unigram: Evaluation) -> tuple[str, str]:
    """Render every source group and explicitly surface Unigram regressions."""
    rows: list[str] = []
    regressions: list[str] = []
    for before, other in zip(bpe.files, unigram.files, strict=True):
        if before.input.name != other.input.name:
            raise ValueError("baseline reports use different file orders")
        name = before.input.stem.removeprefix("heldout-").capitalize()
        rows.append(
            f"| {name} | {before.metrics.counts.baseline.tokens:,} | {other.metrics.counts.baseline.tokens:,} | "
            f"{before.metrics.counts.candidate.tokens:,} | {before.metrics.saved:.2f}% | {other.metrics.saved:.2f}% |"
        )
        if other.metrics.saved < 0:
            regressions.append(
                f"{name}: {-other.metrics.saved:.2f}% more tokens than Unigram"
            )
    warning = "No source-group regressions against Unigram were observed in this run."
    if regressions:
        warning = (
            "**Do not replace Unigram unconditionally.** Aggregate gains hide regressions: "
            + "; ".join(regressions)
            + "."
        )
    return "\n".join(rows), warning


def record_changes(label: str, report: Evaluation) -> str:
    """Render paired record-level changes without hiding longer outputs."""
    counts = report.total.counts
    return (
        f"Against {label}: {counts.improved_records:,} records improved, "
        f"{counts.unchanged_records:,} were unchanged, and {counts.worsened_records:,} became longer."
    )


if __name__ == "__main__":
    CliApp.run(Settings)
