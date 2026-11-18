use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::{de::DeserializeOwned, Serialize};

use crate::balance::{self, BalanceSummary};
use crate::cancel::CancellationToken;
use crate::config::{self, EffectiveConfig};
use crate::corpus::{self, DiscoveredCorpus};
use crate::progress::ProgressReporter;
use crate::repair::{self, RepairSummary};
use crate::tokenizer::{evaluate, model::Model, records::DEFAULT_MAX_RECORD_BYTES};
use crate::trainer::{self, TrainerOutput};

#[derive(Debug, Parser)]
#[command(name = "spm-ocr", version, about)]
pub struct Cli {
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train(TrainArgs),
    Evaluate(EvaluateArgs),
}

#[derive(Debug, Args)]
struct EvaluateArgs {
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long, required = true, num_args = 1..)]
    input: Vec<PathBuf>,
    #[arg(long, default_value_t = DEFAULT_MAX_RECORD_BYTES)]
    max_record_bytes: usize,
    #[arg(long)]
    quiet: bool,
}

#[derive(Debug, Args)]
struct TrainArgs {
    #[arg(long, default_value = "cfg.json")]
    config: PathBuf,
    #[arg(long, default_value = "cfg.json.ocr")]
    preset_config: PathBuf,
    #[arg(long)]
    train_only: bool,
}

#[derive(Debug, Serialize)]
struct TrainSummary {
    preset: String,
    corpus_path: PathBuf,
    text_files: usize,
    fixed_corpus: PathBuf,
    training_files: Vec<PathBuf>,
    issue_log: PathBuf,
    lines_read: u64,
    lines_written: u64,
    lines_fixed: u64,
    lines_skipped: u64,
    source_issues: usize,
    balanced_lines: u64,
    balance_report: PathBuf,
    model: PathBuf,
    vocab: PathBuf,
    trainer_output: PathBuf,
    work_dir: PathBuf,
    model_prefix: String,
}

#[derive(Debug, Serialize)]
struct TrainOnlySummary {
    preset: String,
    training_files: Vec<PathBuf>,
    balanced_lines: u64,
    balance_report: PathBuf,
    model: PathBuf,
    vocab: PathBuf,
    trainer_output: PathBuf,
    work_dir: PathBuf,
    model_prefix: String,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let cancellation =
        CancellationToken::install_ctrlc_handler().context("could not install Ctrl+C handler")?;

    match cli.command {
        Command::Train(args) => train(args, cli.json, &cancellation),
        Command::Evaluate(args) => evaluate_models(args, cli.json, &cancellation),
    }
}

fn evaluate_models(
    args: EvaluateArgs,
    json_output: bool,
    cancellation: &CancellationToken,
) -> Result<()> {
    let baseline = Model::load(&args.baseline)?;
    let candidate = Model::load(&args.candidate)?;
    let report = evaluate::compare(
        &baseline,
        &candidate,
        &args.input,
        args.max_record_bytes,
        args.quiet,
        cancellation,
    )?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "Vocabulary: baseline {}, candidate {}",
            report.baseline_vocab_size, report.candidate_vocab_size
        );
        for file in &report.files {
            print_metrics(&file.input.display().to_string(), &file.metrics);
        }
        print_metrics("Total", &report.total);
    }
    Ok(())
}

fn print_metrics(label: &str, metrics: &evaluate::Metrics) {
    println!(
        "{label}: {} records, {} → {} tokens",
        metrics.counts.records, metrics.counts.baseline.tokens, metrics.counts.candidate.tokens
    );
    match metrics.percent_tokens_saved {
        evaluate::Ratio::Defined { value } => println!("  Tokens saved: {value:.3}%"),
        evaluate::Ratio::Undefined { reason } => println!("  Tokens saved: undefined ({reason})"),
    }
    println!(
        "  Records: {} improved, {} unchanged, {} worsened",
        metrics.counts.improved_records,
        metrics.counts.unchanged_records,
        metrics.counts.worsened_records
    );
    println!("  Raw round-trip failures: baseline {}, candidate {}; normalized failures: baseline {}, candidate {}",
        metrics.counts.baseline.raw_round_trip_failures, metrics.counts.candidate.raw_round_trip_failures,
        metrics.counts.baseline.normalized_round_trip_failures, metrics.counts.candidate.normalized_round_trip_failures);
}

fn train(args: TrainArgs, json_output: bool, cancellation: &CancellationToken) -> Result<()> {
    let progress = ProgressReporter::new(json_output);

    let config = load_config(&args, &progress)?;
    cancellation.check()?;

    if args.train_only {
        return train_from_existing_corpus(&config, &progress, cancellation, json_output);
    }

    let corpus = discover_corpus(&config, &progress)?;
    cancellation.check()?;
    let repair = repair_corpus(&config, &corpus, &progress, cancellation)?;
    if repair.lines_written == 0 {
        bail!(
            "no usable text lines found under {}; see {}",
            config.corpus.path.display(),
            repair.issue_log.display()
        );
    }
    cancellation.check()?;
    let balance = balance_corpus(&config, &repair, &progress)?;
    cancellation.check()?;
    let trainer = train_sentencepiece(&config, &balance, &progress, cancellation)?;
    let summary = TrainSummary::from_outputs(&config, &corpus, &repair, &balance, &trainer);

    if json_output {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        let source_issue_suffix = if summary.source_issues == 0 {
            String::new()
        } else {
            format!(", {} source issue(s) logged", summary.source_issues)
        };
        println!(
            "Trained: preset `{}`, {} text file(s), {} training part(s), {} repaired line(s), {} balanced line(s), {} fixed, {} skipped{}, model `{}`",
            summary.preset,
            summary.text_files,
            summary.training_files.len(),
            summary.lines_written,
            summary.balanced_lines,
            summary.lines_fixed,
            summary.lines_skipped,
            source_issue_suffix,
            summary.model.display()
        );
    }

    Ok(())
}

fn train_from_existing_corpus(
    config: &EffectiveConfig,
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
    json_output: bool,
) -> Result<()> {
    let balance = load_existing_balance(config, progress)?;
    cancellation.check()?;
    let trainer = train_sentencepiece(config, &balance, progress, cancellation)?;
    let summary = TrainOnlySummary::from_outputs(config, &balance, &trainer);

    if json_output {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "Trained from existing corpus: preset `{}`, {} training part(s), {} balanced line(s), model `{}`",
            summary.preset,
            summary.training_files.len(),
            summary.balanced_lines,
            summary.model.display()
        );
    }

    Ok(())
}

fn load_config(args: &TrainArgs, progress: &ProgressReporter) -> Result<EffectiveConfig> {
    let stage = progress.stage(format!("loading config: {}", args.config.display()));
    let config = config::load_effective_config(&args.config, &args.preset_config)
        .with_context(|| format!("could not load {}", args.config.display()))?;
    stage.finish(format!(
        "loaded config: preset `{}`, work_dir `{}`",
        config.preset,
        config.output.work_dir.display()
    ));
    Ok(config)
}

fn discover_corpus(
    config: &EffectiveConfig,
    progress: &ProgressReporter,
) -> Result<DiscoveredCorpus> {
    let stage = progress.stage(format!(
        "discovering corpus: {}",
        config.corpus.path.display()
    ));
    let unpack_dir = config.output.work_dir.join("unpacked");
    let corpus = corpus::discover_text_files(&config.corpus.path, &unpack_dir)
        .with_context(|| format!("could not discover {}", config.corpus.path.display()))?;

    stage.finish(format!(
        "found {} text file(s), {} source issue(s)",
        corpus.files.len(),
        corpus.issues.len()
    ));
    Ok(corpus)
}

fn repair_corpus(
    config: &EffectiveConfig,
    corpus: &DiscoveredCorpus,
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<RepairSummary> {
    repair::repair_corpus(config, corpus, progress, cancellation)
        .with_context(|| format!("could not repair {}", config.corpus.path.display()))
}

fn balance_corpus(
    config: &EffectiveConfig,
    repair: &RepairSummary,
    progress: &ProgressReporter,
) -> Result<BalanceSummary> {
    balance::balance_corpus(config, repair, progress).context("could not balance corpus")
}

fn load_existing_balance(
    config: &EffectiveConfig,
    progress: &ProgressReporter,
) -> Result<BalanceSummary> {
    let report = config.output.work_dir.join("reports/balance.json");
    let stage = progress.stage(format!(
        "loading existing training corpus: {}",
        report.display()
    ));
    let balance = read_json::<BalanceSummary>(&report).with_context(|| {
        format!(
            "could not load existing balance report {}",
            report.display()
        )
    })?;
    validate_existing_training_files(&balance.training_files)?;
    stage.finish(format!(
        "loaded existing training corpus: {} part(s)",
        balance.training_files.len()
    ));
    Ok(balance)
}

fn validate_existing_training_files(paths: &[PathBuf]) -> Result<()> {
    if paths.is_empty() {
        bail!("existing balance report contains no training files; rerun without --train-only");
    }

    for path in paths {
        if !path.is_file() {
            bail!(
                "training corpus file is missing: {}; rerun without --train-only",
                path.display()
            );
        }
    }

    Ok(())
}

fn read_json<T>(path: &Path) -> Result<T>
where
    T: DeserializeOwned,
{
    let file = File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("could not parse {}", path.display()))
}

fn train_sentencepiece(
    config: &EffectiveConfig,
    balance: &BalanceSummary,
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<TrainerOutput> {
    trainer::train_sentencepiece(config, &balance.training_files, progress, cancellation)
        .context("could not train SentencePiece")
}

impl TrainSummary {
    fn from_outputs(
        config: &EffectiveConfig,
        corpus: &DiscoveredCorpus,
        repair: &RepairSummary,
        balance: &BalanceSummary,
        trainer: &TrainerOutput,
    ) -> Self {
        Self {
            preset: config.preset.clone(),
            corpus_path: corpus.root.clone(),
            text_files: corpus.files.len(),
            fixed_corpus: repair.fixed_corpus.clone(),
            training_files: balance.training_files.clone(),
            issue_log: repair.issue_log.clone(),
            lines_read: repair.lines_read,
            lines_written: repair.lines_written,
            lines_fixed: repair.lines_fixed,
            lines_skipped: repair.lines_skipped,
            source_issues: repair.source_issues,
            balanced_lines: balance.output_lines,
            balance_report: balance.report.clone(),
            model: trainer.model.clone(),
            vocab: trainer.vocab.clone(),
            trainer_output: config.output.work_dir.join("trainer_output.json"),
            work_dir: config.output.work_dir.clone(),
            model_prefix: config.output.model_prefix.clone(),
        }
    }
}

impl TrainOnlySummary {
    fn from_outputs(
        config: &EffectiveConfig,
        balance: &BalanceSummary,
        trainer: &TrainerOutput,
    ) -> Self {
        Self {
            preset: config.preset.clone(),
            training_files: balance.training_files.clone(),
            balanced_lines: balance.output_lines,
            balance_report: balance.report.clone(),
            model: trainer.model.clone(),
            vocab: trainer.vocab.clone(),
            trainer_output: config.output.work_dir.join("trainer_output.json"),
            work_dir: config.output.work_dir.clone(),
            model_prefix: config.output.model_prefix.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn parses_train_only_flag() {
        let cli = Cli::try_parse_from(["spm-ocr", "train", "--train-only"]).expect("parse cli");

        match cli.command {
            Command::Train(args) => assert!(args.train_only),
            other => panic!("expected train, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_existing_training_files() {
        let error = validate_existing_training_files(&[])
            .expect_err("empty training file list should be rejected");

        assert!(error.to_string().contains("no training files"));
    }

    #[test]
    fn rejects_missing_existing_training_file() {
        let temp = TempDir::new().expect("temp dir");
        let missing = temp.path().join("missing.txt");

        let error = validate_existing_training_files(&[missing])
            .expect_err("missing training file should be rejected");

        assert!(error
            .to_string()
            .contains("training corpus file is missing"));
    }
}
