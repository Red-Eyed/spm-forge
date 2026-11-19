//! Selects the configured training algorithm after corpus preparation.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::cancel::CancellationToken;
use crate::config::{BalancingMode, EffectiveConfig, MingramConfig, TrainingAlgorithm};
use crate::corpus_split::SplitManifest;
use crate::progress::ProgressReporter;
use crate::tokenizer::{
    checkpoint::{self, Artifacts, Store},
    mingram,
    model::Model,
};
use crate::trainer::{self, TrainerOutput, TrainerPaths, TrainerRequest, TrainerStatus};

/// Fits both models on training files only, then evaluates on the saved test partition.
pub fn train_and_report(
    config: &EffectiveConfig,
    split: &SplitManifest,
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<TrainerOutput> {
    let mut training_config = config.clone();
    let (inputs, identity) =
        if config.balancing.enabled && config.balancing.mode == BalancingMode::ScriptSample {
            let sample = crate::script_sampling::prepare(config, split, progress, cancellation)?;
            training_config.sentencepiece.input_sentence_size = 0;
            training_config.sentencepiece.shuffle_input_sentence = false;
            let identity = checkpoint::key(&(1, &split.train, &sample))?;
            (sample.files, identity)
        } else {
            (split.train_paths(), checkpoint::key(&(1, &split.train))?)
        };
    let fitted = &training_config;
    let store = Store::new(
        config
            .output
            .work_dir
            .join(format!("{}.checkpoints", config.output.model_prefix)),
        identity,
    );
    let output = match config.sentencepiece.algorithm {
        TrainingAlgorithm::Bpe | TrainingAlgorithm::Unigram { .. } => {
            cached_native(fitted, &inputs, &store, "candidate", progress, cancellation)?
        }
        TrainingAlgorithm::Mingram { mingram } => {
            train_mingram(fitted, &inputs, mingram, &store, progress, cancellation)?
        }
    };
    publish_training_metadata(fitted, &inputs, &output)?;
    let baseline = match config.sentencepiece.algorithm {
        TrainingAlgorithm::Bpe => output.model.clone(),
        TrainingAlgorithm::Mingram { .. } | TrainingAlgorithm::Unigram { .. } => {
            let mut baseline = fitted.clone();
            baseline.sentencepiece.algorithm = TrainingAlgorithm::Bpe;
            baseline.output.work_dir = config
                .output
                .work_dir
                .join(format!("{}.baseline", config.output.model_prefix));
            baseline.output.model_prefix = "bpe".into();
            cached_native(
                &baseline,
                &inputs,
                &store,
                "baseline",
                progress,
                cancellation,
            )?
            .model
        }
    };
    evaluate_and_report(
        config,
        split,
        &baseline,
        &output.model,
        &store,
        progress,
        cancellation,
    )?;
    Ok(output)
}

fn publish_training_metadata(
    config: &EffectiveConfig,
    inputs: &[PathBuf],
    output: &TrainerOutput,
) -> Result<()> {
    let paths = TrainerPaths::from_config(config);
    checkpoint::atomic_write(&paths.output, &serde_json::to_vec_pretty(output)?)?;
    match config.sentencepiece.algorithm {
        TrainingAlgorithm::Bpe | TrainingAlgorithm::Unigram { .. } => checkpoint::atomic_write(
            &paths.request,
            &serde_json::to_vec_pretty(&trainer::build_trainer_request(config, inputs, &paths))?,
        )?,
        TrainingAlgorithm::Mingram { mingram } => {
            let seed = TrainerPaths::from_config(&seed_config(config)?);
            checkpoint::atomic_write(
                &paths.request,
                &serde_json::to_vec_pretty(&MingramRequest {
                    algorithm: "mingram",
                    input: inputs,
                    seed_model: &seed.model,
                    vocab_size: config.sentencepiece.vocab_size,
                    iterations: mingram.iterations.get(),
                })?,
            )?;
        }
    }
    Ok(())
}

fn evaluate_and_report(
    config: &EffectiveConfig,
    split: &SplitManifest,
    baseline: &std::path::Path,
    candidate: &std::path::Path,
    store: &Store,
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<()> {
    cancellation.check()?;
    let reports = [
        config.output.work_dir.join("reports/tokenizer.md"),
        config.output.work_dir.join("reports/tokenizer.json"),
    ];
    let evaluation_key = checkpoint::key(&(
        3,
        &store.identity,
        &split.test,
        Model::load(baseline)?.fingerprint(),
        Model::load(candidate)?.fingerprint(),
    ))?;
    if let Some(saved) = store.read::<Artifacts>("evaluation", &evaluation_key)? {
        if saved.verify(&reports).is_ok() {
            progress
                .stage("reusing completed held-out evaluation")
                .finish("held-out reports verified");
            return Ok(());
        }
        progress
            .stage("held-out reports were replaced or incomplete")
            .finish("regenerating held-out evaluation");
    }
    let before = Model::load(baseline)?;
    let after = Model::load(candidate)?;
    let report = crate::tokenizer::evaluate::compare_with_options(
        &before,
        &after,
        &split.test_paths(),
        &crate::tokenizer::evaluate::Options {
            max_record_bytes: config.sentencepiece.max_sentence_length as usize,
            threads: config.num_threads as usize,
            progress,
        },
        cancellation,
    )?;
    crate::tokenizer::report::write(config, split, baseline, candidate, &report)?;
    store.write(
        "evaluation",
        &evaluation_key,
        &Artifacts::capture(&reports)?,
    )?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct CompletedTraining {
    output: TrainerOutput,
    artifacts: Artifacts,
}

fn native_request(config: &EffectiveConfig, inputs: &[PathBuf]) -> TrainerRequest {
    let mut request =
        trainer::build_trainer_request(config, inputs, &TrainerPaths::from_config(config));
    request.sentencepiece.num_threads = 0;
    request
}

fn cached_native(
    config: &EffectiveConfig,
    inputs: &[PathBuf],
    store: &Store,
    name: &str,
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<TrainerOutput> {
    let paths = TrainerPaths::from_config(config);
    let expected = native_request(config, inputs);
    let key = checkpoint::key(&(1, &store.identity, &expected))?;
    let artifacts = [paths.model.clone(), paths.vocab.clone()];
    if let Some(saved) = store.read::<CompletedTraining>(name, &key)? {
        saved.artifacts.verify(&artifacts)?;
        ensure!(
            saved.output.status == TrainerStatus::Succeeded
                && saved.output.model == paths.model
                && saved.output.vocab == paths.vocab,
            "invalid completed native training state"
        );
        Model::load(&saved.output.model)?;
        progress
            .stage(format!("reusing completed {name}"))
            .finish(format!("{name} model verified"));
        return Ok(saved.output);
    }
    let intent_name = format!("{name}-intent");
    let own_intent = store.read::<TrainerRequest>(&intent_name, &key)?;
    if let Some(intent) = &own_intent {
        ensure!(intent == &expected, "native training intent differs");
    }
    if let Some(output) = legacy_native_output(&paths, &expected, own_intent.is_some())? {
        let saved = CompletedTraining {
            artifacts: Artifacts::capture(&artifacts)?,
            output,
        };
        store.write(name, &key, &saved)?;
        progress
            .stage(format!("adopting existing {name}"))
            .finish(format!("existing {name} verified"));
        return Ok(saved.output);
    }
    store.write(&intent_name, &key, &expected)?;
    let output = trainer::train_sentencepiece(config, inputs, progress, cancellation)?;
    store.write(
        name,
        &key,
        &CompletedTraining {
            artifacts: Artifacts::capture(&artifacts)?,
            output: output.clone(),
        },
    )?;
    Ok(output)
}

fn legacy_native_output(
    paths: &TrainerPaths,
    expected: &TrainerRequest,
    owned: bool,
) -> Result<Option<TrainerOutput>> {
    if !paths.model.exists() && !paths.vocab.exists() {
        return Ok(None);
    }
    if !paths.request.exists() {
        ensure!(
            owned || (!paths.model.exists() && !paths.vocab.exists()),
            "untracked native artifacts exist; use a new model_prefix"
        );
        return Ok(None);
    }
    match serde_json::from_slice::<TrainerRequest>(&fs::read(&paths.request)?) {
        Ok(mut previous) => {
            previous.sentencepiece.num_threads = 0;
            ensure!(
                &previous == expected,
                "existing native request differs; use a new model_prefix"
            );
        }
        Err(error) if !owned => return Err(error.into()),
        Err(_) => {}
    }
    if !paths.output.exists() {
        return Ok(None);
    }
    let Ok(output) = serde_json::from_slice::<TrainerOutput>(&fs::read(&paths.output)?) else {
        return Ok(None);
    };
    if output.status != TrainerStatus::Succeeded {
        return Ok(None);
    }
    ensure!(
        output.model == paths.model && output.vocab == paths.vocab,
        "native output paths differ"
    );
    let model = Model::load(&output.model)?;
    ensure!(
        model.proto.pieces.len() == expected.sentencepiece.vocab_size as usize,
        "saved native vocabulary size differs"
    );
    ensure!(
        model.proto.trainer_spec.as_ref().is_some_and(|s| {
            match expected.sentencepiece.model_type.as_str() {
                "bpe" => s.model_type() == sentencepiece_model::ModelType::Bpe,
                "unigram" => s.model_type() == sentencepiece_model::ModelType::Unigram,
                _ => false,
            }
        }),
        "saved native model type differs from the training request"
    );
    Ok(Some(output))
}

#[derive(Serialize)]
struct MingramRequest<'a> {
    algorithm: &'static str,
    input: &'a [PathBuf],
    seed_model: &'a std::path::Path,
    vocab_size: u32,
    iterations: usize,
}

fn train_mingram(
    config: &EffectiveConfig,
    inputs: &[PathBuf],
    options: MingramConfig,
    store: &Store,
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<TrainerOutput> {
    let started = Instant::now();
    let paths = TrainerPaths::from_config(config);
    ensure!(
        config.sentencepiece.vocab_size > 0,
        "vocab_size must be positive"
    );
    let key = checkpoint::key(&(1, &store.identity, &config.sentencepiece))?;
    let artifacts = [
        paths.model.clone(),
        paths.vocab.clone(),
        paths.model.with_extension("mingram.json"),
    ];
    if let Some(saved) = store.read::<CompletedTraining>("mingram", &key)? {
        saved.artifacts.verify(&artifacts)?;
        ensure!(
            saved.output.model == paths.model
                && saved.output.vocab == paths.vocab
                && saved.output.status == TrainerStatus::Succeeded,
            "invalid completed MinGram state"
        );
        Model::load(&saved.output.model)?;
        progress
            .stage("reusing completed MinGram")
            .finish("MinGram model verified");
        return Ok(saved.output);
    }
    let intent = store.read::<String>("export-intent", &key)?;
    if intent.is_none() {
        ensure!(
            artifacts.iter().all(|p| !p.exists()),
            "untracked MinGram artifacts exist; use a new model_prefix"
        );
    }
    let seed_config = seed_config(config)?;
    let mut output = cached_native(
        &seed_config,
        inputs,
        store,
        "seed-artifacts",
        progress,
        cancellation,
    )?;
    let request = MingramRequest {
        algorithm: "mingram",
        input: inputs,
        seed_model: &output.model,
        vocab_size: config.sentencepiece.vocab_size,
        iterations: options.iterations.get(),
    };
    fs::write(&paths.request, serde_json::to_vec_pretty(&request)?)?;
    let seed = Model::load(&output.model)?;
    let runtime = mingram::Runtime::new(config.num_threads as usize, progress, Some(store))?;
    let trained = mingram::train_with_runtime(
        &seed,
        inputs,
        &mingram::Options {
            vocab_size: config.sentencepiece.vocab_size as usize,
            iterations: options.iterations.get(),
            max_record_bytes: config.sentencepiece.max_sentence_length as usize,
            quiet: false,
        },
        &runtime,
        cancellation,
    )?;
    cancellation.check()?;
    let fingerprint = trained.model.fingerprint();
    ensure!(
        intent
            .as_ref()
            .is_none_or(|previous| previous == &fingerprint),
        "interrupted export model differs"
    );
    store.write("export-intent", &key, &fingerprint)?;
    mingram::save_atomic(&trained, &paths.model)?;
    output.model = paths.model;
    output.vocab = paths.vocab;
    output.elapsed_ms = started.elapsed().as_millis();
    fs::write(&paths.output, serde_json::to_vec_pretty(&output)?)?;
    store.write(
        "mingram",
        &key,
        &CompletedTraining {
            artifacts: Artifacts::capture(&artifacts)?,
            output: output.clone(),
        },
    )?;
    progress
        .stage("exporting MinGram")
        .finish("MinGram model and checkpoint written");
    Ok(output)
}

fn seed_config(config: &EffectiveConfig) -> Result<EffectiveConfig> {
    let mut seed = config.clone();
    let size = u64::from(config.sentencepiece.vocab_size)
        .checked_mul(115)
        .context("seed size overflow")?
        .div_ceil(100);
    seed.sentencepiece.algorithm = TrainingAlgorithm::Bpe;
    seed.sentencepiece.vocab_size = u32::try_from(size)?;
    seed.output.work_dir = config
        .output
        .work_dir
        .join(format!("{}.seed", config.output.model_prefix));
    seed.output.model_prefix = "seed".to_owned();
    Ok(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn new_native_prefix_ignores_another_models_shared_request() {
        let temp = TempDir::new().unwrap();
        let mut config = trainer::tests::config(temp.path());
        config.output.model_prefix = "first".into();
        let previous = native_request(&config, &[]);
        let paths = TrainerPaths::from_config(&config);
        fs::write(&paths.request, serde_json::to_vec(&previous).unwrap()).unwrap();
        config.output.model_prefix = "second".into();
        let paths = TrainerPaths::from_config(&config);
        let expected = native_request(&config, &[]);
        assert!(legacy_native_output(&paths, &expected, false)
            .unwrap()
            .is_none());
        fs::write(&paths.model, b"untracked model").unwrap();
        assert!(legacy_native_output(&paths, &expected, false).is_err());
    }

    #[test]
    fn seed_configuration_is_bpe_without_mutating_the_final_algorithm() {
        let mut config = trainer::tests::config(std::path::Path::new("unused"));
        config.sentencepiece.algorithm = TrainingAlgorithm::Mingram {
            mingram: MingramConfig::default(),
        };
        config.sentencepiece.vocab_size = 512;
        let seed = seed_config(&config).unwrap();
        assert_eq!(seed.sentencepiece.algorithm, TrainingAlgorithm::Bpe);
        assert_eq!(seed.sentencepiece.vocab_size, 589);
        assert!(matches!(
            config.sentencepiece.algorithm,
            TrainingAlgorithm::Mingram { .. }
        ));
        assert_ne!(seed.output.work_dir, config.output.work_dir);
    }

    #[test]
    fn native_bridge_refuses_mingram_before_spawning_python() {
        let mut config = trainer::tests::config(std::path::Path::new("unused"));
        config.sentencepiece.algorithm = TrainingAlgorithm::Mingram {
            mingram: MingramConfig::default(),
        };
        let error = trainer::train_sentencepiece(
            &config,
            &[],
            &ProgressReporter::silent(),
            &CancellationToken::default(),
        )
        .unwrap_err();
        assert!(matches!(error, trainer::TrainerError::UnsupportedAlgorithm));
    }

    #[test]
    #[ignore = "requires uv-managed Python seed training; included in just check"]
    fn mingram_config_workflow() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("corpus.txt");
        fs::write(
            &input,
            "alpha beta gamma\nbeta alpha delta\ngamma delta alpha\nก้กา ก้กี\n中文文字 中文\n",
        )
        .unwrap();
        let mut config = trainer::tests::config(&temp.path().join("work"));
        config.sentencepiece.algorithm = TrainingAlgorithm::Mingram {
            mingram: MingramConfig::default(),
        };
        config.sentencepiece.vocab_size = 32;
        config.sentencepiece.byte_fallback = false;
        config.sentencepiece.character_coverage = 1.0;
        config.sentencepiece.user_defined_symbols.clear();
        config.sentencepiece.input_sentence_size = 0;
        config.num_threads = 1;
        let test = temp.path().join("test.txt");
        fs::write(&test, "alpha delta beta\n中文文字\nก้กี\n").unwrap();
        let split = SplitManifest::create(
            &[input],
            &[test],
            config.evaluation.clone(),
            "test-settings".into(),
            crate::corpus_split::SourceRecords {
                training: 5,
                test: 3,
            },
            &ProgressReporter::silent(),
            &CancellationToken::default(),
        )
        .unwrap();
        let legacy_seed = trainer::train_sentencepiece(
            &seed_config(&config).unwrap(),
            &split.train_paths(),
            &ProgressReporter::silent(),
            &CancellationToken::default(),
        )
        .unwrap();
        let seed_modified = fs::metadata(&legacy_seed.model)
            .unwrap()
            .modified()
            .unwrap();
        let output = train_and_report(
            &config,
            &split,
            &ProgressReporter::silent(),
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(
            fs::metadata(&legacy_seed.model)
                .unwrap()
                .modified()
                .unwrap(),
            seed_modified
        );
        let original = fs::read(&output.model).unwrap();
        config.num_threads = 4;
        let resumed = train_and_report(
            &config,
            &split,
            &ProgressReporter::silent(),
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(fs::read(&resumed.model).unwrap(), original);
        // Simulate a kill during export, before the completion marker was published.
        fs::remove_file(
            config
                .output
                .work_dir
                .join("ocr_tokenizer.checkpoints/mingram.json"),
        )
        .unwrap();
        fs::write(&output.model, "partial model").unwrap();
        let recovered = train_and_report(
            &config,
            &split,
            &ProgressReporter::silent(),
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(fs::read(&recovered.model).unwrap(), original);
        let mut incompatible = config.clone();
        incompatible.sentencepiece.algorithm = TrainingAlgorithm::Mingram {
            mingram: MingramConfig {
                iterations: std::num::NonZeroUsize::new(3).unwrap(),
            },
        };
        assert!(train_and_report(
            &incompatible,
            &split,
            &ProgressReporter::silent(),
            &CancellationToken::default()
        )
        .is_err());
        let report_path = config.output.work_dir.join("reports/tokenizer.json");
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(report_path).unwrap()).unwrap();
        assert_eq!(report["evaluation"]["total"]["counts"]["records"], 3);
        assert_eq!(report["evaluation"]["same_vocab_size"], true);
        assert_eq!(
            report["evaluation"]["total"]["counts"]["candidate"]["raw_round_trip_failures"],
            0
        );
        let markdown =
            fs::read_to_string(config.output.work_dir.join("reports/tokenizer.md")).unwrap();
        assert!(markdown.contains("Evaluation checks: PASS"));
        assert!(markdown.contains("P95 tokens per record"));
        let model = Model::load(&output.model).unwrap();
        assert_eq!(model.proto.pieces.len(), 32);
        assert_eq!(
            model.proto.trainer_spec.as_ref().unwrap().model_type(),
            sentencepiece_model::ModelType::Unigram
        );
        let paths = TrainerPaths::from_config(&config);
        assert!(paths.output.exists());
        assert!(output.model.with_extension("mingram.json").exists());
        let request = fs::read_to_string(&paths.request).unwrap();
        assert!(request.contains("\"algorithm\": \"mingram\""));
        let seed_paths = TrainerPaths::from_config(&seed_config(&config).unwrap());
        let seed_request = fs::read_to_string(&seed_paths.request).unwrap();
        assert!(seed_request.contains("\"model_type\": \"bpe\""));
    }
}
