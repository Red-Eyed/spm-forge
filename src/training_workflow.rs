//! Selects the configured training algorithm after corpus preparation.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use crate::cancel::CancellationToken;
use crate::config::{EffectiveConfig, MingramConfig, TrainingAlgorithm};
use crate::progress::ProgressReporter;
use crate::tokenizer::{mingram, model::Model};
use crate::trainer::{self, TrainerOutput, TrainerPaths};

pub fn train(
    config: &EffectiveConfig,
    inputs: &[PathBuf],
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<TrainerOutput> {
    match config.sentencepiece.algorithm {
        TrainingAlgorithm::Bpe => Ok(trainer::train_sentencepiece(
            config,
            inputs,
            progress,
            cancellation,
        )?),
        TrainingAlgorithm::Mingram { mingram } => {
            train_mingram(config, inputs, mingram, progress, cancellation)
        }
    }
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
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<TrainerOutput> {
    let started = Instant::now();
    let paths = TrainerPaths::from_config(config);
    ensure!(
        config.sentencepiece.vocab_size > 0,
        "vocab_size must be positive"
    );
    for path in [
        &paths.model,
        &paths.vocab,
        &paths.model.with_extension("mingram.json"),
    ] {
        ensure!(
            !path.exists(),
            "refusing to overwrite {}; choose a new model_prefix or work_dir",
            path.display()
        );
    }
    let seed_config = seed_config(config)?;
    ensure!(
        !seed_config.output.work_dir.exists(),
        "MinGram seed directory exists; choose a new model_prefix or work_dir"
    );
    let mut output = trainer::train_sentencepiece(&seed_config, inputs, progress, cancellation)?;
    let request = MingramRequest {
        algorithm: "mingram",
        input: inputs,
        seed_model: &output.model,
        vocab_size: config.sentencepiece.vocab_size,
        iterations: options.iterations.get(),
    };
    fs::write(&paths.request, serde_json::to_vec_pretty(&request)?)?;
    let seed = Model::load(&output.model)?;
    let stage = progress.stage("optimizing MinGram vocabulary");
    let trained = mingram::train(
        &seed,
        inputs,
        &mingram::Options {
            vocab_size: config.sentencepiece.vocab_size as usize,
            iterations: options.iterations.get(),
            max_record_bytes: config.sentencepiece.max_sentence_length as usize,
            quiet: true,
        },
        cancellation,
    )?;
    cancellation.check()?;
    mingram::save(&trained, &paths.model)?;
    output.model = paths.model;
    output.vocab = paths.vocab;
    output.elapsed_ms = started.elapsed().as_millis();
    fs::write(&paths.output, serde_json::to_vec_pretty(&output)?)?;
    stage.finish("MinGram model and evaluation metadata written");
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
        let output = train(
            &config,
            &[input],
            &ProgressReporter::silent(),
            &CancellationToken::default(),
        )
        .unwrap();
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
