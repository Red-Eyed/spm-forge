//! Content-only, bounded-memory sampling shared by native trainers.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use unicode_script::{Script, UnicodeScript};

use crate::cancel::CancellationToken;
use crate::config::EffectiveConfig;
use crate::corpus_split::SplitManifest;
use crate::progress::ProgressReporter;
use crate::tokenizer::{
    checkpoint::{self, Store},
    records,
};

#[derive(Clone, Default, Serialize, Deserialize)]
struct Census {
    records: u64,
    codepoints: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct Scan {
    next_file: usize,
    groups: BTreeMap<String, Census>,
}

#[derive(Default, Serialize, Deserialize)]
struct Selected {
    records: u64,
    codepoints: u64,
    seen: u64,
    rng_position: u128,
}

#[derive(Serialize, Deserialize)]
struct SampleFile {
    path: PathBuf,
    blake3: String,
    bytes: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct Selection {
    files: Vec<SampleFile>,
    groups: BTreeMap<String, Selected>,
}

#[derive(Serialize, Deserialize)]
pub struct GroupReport {
    pub script: String,
    pub input_records: u64,
    pub input_codepoints: u64,
    pub target_records: u64,
    pub expected_codepoints: f64,
    pub sampled_records: u64,
    pub sampled_codepoints: u64,
}

#[derive(Serialize, Deserialize)]
pub struct Report {
    pub version: u32,
    pub seed: u64,
    pub requested_records: u64,
    pub sampled_records: u64,
    pub groups: Vec<GroupReport>,
    pub files: Vec<PathBuf>,
}

struct Classified {
    script: String,
    codepoints: u64,
}

fn classify(text: &str) -> Classified {
    let mut first = None;
    let mut mixed = false;
    let mut codepoints = 0;
    for character in text.chars() {
        codepoints += 1;
        let script = character.script();
        if matches!(script, Script::Common | Script::Inherited | Script::Unknown) {
            continue;
        }
        match first {
            None => first = Some(script),
            Some(previous) if previous != script => mixed = true,
            _ => {}
        }
    }
    let script = match (mixed, first) {
        (true, _) => "mixed".to_owned(),
        (false, None) => "common".to_owned(),
        (false, Some(script)) => format!("{script:?}").to_lowercase(),
    };
    Classified { script, codepoints }
}

/// Allocate sqrt(codepoint mass), converted to records using each group's mean length.
/// Saturated groups keep all available records; no duplicates are manufactured.
fn quotas(groups: &BTreeMap<String, Census>, requested: u64) -> Result<BTreeMap<String, u64>> {
    let total: u64 = groups.values().map(|g| g.records).sum();
    let budget = requested.min(total);
    ensure!(
        budget > 0,
        "script sampling requires nonempty training text and a positive budget"
    );
    ensure!(
        budget >= groups.len() as u64,
        "sample budget is too small to retain every script group"
    );
    let mut targets: BTreeMap<_, _> = groups.keys().map(|name| (name.clone(), 0)).collect();
    let weights: BTreeMap<_, _> = groups
        .iter()
        .map(|(name, stats)| {
            // Empty records remain eligible without introducing a zero denominator.
            let mass = stats.codepoints.max(stats.records) as f64;
            (name.clone(), stats.records as f64 / mass.sqrt())
        })
        .collect();
    let mut available = budget;
    let mut active: Vec<_> = groups.keys().cloned().collect();
    while !active.is_empty() {
        let sum: f64 = active.iter().map(|name| weights[name]).sum();
        let saturated: Vec<_> = active
            .iter()
            .filter(|name| available as f64 * weights[*name] / sum >= groups[*name].records as f64)
            .cloned()
            .collect();
        if saturated.is_empty() {
            let mut remainders = Vec::new();
            for name in &active {
                let exact = available as f64 * weights[name] / sum;
                let count = (exact.floor() as u64).min(groups[name].records);
                targets.insert(name.clone(), count);
                remainders.push((name.clone(), exact - count as f64));
            }
            let used: u64 = targets.values().sum();
            remainders.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (name, _) in remainders.into_iter().take((budget - used) as usize) {
                *targets.get_mut(&name).expect("known script") += 1;
            }
            break;
        }
        for name in &saturated {
            targets.insert(name.clone(), groups[name].records);
            available -= groups[name].records;
        }
        active.retain(|name| !saturated.contains(name));
    }
    // Preserve tiny groups even when largest-remainder rounding gave them zero.
    for name in groups.keys() {
        if targets[name] != 0 {
            continue;
        }
        let donor = targets
            .iter()
            .filter(|(_, n)| **n > 1)
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(name, _)| name.clone())
            .expect("budget covers all groups");
        *targets.get_mut(&donor).expect("known donor") -= 1;
        targets.insert(name.clone(), 1);
    }
    Ok(targets)
}

struct Inputs<'a> {
    paths: Vec<PathBuf>,
    bytes: Vec<u64>,
    max_record_bytes: usize,
    pool: rayon::ThreadPool,
    progress: &'a ProgressReporter,
    cancellation: &'a CancellationToken,
}

impl Inputs<'_> {
    fn classify_batch(&self, batch: &records::Batch) -> Vec<Classified> {
        self.pool.install(|| {
            batch
                .records
                .par_iter()
                .map(|text| classify(text))
                .collect()
        })
    }
}

struct Plan<'a> {
    groups: &'a BTreeMap<String, Census>,
    targets: &'a BTreeMap<String, u64>,
    seed: u64,
}

/// Cache input census and exact record selection without retaining selected text in RAM.
pub fn prepare(
    config: &EffectiveConfig,
    split: &SplitManifest,
    progress: &ProgressReporter,
    cancellation: &CancellationToken,
) -> Result<Report> {
    let requested = match config.sentencepiece.input_sentence_size {
        0 => config.balancing.total_lines,
        limit => limit.min(config.balancing.total_lines),
    };
    let seed = config.balancing.shuffle_seed;
    let key = checkpoint::key(&(
        1,
        &split.train,
        requested,
        seed,
        config.sentencepiece.max_sentence_length,
    ))?;
    let directory = config
        .output
        .work_dir
        .join(format!("{}.script-sample", config.output.model_prefix))
        .join(&key);
    let store = Store::new(directory.clone(), key.clone());
    let paths = split.train_paths();
    let bytes = paths
        .iter()
        .map(|p| p.metadata().map(|m| m.len()))
        .collect::<std::io::Result<Vec<_>>>()?;
    let inputs = Inputs {
        paths,
        bytes,
        max_record_bytes: config.sentencepiece.max_sentence_length as usize,
        pool: rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads as usize)
            .build()?,
        progress,
        cancellation,
    };
    let scan = census(&inputs, &store, &key)?;
    let targets = quotas(&scan.groups, requested)?;
    let plan = Plan {
        groups: &scan.groups,
        targets: &targets,
        seed,
    };
    let selected = select(&inputs, &store, &key, &directory, &plan)?;
    let report = make_report(&plan, &selected, requested)?;
    checkpoint::atomic_write(
        &config.output.work_dir.join("reports/script_sample.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

fn census(inputs: &Inputs<'_>, store: &Store, key: &str) -> Result<Scan> {
    let mut scan = store.read::<Scan>("census", key)?.unwrap_or_default();
    ensure!(
        scan.next_file <= inputs.paths.len(),
        "invalid script census cursor"
    );
    let stage = inputs.progress.stage_bar(
        "counting training content by script",
        inputs.bytes.iter().sum(),
    );
    for (index, path) in inputs.paths.iter().enumerate() {
        if index < scan.next_file {
            stage.inc(inputs.bytes[index]);
            continue;
        }
        records::batches(
            path,
            inputs.max_record_bytes,
            inputs.cancellation,
            |batch| {
                for row in inputs.classify_batch(&batch) {
                    let group = scan.groups.entry(row.script).or_default();
                    group.records += 1;
                    group.codepoints += row.codepoints;
                }
                stage.inc(batch.bytes);
                Ok(())
            },
        )?;
        scan.next_file = index + 1;
        store.write("census", key, &scan)?;
    }
    stage.finish(format!("counted {} script groups", scan.groups.len()));
    Ok(scan)
}

fn verify_sample(inputs: &Inputs<'_>, selected: &Selection) -> Result<()> {
    ensure!(
        selected.files.len() <= inputs.paths.len(),
        "invalid script sample cursor"
    );
    let stage = inputs.progress.stage_bar(
        "verifying saved script sample",
        selected.files.iter().map(|f| f.bytes).sum(),
    );
    for file in &selected.files {
        ensure!(
            digest(&file.path, |n| stage.inc(n), inputs.cancellation)? == file.blake3,
            "saved script sample changed: {}",
            file.path.display()
        );
    }
    stage.finish("saved sample parts verified");
    Ok(())
}

fn generators(plan: &Plan<'_>, selected: &mut Selection) -> BTreeMap<String, ChaCha20Rng> {
    plan.groups
        .keys()
        .map(|name| {
            let mut hash = blake3::Hasher::new();
            hash.update(&plan.seed.to_le_bytes());
            hash.update(name.as_bytes());
            let mut rng = ChaCha20Rng::from_seed(*hash.finalize().as_bytes());
            let state = selected.groups.entry(name.clone()).or_default();
            rng.set_word_pos(state.rng_position);
            (name.clone(), rng)
        })
        .collect()
}

fn select(
    inputs: &Inputs<'_>,
    store: &Store,
    key: &str,
    directory: &Path,
    plan: &Plan<'_>,
) -> Result<Selection> {
    let mut selected = store
        .read::<Selection>("selection", key)?
        .unwrap_or_default();
    verify_sample(inputs, &selected)?;
    let mut random = generators(plan, &mut selected);
    let stage = inputs.progress.stage_bar(
        "selecting script-balanced training sample",
        inputs.bytes.iter().sum(),
    );
    fs::create_dir_all(directory)?;
    for index in 0..inputs.paths.len() {
        if index < selected.files.len() {
            stage.inc(inputs.bytes[index]);
            continue;
        }
        let path = directory.join(format!("part-{index:06}.txt"));
        let file = select_file(
            inputs,
            index,
            &path,
            &mut selected,
            &mut random,
            plan,
            &stage,
        )?;
        selected.files.push(file);
        for (name, rng) in &random {
            selected
                .groups
                .get_mut(name)
                .expect("known group")
                .rng_position = rng.get_word_pos();
        }
        store.write("selection", key, &selected)?;
    }
    stage.finish(format!(
        "sampled {} training records",
        selected.groups.values().map(|s| s.records).sum::<u64>()
    ));
    Ok(selected)
}

fn take(
    row: &Classified,
    plan: &Plan<'_>,
    selected: &mut Selection,
    random: &mut BTreeMap<String, ChaCha20Rng>,
) -> Result<bool> {
    let state = selected
        .groups
        .get_mut(&row.script)
        .ok_or_else(|| anyhow::anyhow!("script groups changed after census"))?;
    let remaining = plan.groups[&row.script]
        .records
        .checked_sub(state.seen)
        .filter(|n| *n > 0)
        .ok_or_else(|| anyhow::anyhow!("training records changed after script census"))?;
    let needed = plan.targets[&row.script]
        .checked_sub(state.records)
        .ok_or_else(|| anyhow::anyhow!("invalid selected script count"))?;
    let keep = random
        .get_mut(&row.script)
        .expect("seeded group")
        .gen_range(0..remaining)
        < needed;
    state.seen += 1;
    if keep {
        state.records += 1;
        state.codepoints += row.codepoints;
    }
    Ok(keep)
}

fn select_file(
    inputs: &Inputs<'_>,
    index: usize,
    path: &Path,
    selected: &mut Selection,
    random: &mut BTreeMap<String, ChaCha20Rng>,
    plan: &Plan<'_>,
    stage: &crate::progress::StageProgress,
) -> Result<SampleFile> {
    let pending = path.with_extension("txt.pending");
    let mut writer = BufWriter::with_capacity(1024 * 1024, File::create(&pending)?);
    let mut hash = blake3::Hasher::new();
    let mut bytes = 0;
    records::batches(
        &inputs.paths[index],
        inputs.max_record_bytes,
        inputs.cancellation,
        |batch| {
            for (text, row) in batch.records.iter().zip(inputs.classify_batch(&batch)) {
                if !take(&row, plan, selected, random)? {
                    continue;
                }
                writer.write_all(text.as_bytes())?;
                writer.write_all(b"\n")?;
                hash.update(text.as_bytes());
                hash.update(b"\n");
                bytes += text.len() as u64 + 1;
            }
            stage.inc(batch.bytes);
            Ok(())
        },
    )?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(pending, path)?;
    Ok(SampleFile {
        path: path.to_owned(),
        blake3: hash.finalize().to_hex().to_string(),
        bytes,
    })
}

fn make_report(plan: &Plan<'_>, selected: &Selection, requested: u64) -> Result<Report> {
    for (name, census) in plan.groups {
        ensure!(
            selected.groups[name].seen == census.records
                && selected.groups[name].records == plan.targets[name],
            "script sample quota mismatch"
        );
    }
    Ok(Report {
        version: 1,
        seed: plan.seed,
        requested_records: requested,
        sampled_records: selected.groups.values().map(|g| g.records).sum(),
        groups: plan
            .groups
            .iter()
            .map(|(name, census)| GroupReport {
                script: name.clone(),
                input_records: census.records,
                input_codepoints: census.codepoints,
                target_records: plan.targets[name],
                expected_codepoints: plan.targets[name] as f64 * census.codepoints as f64
                    / census.records as f64,
                sampled_records: selected.groups[name].records,
                sampled_codepoints: selected.groups[name].codepoints,
            })
            .collect(),
        files: selected
            .files
            .iter()
            .filter(|f| f.bytes > 0)
            .map(|f| f.path.clone())
            .collect(),
    })
}

fn digest(
    path: &Path,
    mut progress: impl FnMut(u64),
    cancellation: &CancellationToken,
) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = blake3::Hasher::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        cancellation.check()?;
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        progress(n as u64);
    }
    Ok(hash.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus_split::CorpusFile;
    use tempfile::TempDir;

    fn inputs<'a>(
        paths: Vec<PathBuf>,
        progress: &'a ProgressReporter,
        cancellation: &'a CancellationToken,
        threads: usize,
    ) -> Inputs<'a> {
        Inputs {
            bytes: paths.iter().map(|p| p.metadata().unwrap().len()).collect(),
            paths,
            max_record_bytes: 1024,
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap(),
            progress,
            cancellation,
        }
    }

    #[test]
    fn content_classification_has_no_language_or_filename_input() {
        assert_eq!(classify("hello bonjour café").script, "latin");
        assert_eq!(classify("ภาษาไทย").script, "thai");
        assert_eq!(classify("ก้").codepoints, 2);
        assert_eq!(classify("中文 hello").script, "mixed");
        assert_eq!(classify("123 !😀").script, "common");
    }

    #[test]
    fn quotas_balance_codepoints_account_for_length_and_cap_supply() {
        let groups = BTreeMap::from([
            (
                "latin".into(),
                Census {
                    records: 10000,
                    codepoints: 10000,
                },
            ),
            (
                "thai".into(),
                Census {
                    records: 100,
                    codepoints: 1000,
                },
            ),
        ]);
        let target = quotas(&groups, 1010).unwrap();
        assert_eq!(target.values().sum::<u64>(), 1010);
        let ratio = target["latin"] as f64 / (target["thai"] as f64 * 10.0);
        assert!((ratio - 10_f64.sqrt()).abs() < 0.05);
        let all = quotas(&groups, 20000).unwrap();
        assert_eq!(all["latin"], 10000);
        assert_eq!(all["thai"], 100);
        assert!(quotas(&groups, 1).is_err());
        assert!(quotas(&BTreeMap::new(), 100).is_err());
    }

    #[test]
    fn sampler_is_chunked_rename_independent_and_resumes_exactly() {
        let temp = TempDir::new().unwrap();
        let a = temp.path().join("english.txt");
        let b = temp.path().join("french.txt");
        fs::write(&a, "hello\nภาษาไทย\n".repeat(records::BATCH_RECORDS + 7)).unwrap();
        fs::write(
            &b,
            "bonjour\n中文 hello\n123\n".repeat(records::BATCH_RECORDS + 9),
        )
        .unwrap();
        let progress = ProgressReporter::silent();
        let cancellation = CancellationToken::default();
        let inputs = inputs(vec![a.clone(), b.clone()], &progress, &cancellation, 4);
        let store = Store::new(temp.path().join("cache"), "key".into());
        let scan = census(&inputs, &store, "key").unwrap();
        assert_eq!(
            scan.groups.values().map(|g| g.records).sum::<u64>(),
            ((records::BATCH_RECORDS + 7) * 2 + (records::BATCH_RECORDS + 9) * 3) as u64
        );
        let targets = quotas(&scan.groups, 1000).unwrap();
        let plan = Plan {
            groups: &scan.groups,
            targets: &targets,
            seed: 42,
        };
        let directory = temp.path().join("parts");
        fs::create_dir(&directory).unwrap();
        let mut partial = Selection::default();
        let mut rngs = generators(&plan, &mut partial);
        let stage = progress.stage("test");
        let file = select_file(
            &inputs,
            0,
            &directory.join("part-000000.txt"),
            &mut partial,
            &mut rngs,
            &plan,
            &stage,
        )
        .unwrap();
        partial.files.push(file);
        for (name, rng) in &rngs {
            partial.groups.get_mut(name).unwrap().rng_position = rng.get_word_pos();
        }
        store.write("selection", "key", &partial).unwrap();
        let resumed = select(&inputs, &store, "key", &directory, &plan).unwrap();
        let original: Vec<_> = resumed
            .files
            .iter()
            .map(|f| fs::read(&f.path).unwrap())
            .collect();
        let clean_store = Store::new(temp.path().join("clean-cache"), "key".into());
        let clean = select(
            &inputs,
            &clean_store,
            "key",
            &temp.path().join("clean-parts"),
            &plan,
        )
        .unwrap();
        assert_eq!(
            original,
            clean
                .files
                .iter()
                .map(|f| fs::read(&f.path).unwrap())
                .collect::<Vec<_>>()
        );
        let report = make_report(&plan, &resumed, 1000).unwrap();
        assert_eq!(report.sampled_records, 1000);
        for group in report.groups {
            assert!(group.sampled_records > 0);
        }
        let renamed_a = temp.path().join("ar.txt");
        let renamed_b = temp.path().join("th.txt");
        fs::rename(a, &renamed_a).unwrap();
        fs::rename(b, &renamed_b).unwrap();
        let renamed_inputs = self::inputs(vec![renamed_a, renamed_b], &progress, &cancellation, 1);
        let other = Store::new(temp.path().join("renamed-cache"), "key".into());
        let renamed = select(
            &renamed_inputs,
            &other,
            "key",
            &temp.path().join("renamed-parts"),
            &plan,
        )
        .unwrap();
        assert_eq!(
            original,
            renamed
                .files
                .iter()
                .map(|f| fs::read(&f.path).unwrap())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn preparation_reuse_and_cached_sample_never_read_test_text() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("unlabelled.txt");
        fs::write(&input, "hello world\nภาษาไทย\n".repeat(20)).unwrap();
        let mut config = crate::trainer::tests::config(temp.path());
        config.balancing.enabled = false;
        let previous = crate::corpus_split::preparation_hash(&config).unwrap();
        config.balancing.mode = crate::config::BalancingMode::ScriptSample;
        config.balancing.enabled = true;
        assert_eq!(
            previous,
            crate::corpus_split::preparation_hash(&config).unwrap()
        );
        config.sentencepiece.input_sentence_size = 20;
        let split = SplitManifest {
            version: 1,
            policy: config.evaluation.clone(),
            preparation_blake3: previous,
            training_records_before_balancing: 40,
            test_records: 1,
            train: vec![CorpusFile {
                path: input,
                blake3: "test-owned-corpus".into(),
            }],
            test: vec![CorpusFile {
                path: temp.path().join("intentionally-missing-test.txt"),
                blake3: "unused".into(),
            }],
        };
        let progress = ProgressReporter::silent();
        let cancellation = CancellationToken::default();
        let first = prepare(&config, &split, &progress, &cancellation).unwrap();
        let second = prepare(&config, &split, &progress, &cancellation).unwrap();
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
        assert_eq!(first.sampled_records, 20);
        fs::write(&first.files[0], "corrupted").unwrap();
        assert!(prepare(&config, &split, &progress, &cancellation).is_err());
    }
}
