//! Small, test-owned multilingual model fixtures; no external data dependencies.

use std::fs;

use prost::Message;
use sentencepiece_model::{
    ModelProto, ModelType, NormalizerSpec, SentencePiece, TrainerSpec, Type,
};
use tempfile::TempDir;

use super::mingram::{self, Options};
use super::{
    evaluate::{self, Ratio},
    model::Model,
    records,
};
use crate::cancel::CancellationToken;

fn options(vocab_size: usize) -> Options {
    Options {
        vocab_size,
        iterations: 2,
        max_record_bytes: 1024,
        quiet: true,
    }
}

pub(super) fn proto(pieces: &[&str]) -> ModelProto {
    let mut vocabulary = vec![SentencePiece {
        piece: Some("<unk>".to_owned()),
        score: Some(0.0),
        r#type: Some(Type::Unknown as i32),
    }];
    vocabulary.extend(pieces.iter().map(|piece| SentencePiece {
        piece: Some((*piece).to_owned()),
        score: Some(-1.0),
        r#type: Some(Type::Normal as i32),
    }));
    ModelProto {
        trainer_spec: Some(TrainerSpec {
            model_type: Some(ModelType::Unigram as i32),
            vocab_size: Some(vocabulary.len() as i32),
            unk_id: Some(0),
            bos_id: Some(-1),
            eos_id: Some(-1),
            pad_id: Some(-1),
            ..Default::default()
        }),
        normalizer_spec: Some(NormalizerSpec {
            name: Some("identity".to_owned()),
            add_dummy_prefix: Some(false),
            remove_extra_whitespaces: Some(false),
            escape_whitespaces: Some(true),
            ..Default::default()
        }),
        pieces: vocabulary,
        ..Default::default()
    }
}

pub(super) fn model(pieces: &[&str]) -> Model {
    Model::from_bytes(&proto(pieces).encode_to_vec()).expect("synthetic model")
}

#[test]
fn paired_counts_measure_known_multilingual_savings() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("heldout.txt");
    fs::write(&path, "ab\r\nก้\n中文\n\n").unwrap();
    let baseline = model(&["a", "b", "ก", "้", "中", "文"]);
    let candidate = model(&["ab", "ก้", "中文", "a", "ก", "中"]);
    let report = evaluate::compare(
        &baseline,
        &candidate,
        &[path],
        64,
        true,
        &CancellationToken::default(),
    )
    .unwrap();
    assert!(report.same_vocab_size);
    assert_eq!(report.total.counts.records, 4);
    assert_eq!(report.total.counts.empty_records, 1);
    assert_eq!(report.total.counts.baseline.tokens, 6);
    assert_eq!(report.total.counts.candidate.tokens, 3);
    assert_eq!(
        report.total.token_count_ratio,
        Ratio::Defined { value: 0.5 }
    );
    assert_eq!(report.total.counts.improved_records, 3);
    assert_eq!(report.total.counts.candidate.raw_round_trip_failures, 0);
}

#[test]
fn empty_corpus_has_undefined_ratio_and_valid_json() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("empty.txt");
    fs::write(&path, "").unwrap();
    let model = model(&["a"]);
    let report = evaluate::compare(
        &model,
        &model,
        &[path],
        64,
        true,
        &CancellationToken::default(),
    )
    .unwrap();
    assert!(matches!(
        report.total.token_count_ratio,
        Ratio::Undefined { .. }
    ));
    assert!(serde_json::to_string(&report)
        .unwrap()
        .contains("undefined"));
}

#[test]
fn unknown_ids_are_not_mistaken_for_lossless_compression() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("unknown.txt");
    fs::write(&path, "ab\n").unwrap();
    let model = model(&["a"]);
    let report = evaluate::compare(
        &model,
        &model,
        &[path],
        64,
        true,
        &CancellationToken::default(),
    )
    .unwrap();
    assert_eq!(report.total.counts.candidate.unknown_tokens, 1);
    assert_eq!(report.total.counts.candidate.raw_round_trip_failures, 1);
    assert_eq!(
        report.total.counts.candidate.normalized_round_trip_failures,
        1
    );
}

#[test]
fn normalization_changes_are_reported_separately() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("spaces.txt");
    fs::write(&path, "  a  \n").unwrap();
    let mut proto = proto(&["a", "▁"]);
    proto
        .normalizer_spec
        .as_mut()
        .unwrap()
        .remove_extra_whitespaces = Some(true);
    let model = Model::from_bytes(&proto.encode_to_vec()).unwrap();
    let report = evaluate::compare(
        &model,
        &model,
        &[path],
        64,
        true,
        &CancellationToken::default(),
    )
    .unwrap();
    assert_eq!(report.total.counts.candidate.raw_round_trip_failures, 1);
    assert_eq!(
        report.total.counts.candidate.normalized_round_trip_failures,
        0
    );
}

#[test]
fn incompatible_normalization_is_rejected() {
    let baseline = model(&["a"]);
    let mut proto = proto(&["a"]);
    proto.normalizer_spec.as_mut().unwrap().add_dummy_prefix = Some(true);
    let candidate = Model::from_bytes(&proto.encode_to_vec()).unwrap();
    let error = evaluate::compare(
        &baseline,
        &candidate,
        &["unused".into()],
        64,
        true,
        &CancellationToken::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("not comparable"));
}

#[test]
fn bounded_reader_rejects_oversized_records_before_visiting() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("long.txt");
    fs::write(&path, "abcdefgh\n").unwrap();
    let mut visited = false;
    let error = records::visit(&path, 4, &CancellationToken::default(), |_, _| {
        visited = true;
        Ok(())
    })
    .unwrap_err();
    assert!(!visited);
    assert!(error.to_string().contains("exceeds"));
}

#[test]
fn reader_preserves_whitespace_and_final_record_without_newline() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("records.txt");
    fs::write(&path, " a \r\n\nก้").unwrap();
    let mut records = Vec::new();
    records::visit(&path, 64, &CancellationToken::default(), |text, _| {
        records.push(text.to_owned());
        Ok(())
    })
    .unwrap();
    assert_eq!(records, [" a ", "", "ก้"]);
}

#[test]
fn cancelled_evaluation_stops_before_reading_records() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("records.txt");
    fs::write(&path, "a\n").unwrap();
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let model = model(&["a"]);
    assert!(evaluate::compare(&model, &model, &[path], 64, true, &cancellation).is_err());
}

#[test]
fn invalid_utf8_is_an_error_not_a_skipped_record() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("invalid.txt");
    fs::write(&path, [0xff, b'\n']).unwrap();
    assert!(records::visit(&path, 64, &CancellationToken::default(), |_, _| Ok(())).is_err());
}

#[test]
fn mingram_preserves_clusters_and_reduces_known_token_counts() {
    let temp = TempDir::new().unwrap();
    let train = temp.path().join("train.txt");
    let heldout = temp.path().join("heldout.txt");
    fs::write(&train, "ab\nก้\n中文\nab\nก้\n中文\n").unwrap();
    fs::write(&heldout, "abก้中文\n中文abก้\n").unwrap();
    let seed = model(&["a", "b", "ก", "้", "中", "文", "ab", "ก้", "中文", "bก"]);
    let trained =
        mingram::train(&seed, &[train], &options(10), &CancellationToken::default()).unwrap();
    let baseline = model(&["a", "b", "ก", "้", "中", "文", "aa", "bb", "中中"]);
    let report = evaluate::compare(
        &baseline,
        &trained.model,
        &[heldout],
        1024,
        true,
        &CancellationToken::default(),
    )
    .unwrap();
    assert_eq!(trained.model.proto.pieces.len(), 10);
    assert_eq!(report.total.counts.baseline.tokens, 12);
    assert_eq!(report.total.counts.candidate.tokens, 6);
    assert_eq!(report.total.counts.candidate.raw_round_trip_failures, 0);
    assert!(trained.model.proto.pieces.iter().any(|p| p.piece() == "ก้"));
    assert_eq!(trained.report.passes.len(), 4);
}

#[test]
fn mingram_improves_bpe_segmentation_without_changing_vocabulary() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("train.txt");
    fs::write(&path, "abcd\nabcd\n").unwrap();
    let mut proto = proto(&["a", "b", "c", "d", "ab", "bc", "bcd"]);
    proto.trainer_spec.as_mut().unwrap().model_type = Some(ModelType::Bpe as i32);
    proto.pieces[5].score = Some(0.0);
    proto.pieces[6].score = Some(-1.0);
    proto.pieces[7].score = Some(-2.0);
    let seed = Model::from_bytes(&proto.encode_to_vec()).unwrap();
    assert_eq!(seed.processor.encode("abcd").unwrap().len(), 3);
    let trained = mingram::train(
        &seed,
        std::slice::from_ref(&path),
        &options(8),
        &CancellationToken::default(),
    )
    .unwrap();
    let report = evaluate::compare(
        &seed,
        &trained.model,
        &[path],
        1024,
        true,
        &CancellationToken::default(),
    )
    .unwrap();
    assert!(report.same_vocab_size);
    assert_eq!(report.total.counts.baseline.tokens, 6);
    assert_eq!(report.total.counts.candidate.tokens, 4);
    assert_eq!(report.total.counts.candidate.raw_round_trip_failures, 0);
}

#[test]
fn comparison_rejects_different_unknown_normalizer_extensions() {
    use super::wire;
    let bytes = proto(&["a"]).encode_to_vec();
    let fields = wire::fields(&bytes).unwrap();
    let mut normalizer = fields.iter().find(|f| f.tag == 3).unwrap().payload.to_vec();
    normalizer.extend(wire::message(200, b"new normalization setting"));
    let bytes_with_extension = wire::patch(&bytes, &[(3, &wire::message(3, &normalizer))]).unwrap();
    let baseline = Model::from_bytes(&bytes).unwrap();
    let candidate = Model::from_bytes(&bytes_with_extension).unwrap();
    assert!(!baseline.same_normalization(&candidate));
}

#[test]
fn native_mingram_counts_match_exhaustive_shortest_segmentations() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("train.txt");
    fs::write(&path, "ab\nabc\nbc\nก้\n中文\n").unwrap();
    let seed = model(&["a", "b", "c", "ก", "้", "中", "文", "ab", "bc", "ก้", "中文"]);
    let trained = mingram::train(
        &seed,
        &[path],
        &options(seed.proto.pieces.len()),
        &CancellationToken::default(),
    )
    .unwrap();
    for text in ["abc", "ababc", "bca", "ก้ก้", "中文中文", "abcก้中文"] {
        let expected = minimum_tokens(text, &trained.model);
        assert_eq!(
            trained.model.processor.encode(text).unwrap().len(),
            expected,
            "{text}"
        );
    }
}

fn minimum_tokens(text: &str, model: &Model) -> usize {
    if text.is_empty() {
        return 0;
    }
    model
        .proto
        .pieces
        .iter()
        .filter(|p| p.r#type() == Type::Normal)
        .filter_map(|p| text.strip_prefix(p.piece()))
        .map(|tail| 1 + minimum_tokens(tail, model))
        .min()
        .expect("fixture has complete character coverage")
}

#[test]
fn probability_term_selects_frequent_piece_among_equal_token_counts() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("train.txt");
    fs::write(&path, "ab\nab\nab\nab\nc\nc\nc\nc\nabc\n").unwrap();
    let seed = model(&["a", "b", "c", "ab", "bc"]);
    let trained =
        mingram::train(&seed, &[path], &options(6), &CancellationToken::default()).unwrap();
    let pieces: Vec<_> = trained
        .model
        .processor
        .encode("abc")
        .unwrap()
        .into_iter()
        .map(|p| p.piece)
        .collect();
    assert_eq!(pieces, ["ab", "c"]);
}

#[test]
fn nonprefix_reserved_ids_remain_pinned_after_pruning() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("train.txt");
    fs::write(&path, "abc\n").unwrap();
    let mut proto = proto(&["a", "b", "<eos>", "ab", "c"]);
    proto.pieces[3].r#type = Some(Type::Control as i32);
    proto.trainer_spec.as_mut().unwrap().eos_id = Some(3);
    proto.trainer_spec.as_mut().unwrap().eos_piece = Some("<eos>".to_owned());
    let seed = Model::from_bytes(&proto.encode_to_vec()).unwrap();
    let trained =
        mingram::train(&seed, &[path], &options(5), &CancellationToken::default()).unwrap();
    assert_eq!(trained.model.processor.eos_id(), Some(3));
    assert_eq!(trained.model.proto.pieces[3].piece(), "<eos>");
    assert_eq!(
        trained
            .model
            .processor
            .decode_piece_ids(&[1, 2, 4])
            .unwrap(),
        "abc"
    );
    assert!(trained
        .report
        .id_mapping
        .iter()
        .any(|m| m.seed_id == 5 && m.output_id == 4));
}

#[test]
fn insufficient_coverage_budget_is_rejected() {
    let seed = model(&["a", "b", "c", "ab"]);
    let error = mingram::train(
        &seed,
        &["unused".into()],
        &options(3),
        &CancellationToken::default(),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("protected"));
}

#[test]
fn training_rejects_lossy_unknown_encoding() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("train.txt");
    fs::write(&path, "ab\n").unwrap();
    let seed = model(&["a"]);
    let error = mingram::train(&seed, &[path], &options(2), &CancellationToken::default())
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("unknown IDs"));
}

pub(super) fn byte_proto() -> ModelProto {
    let mut proto = proto(&["a", "▁"]);
    for byte in 0..=255 {
        proto.pieces.push(SentencePiece {
            piece: Some(format!("<0x{byte:02X}>")),
            score: Some(0.0),
            r#type: Some(Type::Byte as i32),
        });
    }
    let size = proto.pieces.len() as i32;
    let spec = proto.trainer_spec.as_mut().unwrap();
    spec.byte_fallback = Some(true);
    spec.vocab_size = Some(size);
    proto
}

#[test]
fn byte_fallback_is_counted_and_preserved_through_training() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("train.txt");
    fs::write(&path, "a中\n").unwrap();
    let seed = Model::from_bytes(&byte_proto().encode_to_vec()).unwrap();
    let trained = mingram::train(
        &seed,
        std::slice::from_ref(&path),
        &options(seed.proto.pieces.len()),
        &CancellationToken::default(),
    )
    .unwrap();
    let report = evaluate::compare(
        &seed,
        &trained.model,
        &[path],
        1024,
        true,
        &CancellationToken::default(),
    )
    .unwrap();
    assert_eq!(report.total.counts.candidate.records_with_byte_fallback, 1);
    assert_eq!(report.total.counts.candidate.byte_tokens, 3);
    assert_eq!(report.total.counts.candidate.tokens, 4);
    assert_eq!(report.total.counts.candidate.unknown_tokens, 0);
    assert_eq!(report.total.counts.candidate.raw_round_trip_failures, 0);
    assert_eq!(trained.report.passes.last().unwrap().byte_tokens, 3);
}

#[test]
fn model_rewrite_preserves_unknown_extensions_and_removes_stale_self_tests() {
    use super::wire;
    let mut proto = proto(&["a", "ab", "b"]);
    proto.self_test_data = Some(Default::default());
    let mut bytes = proto.encode_to_vec();
    bytes.extend(wire::message(200, b"model extension"));
    let original_trainer = wire::fields(&bytes)
        .unwrap()
        .into_iter()
        .find(|f| f.tag == 2)
        .unwrap()
        .payload
        .to_vec();
    let mut trainer = original_trainer;
    trainer.extend(wire::message(201, b"trainer extension"));
    let trainer = wire::message(2, &trainer);
    bytes = wire::patch(&bytes, &[(2, &trainer)]).unwrap();
    let fields = wire::fields(&bytes).unwrap();
    let mut piece = fields.iter().find(|f| f.tag == 1).unwrap().payload.to_vec();
    piece.extend(wire::message(202, b"piece extension"));
    let mut pieces = wire::message(1, &piece);
    for field in fields.iter().filter(|f| f.tag == 1).skip(1) {
        pieces.extend_from_slice(field.raw);
    }
    bytes = wire::patch(&bytes, &[(1, &pieces)]).unwrap();
    let seed = Model::from_bytes(&bytes).unwrap();
    let selected: Vec<_> = seed.proto.pieces.iter().cloned().enumerate().collect();
    let rewritten = seed.with_pieces(&selected).unwrap();
    let fields = wire::fields(rewritten.serialized()).unwrap();
    assert!(fields
        .iter()
        .any(|f| f.tag == 200 && f.payload == b"model extension"));
    assert!(!fields.iter().any(|f| f.tag == 4));
    let trainer = fields.iter().find(|f| f.tag == 2).unwrap();
    assert!(wire::fields(trainer.payload)
        .unwrap()
        .iter()
        .any(|f| f.tag == 201));
    let piece = fields.iter().find(|f| f.tag == 1).unwrap();
    assert!(wire::fields(piece.payload)
        .unwrap()
        .iter()
        .any(|f| f.tag == 202));
}

#[test]
fn export_reloads_and_refuses_to_overwrite_existing_outputs() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("train.txt");
    fs::write(&path, "ab\n").unwrap();
    let seed = model(&["a", "b", "ab"]);
    let trained =
        mingram::train(&seed, &[path], &options(4), &CancellationToken::default()).unwrap();
    let output = temp.path().join("output.model");
    mingram::save(&trained, &output).unwrap();
    assert!(output.with_extension("vocab").exists());
    assert!(output.with_extension("mingram.json").exists());
    let loaded = Model::load(&output).unwrap();
    assert_eq!(loaded.processor.encode("ab").unwrap().len(), 1);
    assert!(mingram::save(&trained, &output).is_err());
}

#[test]
#[ignore = "requires the uv-managed Python SentencePiece runtime; included in just check"]
fn python_runtime_compatibility() {
    use std::process::{Command, Stdio};
    let temp = TempDir::new().unwrap();
    let corpus = temp.path().join("train.txt");
    let prefix = temp.path().join("seed");
    fs::write(
        &corpus,
        "ab ab ab\nก้กา ก้กา\n中文 中文\nالعربية\nहिन्दी\n𠀀\n",
    )
    .unwrap();
    let output = Command::new("uv").args(["run", "--no-sync", "python", "-c",
        "import sentencepiece as sp,sys; sp.SentencePieceTrainer.train(input=sys.argv[1],model_prefix=sys.argv[2],model_type='bpe',vocab_size=64,hard_vocab_limit=False,character_coverage=1.0,normalization_rule_name='identity',add_dummy_prefix=False,remove_extra_whitespaces=False,minloglevel=2)"])
        .arg(&corpus).arg(&prefix).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let seed = Model::load(&prefix.with_extension("model")).unwrap();
    let required = seed
        .proto
        .pieces
        .iter()
        .filter(|p| p.r#type() != Type::Normal || p.piece().chars().count() == 1)
        .count();
    let target = (required + 4).min(seed.proto.pieces.len());
    let trained = mingram::train(
        &seed,
        &[corpus],
        &options(target),
        &CancellationToken::default(),
    )
    .unwrap();
    let model_path = temp.path().join("mingram.model");
    mingram::save(&trained, &model_path).unwrap();
    let samples = ["ab ก้กา 中文", "العربية", "हिन्दी", "  ab  ", "𠀀"];
    let counts: Vec<_> = samples
        .iter()
        .map(|text| trained.model.processor.encode(text).unwrap().len())
        .collect();
    let mut child = Command::new("uv")
        .args([
            "run",
            "--no-sync",
            "python",
            "-c",
            r#"
import json, sys
import sentencepiece as sp
data = json.load(sys.stdin)
model = sp.SentencePieceProcessor(model_file=sys.argv[1])
assert model.get_piece_size() == data['vocab_size']
for text, expected in zip(data['samples'], data['counts']):
    ids = model.encode(text, out_type=int)
    assert len(ids) == expected, (text, ids, expected)
    assert model.decode(ids) == text, (text, model.decode(ids))
"#,
        ])
        .arg(&model_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    #[derive(serde::Serialize)]
    struct Expected<'a> {
        samples: &'a [&'a str],
        counts: Vec<usize>,
        vocab_size: usize,
    }
    serde_json::to_writer(
        child.stdin.take().unwrap(),
        &Expected {
            samples: &samples,
            counts,
            vocab_size: target,
        },
    )
    .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn parallel_evaluation_matches_sequential_metrics_exactly() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("heldout.txt");
    fs::write(&path, "ab\nก้\n中文\n\n".repeat(records::BATCH_RECORDS + 1)).unwrap();
    let baseline = model(&["a", "b", "ก", "้", "中", "文"]);
    let candidate = model(&["ab", "ก้", "中文", "a", "ก", "中"]);
    let inputs = [path];
    let cancel = CancellationToken::default();
    let serial = evaluate::compare(&baseline, &candidate, &inputs, 1024, true, &cancel).unwrap();
    let parallel = evaluate::compare_with_options(
        &baseline,
        &candidate,
        &inputs,
        &evaluate::Options {
            max_record_bytes: 1024,
            threads: 4,
            progress: &crate::progress::ProgressReporter::silent(),
        },
        &cancel,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(serial).unwrap(),
        serde_json::to_value(parallel).unwrap()
    );
}

#[test]
fn batches_bound_records_and_bytes_and_preserve_delimiters() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("records.txt");
    fs::write(&path, "x\r\n".repeat(records::BATCH_RECORDS * 2 + 1)).unwrap();
    let mut sizes = Vec::new();
    records::batches(&path, 1024, &CancellationToken::default(), |batch| {
        assert!(batch.records.len() <= records::BATCH_RECORDS);
        assert!(batch.bytes <= (records::BATCH_BYTES + 1024 + 3) as u64);
        assert!(batch.records.iter().all(|text| text == "x"));
        sizes.push((batch.records.len(), batch.bytes, batch.first_line));
        Ok(())
    })
    .unwrap();
    assert_eq!(
        sizes,
        vec![(4096, 12288, 1), (4096, 12288, 4097), (1, 3, 8193)]
    );
    fs::write(&path, format!("{}\n", "a".repeat(1024 * 1024)).repeat(6)).unwrap();
    let mut bytes = Vec::new();
    records::batches(&path, 1024 * 1024, &CancellationToken::default(), |batch| {
        bytes.push(batch.bytes);
        Ok(())
    })
    .unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn parallel_counting_preserves_byte_fallback_and_rejects_unknowns() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("byte-records.txt");
    fs::write(&path, "a中\n".repeat(records::BATCH_RECORDS + 1)).unwrap();
    let seed = Model::from_bytes(&byte_proto().encode_to_vec()).unwrap();
    let inputs = [path];
    let progress = crate::progress::ProgressReporter::silent();
    let runtime = mingram::Runtime::new(4, &progress, None).unwrap();
    let cancel = CancellationToken::default();
    let serial =
        mingram::train(&seed, &inputs, &options(seed.proto.pieces.len()), &cancel).unwrap();
    let parallel = mingram::train_with_runtime(
        &seed,
        &inputs,
        &options(seed.proto.pieces.len()),
        &runtime,
        &cancel,
    )
    .unwrap();
    assert_eq!(serial.model.serialized(), parallel.model.serialized());
    assert_eq!(
        serde_json::to_value(serial.report).unwrap(),
        serde_json::to_value(parallel.report).unwrap()
    );
    let unknown = model(&["a"]);
    assert!(mingram::train_with_runtime(
        &unknown,
        &inputs,
        &options(unknown.proto.pieces.len()),
        &runtime,
        &cancel
    )
    .is_err());
}

#[test]
fn source_groups_sum_counts_before_computing_compression() {
    let temp = TempDir::new().unwrap();
    let first = temp
        .path()
        .join("shard-text-source_demo-len_short-part_1.txt");
    let second = temp
        .path()
        .join("shard-text-source_demo-len_long-part_2.txt");
    fs::write(&first, "ab\n").unwrap();
    fs::write(&second, "bbb\n").unwrap();
    let baseline = model(&["a", "b", "ba"]);
    let candidate = model(&["a", "b", "ab"]);
    let report = evaluate::compare(
        &baseline,
        &candidate,
        &[first, second],
        64,
        true,
        &CancellationToken::default(),
    )
    .unwrap();
    assert_eq!(report.sources.len(), 1);
    let source = &report.sources[0];
    assert_eq!(source.source_group, "demo");
    assert_eq!(source.metrics.counts.records, 2);
    assert_eq!(source.metrics.counts.text_codepoints, 5);
    assert_eq!(source.metrics.counts.candidate.tokens, 4);
    assert_eq!(
        source.metrics.counts.candidate.codepoints_per_token,
        Ratio::Defined { value: 1.25 }
    );
}
