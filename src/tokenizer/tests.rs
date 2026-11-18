//! Small, test-owned multilingual model fixtures; no external data dependencies.

use std::fs;

use prost::Message;
use sentencepiece_model::{
    ModelProto, ModelType, NormalizerSpec, SentencePiece, TrainerSpec, Type,
};
use tempfile::TempDir;

use super::{
    evaluate::{self, Ratio},
    model::Model,
    records,
};
use crate::cancel::CancellationToken;

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
