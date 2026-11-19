//! Standard SentencePiece model parsing and native-runtime access.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use prost::Message;
use sentencepiece::SentencePieceProcessor;
use sentencepiece_model::{ModelProto, ModelType, SentencePiece, TrainerSpec, Type};

use super::wire;

pub struct Model {
    pub proto: ModelProto,
    pub processor: SentencePieceProcessor,
    bytes: Vec<u8>,
    normalization_hash: blake3::Hash,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("loading {}", path.display()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let proto = ModelProto::decode(bytes).context("decoding SentencePiece protobuf")?;
        let mut names = HashSet::new();
        for piece in &proto.pieces {
            ensure!(!piece.piece().is_empty(), "empty vocabulary piece");
            ensure!(names.insert(piece.piece()), "duplicate vocabulary piece");
            ensure!(piece.score().is_finite(), "non-finite piece score");
            Type::try_from(piece.r#type.unwrap_or(Type::Normal as i32))
                .context("unrecognized piece type")?;
        }
        let processor = SentencePieceProcessor::from_serialized_proto(bytes)?;
        let mut normalization = blake3::Hasher::new();
        for field in wire::fields(bytes)?
            .into_iter()
            .filter(|f| matches!(f.tag, 3 | 5))
        {
            normalization.update(field.raw);
        }
        Ok(Self {
            proto,
            processor,
            bytes: bytes.to_vec(),
            normalization_hash: normalization.finalize(),
        })
    }

    /// Rebuilds a Unigram model, retaining original piece extensions and unrelated metadata.
    /// Old self-tests are removed because their expected segmentations are no longer valid.
    pub fn with_pieces(&self, pieces: &[(usize, SentencePiece)]) -> Result<Self> {
        let fields = wire::fields(&self.bytes)?;
        let originals: Vec<_> = fields.iter().filter(|f| f.tag == 1).collect();
        let mut encoded_pieces = Vec::new();
        for (old_id, piece) in pieces {
            let original = self
                .proto
                .pieces
                .get(*old_id)
                .context("piece ID outside seed vocabulary")?;
            ensure!(
                original.piece() == piece.piece() && original.r#type() == piece.r#type(),
                "model rewriting may change scores and ordering, not piece identities or types"
            );
            let mut score = Vec::new();
            prost::encoding::encode_key(2, prost::encoding::WireType::ThirtyTwoBit, &mut score);
            score.extend_from_slice(&piece.score().to_le_bytes());
            let payload = wire::patch(originals[*old_id].payload, &[(2, &score)])?;
            encoded_pieces.extend(wire::message(1, &payload));
        }
        let specs: Vec<_> = fields.iter().filter(|f| f.tag == 2).collect();
        ensure!(
            specs.len() <= 1,
            "multiple trainer specifications cannot be edited safely"
        );
        let original_spec = specs.first().map_or(&[][..], |field| field.payload);
        let spec = TrainerSpec {
            model_type: Some(ModelType::Unigram as i32),
            vocab_size: Some(i32::try_from(pieces.len())?),
            ..Default::default()
        }
        .encode_to_vec();
        let spec_fields = wire::fields(&spec)?;
        let replacements: Vec<_> = spec_fields.iter().map(|f| (f.tag, f.raw)).collect();
        let trainer = wire::message(2, &wire::patch(original_spec, &replacements)?);
        let bytes = wire::patch(
            &self.bytes,
            &[(1, &encoded_pieces), (2, &trainer), (4, &[])],
        )?;
        Self::from_bytes(&bytes)
    }

    pub fn serialized(&self) -> &[u8] {
        &self.bytes
    }

    pub fn fingerprint(&self) -> String {
        blake3::hash(&self.bytes).to_hex().to_string()
    }

    pub fn same_normalization(&self, other: &Self) -> bool {
        self.normalization_hash == other.normalization_hash
            && self.whitespace_suffix() == other.whitespace_suffix()
            && self.user_symbols() == other.user_symbols()
    }

    fn whitespace_suffix(&self) -> bool {
        self.proto
            .trainer_spec
            .as_ref()
            .is_some_and(|spec| spec.treat_whitespace_as_suffix())
    }

    fn user_symbols(&self) -> HashSet<&str> {
        self.proto
            .pieces
            .iter()
            .filter(|p| p.r#type() == Type::UserDefined)
            .map(|p| p.piece())
            .collect()
    }
}
