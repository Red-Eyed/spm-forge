//! Standard SentencePiece model parsing and native-runtime access.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use prost::Message;
use sentencepiece::SentencePieceProcessor;
use sentencepiece_model::{ModelProto, Type};

pub struct Model {
    pub proto: ModelProto,
    pub processor: SentencePieceProcessor,
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
        Ok(Self { proto, processor })
    }

    pub fn same_normalization(&self, other: &Self) -> bool {
        self.proto.normalizer_spec == other.proto.normalizer_spec
            && self.proto.denormalizer_spec == other.proto.denormalizer_spec
            && self.proto.trainer_spec.as_ref().is_some_and(|spec| {
                other.proto.trainer_spec.as_ref().is_some_and(|other| {
                    spec.treat_whitespace_as_suffix() == other.treat_whitespace_as_suffix()
                })
            })
            && self.user_symbols() == other.user_symbols()
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
