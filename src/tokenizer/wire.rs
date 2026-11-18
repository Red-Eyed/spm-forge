//! Rewrites selected protobuf fields while preserving unknown model extensions.

use anyhow::{ensure, Result};
use prost::encoding::{decode_key, decode_varint, skip_field, DecodeContext, WireType};

pub struct Field<'a> {
    pub tag: u32,
    pub raw: &'a [u8],
    pub payload: &'a [u8],
}

pub fn fields(mut bytes: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut result = Vec::new();
    while !bytes.is_empty() {
        let start = bytes;
        let (tag, wire) = decode_key(&mut bytes)?;
        let mut value = bytes;
        let payload = if wire == WireType::LengthDelimited {
            let length = usize::try_from(decode_varint(&mut value)?)?;
            ensure!(length <= value.len(), "truncated protobuf field");
            &value[..length]
        } else {
            &[]
        };
        skip_field(wire, tag, &mut bytes, DecodeContext::default())?;
        result.push(Field {
            tag,
            raw: &start[..start.len() - bytes.len()],
            payload,
        });
    }
    Ok(result)
}

/// Replacements are already wire-encoded, including keys. An empty value deletes a field.
pub fn patch(bytes: &[u8], replacements: &[(u32, &[u8])]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    for field in fields(bytes)? {
        if !replacements.iter().any(|(tag, _)| *tag == field.tag) {
            output.extend_from_slice(field.raw);
        }
    }
    for (_, replacement) in replacements {
        output.extend_from_slice(replacement);
    }
    Ok(output)
}

pub fn message(tag: u32, payload: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    prost::encoding::encode_key(tag, WireType::LengthDelimited, &mut output);
    prost::encoding::encode_varint(payload.len() as u64, &mut output);
    output.extend_from_slice(payload);
    output
}
