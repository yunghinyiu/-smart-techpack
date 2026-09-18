//! Core data model for smart-techpack.
//!
//! A pack is a versioned JSON document (see `docs/spec-v0.md`). Every struct
//! denies unknown fields so schema drift fails loudly instead of silently.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Version marker for the spec format. Bump `CURRENT` whenever the shape
/// changes and document the migration in `docs/spec-v*.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecVersion {
    pub version: u32,
}

impl SpecVersion {
    pub const CURRENT: u32 = 1;

    pub fn current() -> Self {
        Self {
            version: Self::CURRENT,
        }
    }
}

/// What is being specced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Garment {
    /// Controlled vocabulary, e.g. "jacket". Lowercase, singular.
    pub garment_type: String,
    /// Human name, e.g. "Cropped Military Jacket".
    pub name: String,
}

/// Reference to the source photo on disk, relative to the project folder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhotoRef {
    pub path: String,
}

/// One fabric color matched to a Pantone TCX code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pantone {
    pub code: String,
    pub name: String,
    /// Sampled color from the photo, `#rrggbb`.
    pub hex: String,
}

/// A numbered marker on the garment image. Coordinates are normalized 0.0-1.0
/// relative to the source photo so they survive resizes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Callout {
    pub id: String,
    pub label: String,
    pub x: f32,
    pub y: f32,
    pub detail_id: Option<String>,
}

/// Construction spec for one callout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetailSpec {
    pub id: String,
    pub callout_id: String,
    pub seam_type: String,
    pub allowance_mm: f32,
    pub topstitch: String,
    pub notes: String,
}

/// One points-of-measure row across sizes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PomRow {
    pub code: String,
    pub description: String,
    pub tolerance_mm: f32,
    pub sizes: BTreeMap<String, f32>,
}

/// The whole pack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Project {
    pub spec_version: SpecVersion,
    pub garment: Garment,
    pub photo: Option<PhotoRef>,
    pub pantones: Vec<Pantone>,
    pub callouts: Vec<Callout>,
    pub details: Vec<DetailSpec>,
    pub pom: Vec<PomRow>,
}

/// Parse a pack from JSON. Unknown fields are an error by construction.
pub fn load_project(json: &str) -> Result<Project, serde_json::Error> {
    serde_json::from_str(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_project() -> Project {
        Project {
            spec_version: SpecVersion::current(),
            garment: Garment {
                garment_type: "jacket".to_string(),
                name: "Test Jacket".to_string(),
            },
            photo: Some(PhotoRef {
                path: "photo.jpg".to_string(),
            }),
            pantones: vec![Pantone {
                code: "19-0515 TCX".to_string(),
                name: "Forest Night".to_string(),
                hex: "#2f3b2f".to_string(),
            }],
            callouts: vec![Callout {
                id: "c01".to_string(),
                label: "Point collar".to_string(),
                x: 0.5,
                y: 0.1,
                detail_id: Some("d01".to_string()),
            }],
            details: vec![DetailSpec {
                id: "d01".to_string(),
                callout_id: "c01".to_string(),
                seam_type: "Overlock".to_string(),
                allowance_mm: 12.7,
                topstitch: "None".to_string(),
                notes: "Snug fit.".to_string(),
            }],
            pom: vec![PomRow {
                code: "chest".to_string(),
                description: "Chest width".to_string(),
                tolerance_mm: 6.0,
                sizes: BTreeMap::from([("S".to_string(), 55.0), ("M".to_string(), 57.5)]),
            }],
        }
    }

    #[test]
    fn round_trip_is_byte_stable() {
        let project = sample_project();
        let first = serde_json::to_string_pretty(&project).expect("serialize");
        let parsed: Project = load_project(&first).expect("parse");
        assert_eq!(parsed, project);
        let second = serde_json::to_string_pretty(&parsed).expect("re-serialize");
        assert_eq!(first, second);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let json = r#"{
            "spec_version": {"version": 1},
            "garment": {"garment_type": "jacket", "name": "X", "surprise": true},
            "photo": null,
            "pantones": [],
            "callouts": [],
            "details": [],
            "pom": []
        }"#;
        let err = load_project(json).expect_err("unknown field must fail");
        assert!(err.to_string().contains("surprise"), "got: {err}");
    }
}
