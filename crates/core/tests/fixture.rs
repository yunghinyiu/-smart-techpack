//! The checked-in jacket fixture must always parse against the current schema.

use std::path::PathBuf;
use techpack_core::load_project;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("jacket.json")
}

#[test]
fn jacket_fixture_parses() {
    let json = std::fs::read_to_string(fixture_path()).expect("read fixture");
    let project = load_project(&json).expect("parse fixture");
    assert_eq!(project.garment.name, "Cropped Military Jacket");
    assert_eq!(project.pantones.len(), 3);
    assert_eq!(project.callouts.len(), 15);
    assert!(!project.details.is_empty());
    assert!(!project.pom.is_empty());
}
