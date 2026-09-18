# Spec v0 (spec_version.version = 1)

A pack is one JSON document. All measurements are millimetres. Callout
coordinates are normalized 0.0-1.0 relative to the source photo so markers
survive resizes. Every struct denies unknown fields: typos fail at parse
time instead of silently dropping data.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `spec_version` | `{ version: u32 }` | Format version. Currently 1. |
| `garment` | object | What is being specced. |
| `photo` | object or null | Source photo reference. |
| `pantones` | array | Fabric colors. |
| `callouts` | array | Numbered markers on the photo. |
| `details` | array | Construction spec per callout. |
| `pom` | array | Points-of-measure rows across sizes. |

## garment

| Field | Type | Meaning |
|---|---|---|
| `garment_type` | string | Controlled vocabulary, lowercase singular, e.g. `jacket`. |
| `name` | string | Human name, e.g. `Cropped Military Jacket`. |

## photo

| Field | Type | Meaning |
|---|---|---|
| `path` | string | Path relative to the project folder. |

## pantones[]

| Field | Type | Meaning |
|---|---|---|
| `code` | string | Pantone TCX code, e.g. `19-0515 TCX`. |
| `name` | string | Pantone name, e.g. `Forest Night`. |
| `hex` | string | Sampled photo color as `#rrggbb`. |

## callouts[]

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable id, e.g. `c01`. |
| `label` | string | Detail name from the controlled vocabulary. |
| `x`, `y` | float | Normalized photo coordinates. |
| `detail_id` | string or null | Link into `details[]` once specced. |

## details[]

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable id, e.g. `d01`. |
| `callout_id` | string | The callout this spec belongs to. |
| `seam_type` | string | e.g. `Overlock`, `Single needle`. |
| `allowance_mm` | float | Seam allowance in mm. |
| `topstitch` | string | e.g. `None`, `6mm double needle`. |
| `notes` | string | Free-text design intent. |

## pom[]

| Field | Type | Meaning |
|---|---|---|
| `code` | string | Short code, e.g. `chest`. |
| `description` | string | How the point is measured. |
| `tolerance_mm` | float | Acceptable factory tolerance. |
| `sizes` | map | Size label to value in mm. |

## Migration policy

- The version field is required from day one and never removed.
- Additive changes (new optional field, new garment template) bump nothing;
  old documents stay valid.
- Breaking changes (rename, remove, retype) bump `CURRENT`, ship a new
  `docs/spec-vN.md`, and add a migration function plus test in core.
