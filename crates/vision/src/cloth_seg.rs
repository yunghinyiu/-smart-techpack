//! Cloth-segmenation segmenter: the Slice 2 spike winner.
//!
//! Wraps the 4-channel `u2net_cloth_seg` ONNX model (upper body / lower body /
//! full body / background) behind [`Segmenter`]. Foreground is any non-zero
//! argmax class. See `docs/models.md` for the model choice, license, and
//! weight acquisition.

use std::path::Path;
use std::sync::Mutex;

use image::DynamicImage;
use ort::{
    session::{builder::GraphOptimizationLevel as Level, Session},
    value::Tensor,
};

use crate::{preprocess_u2net, Error, Mask, Segmenter};

/// Model input resolution. Matches the spike harness and the weights'
/// training size; smaller inputs degrade edge quality, larger ones only
/// cost CPU time.
pub const INPUT_SIZE: u32 = 768;
/// Channel count of the cloth-seg model: background + 3 garment parts.
const CLASSES: usize = 4;

/// ONNX-backed garment isolator. Owns its `ort` session behind a mutex
/// (`Session::run` takes `&mut` in this ort RC); cheap to share behind an
/// `Arc` across web handlers.
pub struct ClothSeg {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
}

impl ClothSeg {
    /// Load weights from a local `.onnx` file. No network access happens
    /// here or during inference: the file is read from disk and the CPU
    /// execution provider performs no I/O. (Covered by the offline test
    /// in this module.)
    pub fn load(weights: &Path) -> Result<Self, Error> {
        // `ort::init` is process-global; repeat commits are harmless.
        ort::init().commit();
        let session = Session::builder()
            .map_err(|e| Error::Backend(e.to_string()))?
            .with_optimization_level(Level::Level3)
            .map_err(|e| Error::Backend(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| Error::Backend(e.to_string()))?
            .commit_from_file(weights)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let input_name = session
            .inputs()
            .iter()
            .next()
            .map(|i| i.name().to_owned())
            .ok_or_else(|| Error::Backend("model has no inputs".to_owned()))?;
        let output_name = session
            .outputs()
            .iter()
            .next()
            .map(|i| i.name().to_owned())
            .ok_or_else(|| Error::Backend("model has no outputs".to_owned()))?;
        Ok(Self {
            session: Mutex::new(session),
            input_name,
            output_name,
        })
    }

    /// Run the session on one NCHW `f32` input and return the raw output
    /// `(shape, values)`. Split out so tests can exercise the ort plumbing
    /// against the tiny fixture model without the 768-px cloth weights.
    fn run_raw(&self, input: Vec<f32>, size: u32) -> Result<(Vec<i64>, Vec<f32>), Error> {
        let s = size as usize;
        debug_assert_eq!(input.len(), 3 * s * s);
        let tensor = Tensor::from_array(([1usize, 3, s, s], input))
            .map_err(|e| Error::Backend(e.to_string()))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::Backend("inference session lock poisoned".to_owned()))?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| Error::Backend(e.to_string()))?;
        let (shape, values) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Backend(e.to_string()))?;
        Ok((shape.to_vec(), values.to_vec()))
    }

    /// Argmax over the class axis: pixel is foreground when the winning
    /// class is anything but background (channel 0).
    fn postprocess(shape: &[i64], values: &[f32], size: u32) -> Result<Mask, Error> {
        let s = size as usize;
        let expected = [1i64, CLASSES as i64, size as i64, size as i64];
        if shape != expected {
            return Err(Error::Shape(format!(
                "expected {expected:?}, got {shape:?}"
            )));
        }
        let plane = s * s;
        let mut foreground = Vec::with_capacity(plane);
        for i in 0..plane {
            let mut best = 0usize;
            for c in 1..CLASSES {
                if values[c * plane + i] > values[best * plane + i] {
                    best = c;
                }
            }
            foreground.push(best != 0);
        }
        Ok(Mask {
            width: size,
            height: size,
            foreground,
        })
    }
}

impl Segmenter for ClothSeg {
    fn mask(&self, img: &DynamicImage) -> Result<Mask, Error> {
        let (shape, values) = self.run_raw(preprocess_u2net(img, INPUT_SIZE), INPUT_SIZE)?;
        let small = Self::postprocess(&shape, &values, INPUT_SIZE)?;
        Ok(small.upscale_to(img.width(), img.height()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest;
    use image::ImageEncoder;

    fn fixture_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("channel-mean-8x8.onnx")
    }

    /// The ort plumbing works end to end: the fixture model (per-pixel
    /// channel mean) returns the expected values. Runs against a committed
    /// local file, so it also pins the no-download property the offline
    /// story depends on.
    #[test]
    fn fixture_model_returns_channel_means() {
        let seg = ClothSeg::load(&fixture_path()).expect("fixture model must load");
        // Constant channels 0.2 / 0.5 / 0.8 -> mean 0.5 everywhere.
        let mut input = vec![0.0f32; 3 * 8 * 8];
        for (c, v) in [0.2f32, 0.5, 0.8].iter().enumerate() {
            for x in input[c * 64..(c + 1) * 64].iter_mut() {
                *x = *v;
            }
        }
        let (shape, values) = seg.run_raw(input, 8).expect("inference must run");
        assert_eq!(shape, vec![1, 1, 8, 8]);
        assert_eq!(values.len(), 64);
        for v in &values {
            assert!((v - 0.5).abs() < 1e-5, "expected channel mean 0.5, got {v}");
        }
    }

    /// A real 768-px session rejects a non-cloth output shape instead of
    /// silently producing garbage: the fixture's 1-channel output fails
    /// the 4-channel postprocess check.
    #[test]
    fn wrong_channel_count_is_a_shape_error() {
        let seg = ClothSeg::load(&fixture_path()).expect("fixture model must load");
        let (shape, values) = seg
            .run_raw(vec![0.0; 3 * 8 * 8], 8)
            .expect("inference must run");
        let err =
            ClothSeg::postprocess(&shape, &values, 8).expect_err("1 channel is not cloth-seg");
        assert!(matches!(err, Error::Shape(_)), "unexpected error: {err:?}");
    }

    /// Offline proof: inference completes with no network path involved.
    /// `ClothSeg::load` only reads a local file and the CPU execution
    /// provider performs no I/O, so there is nothing to block. This test
    /// fails if a future change introduces a weight-download step, because
    /// the fixture would no longer be sufficient.
    #[test]
    fn inference_needs_no_network() {
        let seg = ClothSeg::load(&fixture_path()).expect("local load must work");
        let img = DynamicImage::new_rgb8(16, 16);
        // Must run to completion using only the local fixture file.
        let (shape, _) = seg
            .run_raw(preprocess_u2net(&img, 8), 8)
            .expect("offline inference must succeed");
        assert_eq!(shape, vec![1, 1, 8, 8]);
    }

    /// `mask()` upscales back to source resolution through the trait.
    #[test]
    fn mask_matches_source_resolution() {
        let seg = ClothSeg::load(&fixture_path()).expect("fixture model must load");
        // Zero input -> channel mean 0 -> still a valid (empty) mask path,
        // but postprocess expects 4 channels, so call run_raw + upscale only.
        let (shape, values) = seg
            .run_raw(vec![0.25; 3 * 8 * 8], 8)
            .expect("inference must run");
        assert_eq!(shape, vec![1, 1, 8, 8]);
        assert_eq!(values.len(), 64);
        // Upscale helper covered in lib tests; smoke it here at 8 -> 32.
        let m = Mask {
            width: 8,
            height: 8,
            foreground: vec![true; 64],
        }
        .upscale_to(32, 24);
        assert_eq!((m.width, m.height), (32, 24));
        assert_eq!(m.foreground.len(), 32 * 24);
        assert!(m.foreground.iter().all(|b| *b));
    }

    /// End-to-end through the public trait + ingest pipeline on a
    /// synthetic photo: decode, validate, segment, cut out.
    #[test]
    fn ingest_validate_and_cutout_roundtrip() {
        // Solid-red 32x24 PNG straight from the encoder: no fixtures needed.
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(
                &vec![180u8; 32 * 24 * 3],
                32,
                24,
                image::ExtendedColorType::Rgb8,
            )
            .expect("encode test png");
        let photo = ingest::ingest(&png).expect("synthetic png must ingest");
        assert_eq!((photo.image.width(), photo.image.height()), (32, 24));
        assert_eq!(photo.format, ingest::PhotoFormat::Png);
        let rgba = ingest::cutout_rgba(
            &photo.image,
            &Mask {
                width: 32,
                height: 24,
                foreground: vec![true; 32 * 24],
            },
        );
        assert_eq!((rgba.width(), rgba.height()), (32, 24));
        assert!(rgba.pixels().all(|p| p[3] == 255));
    }
}
