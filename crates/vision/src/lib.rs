//! Garment isolation behind a swappable [`Segmenter`] trait.
//!
//! The trait returns a [`Mask`] at the source photo's resolution so callers
//! (cutout preview, color sampling, callout coordinates) never depend on a
//! specific model. Preprocessing shared by the U2Net-family candidates lives
//! in [`preprocess_u2net`].

use image::DynamicImage;
use thiserror::Error;

pub mod cloth_seg;
pub mod ingest;

pub use cloth_seg::ClothSeg;
pub use ingest::{
    cutout_rgba, downscale_long_edge, ingest, IngestError, IngestedPhoto, PhotoFormat,
};

/// Failures that can occur while producing a mask.
#[derive(Debug, Error)]
pub enum Error {
    /// The inference backend (ONNX Runtime) or model I/O failed.
    #[error("inference backend failed: {0}")]
    Backend(String),
    /// The model returned a tensor with an unexpected shape.
    #[error("model output has unexpected shape: {0}")]
    Shape(String),
}

/// Binary foreground mask at the source photo's resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    /// Row-major foreground flags, exactly `width * height` entries.
    pub foreground: Vec<bool>,
}

impl Mask {
    /// Fraction of pixels flagged foreground, 0.0-1.0. Useful as a sanity
    /// signal: values near 0 or 1 usually mean the model missed.
    pub fn foreground_fraction(&self) -> f64 {
        if self.foreground.is_empty() {
            return 0.0;
        }
        self.foreground.iter().filter(|b| **b).count() as f64 / self.foreground.len() as f64
    }

    /// Render the mask as a grayscale image (0 background, 255 foreground)
    /// for PNG dumps and debug composites.
    pub fn to_luma(&self) -> image::GrayImage {
        let buf: Vec<u8> = self
            .foreground
            .iter()
            .map(|b| u8::from(*b).saturating_mul(255))
            .collect();
        image::GrayImage::from_raw(self.width, self.height, buf)
            .expect("mask buffer always matches dimensions")
    }

    /// Nearest-neighbor resample to `width` x `height`. Used to map a
    /// model-resolution mask back onto the source photo (or a preview).
    pub fn upscale_to(&self, width: u32, height: u32) -> Mask {
        assert!(
            !self.foreground.is_empty() && self.width > 0 && self.height > 0,
            "cannot resample an empty mask"
        );
        let mut foreground = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            let sy = ((u64::from(y) * u64::from(self.height)) / u64::from(height.max(1)))
                .min(u64::from(self.height - 1)) as u32;
            for x in 0..width {
                let sx = ((u64::from(x) * u64::from(self.width)) / u64::from(width.max(1)))
                    .min(u64::from(self.width - 1)) as u32;
                foreground.push(self.foreground[(sy * self.width + sx) as usize]);
            }
        }
        Mask {
            width,
            height,
            foreground,
        }
    }
}

/// Produces a garment mask for a photo. Implementations own their session,
/// so they must be `Send + Sync` to allow web handlers to share them.
pub trait Segmenter: Send + Sync {
    /// Isolate the garment in `img`, returning a mask at `img`'s resolution.
    fn mask(&self, img: &DynamicImage) -> Result<Mask, Error>;
}

/// U2Net-style preprocessing shared by both Slice 2 spike candidates:
/// resize to `size` x `size`, normalize with ImageNet mean/std, emit NCHW
/// `f32` in row-major order.
pub fn preprocess_u2net(img: &DynamicImage, size: u32) -> Vec<f32> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let rgb = img
        .resize_exact(size, size, image::imageops::FilterType::Triangle)
        .to_rgb8();
    let pixels = size as usize * size as usize;
    let mut out = vec![0.0f32; 3 * pixels];
    for (i, px) in rgb.pixels().enumerate() {
        for c in 0..3 {
            out[c * pixels + i] = (f32::from(px[c]) / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}
