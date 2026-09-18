//! Photo upload validation and ingest pipeline.
//!
//! Every upload funnels through [`ingest`]: byte cap, JPEG/PNG magic check,
//! decode, EXIF orientation normalization, megapixel cap. Callers get a
//! normalized [`IngestedPhoto`] plus helpers to build masked cutouts.

use std::io::Cursor;

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use thiserror::Error;

/// Hard cap on request bytes, checked before any decoding work.
pub const MAX_BYTES: usize = 15 * 1024 * 1024;
/// Hard cap on decoded pixels (width x height), checked after decode.
pub const MAX_PIXELS: u64 = 12_000_000;
/// Long edge used for stored preview composites.
pub const PREVIEW_LONG_EDGE: u32 = 1024;

/// Typed upload failures. Maps 1:1 onto HTTP statuses in the web crate:
/// size violations are 413, type/decode problems are 400.
#[derive(Debug, Error, PartialEq)]
pub enum IngestError {
    /// Raw upload is bigger than [`MAX_BYTES`].
    #[error("upload is {0} bytes, limit is {MAX_BYTES} bytes")]
    TooLargeBytes(usize),
    /// Magic bytes are not JPEG or PNG (or there is no image at all).
    #[error("unsupported image type: upload must be JPEG or PNG")]
    UnsupportedType,
    /// Decoded photo is bigger than [`MAX_PIXELS`] pixels.
    #[error("photo is {0:.1} MP, over the 12.0 MP limit")]
    TooManyPixels(f64),
    /// A recognized JPEG/PNG that still fails to decode.
    #[error("could not decode image: {0}")]
    Decode(String),
}

/// Accepted upload encoding, sniffed from magic bytes (never the client
/// `Content-Type`, which is trivially spoofed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotoFormat {
    Jpeg,
    Png,
}

impl PhotoFormat {
    /// Extension for the stored original file.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
        }
    }

    /// Content-Type for serving the stored original back.
    pub fn mime(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
        }
    }
}

/// A validated, orientation-normalized photo ready for inference.
#[derive(Debug)]
pub struct IngestedPhoto {
    /// Pixels with EXIF orientation already applied, at source resolution.
    pub image: DynamicImage,
    pub format: PhotoFormat,
}

/// Validate and normalize one upload. EXIF orientation is applied during
/// ingest (via the decoder's orientation flag) so every downstream consumer
/// — mask coordinates, color sampling, previews — sees upright pixels.
pub fn ingest(bytes: &[u8]) -> Result<IngestedPhoto, IngestError> {
    if bytes.len() > MAX_BYTES {
        return Err(IngestError::TooLargeBytes(bytes.len()));
    }
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| IngestError::Decode(e.to_string()))?;
    let format = match reader.format() {
        Some(ImageFormat::Jpeg) => PhotoFormat::Jpeg,
        Some(ImageFormat::Png) => PhotoFormat::Png,
        _ => return Err(IngestError::UnsupportedType),
    };
    let mut decoder = reader.into_decoder().map_err(decode_err)?;
    let orientation = decoder.orientation().map_err(decode_err)?;
    let mut image = DynamicImage::from_decoder(decoder).map_err(decode_err)?;
    image.apply_orientation(orientation);
    let megapixels = f64::from(image.width()) * f64::from(image.height()) / 1e6;
    if u64::from(image.width()) * u64::from(image.height()) > MAX_PIXELS {
        return Err(IngestError::TooManyPixels(megapixels));
    }
    Ok(IngestedPhoto { image, format })
}

fn decode_err(e: image::ImageError) -> IngestError {
    IngestError::Decode(e.to_string())
}

/// Downscale so the long edge is at most `edge` px (aspect preserved).
/// Smaller images pass through untouched.
pub fn downscale_long_edge(img: &DynamicImage, edge: u32) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    let long = w.max(h);
    if long <= edge {
        return img.clone();
    }
    let scale = f64::from(edge) / f64::from(long);
    let nw = ((f64::from(w) * scale).round() as u32).max(1);
    let nh = ((f64::from(h) * scale).round() as u32).max(1);
    img.resize_exact(nw, nh, image::imageops::FilterType::Triangle)
}

/// Composite the photo over transparency using `mask`: foreground stays
/// opaque, background becomes transparent. Panics if the mask resolution
/// differs from the image — segmenters must return source-resolution masks.
pub fn cutout_rgba(img: &DynamicImage, mask: &crate::Mask) -> image::RgbaImage {
    assert_eq!(
        (mask.width, mask.height),
        (img.width(), img.height()),
        "mask resolution must match the image"
    );
    let mut rgba = img.to_rgba8();
    for (px, fg) in rgba.pixels_mut().zip(mask.foreground.iter()) {
        if !fg {
            px[3] = 0;
        }
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::png::PngEncoder;
    use image::ExtendedColorType;
    use image::ImageEncoder;

    fn encode_png(w: u32, h: u32, rgb: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        PngEncoder::new(&mut buf)
            .write_image(rgb, w, h, ExtendedColorType::Rgb8)
            .expect("test png must encode");
        buf
    }

    #[test]
    fn garbage_bytes_are_unsupported() {
        let err = ingest(b"this is not an image").expect_err("garbage must fail");
        assert_eq!(err, IngestError::UnsupportedType);
    }

    #[test]
    fn empty_upload_is_unsupported() {
        let err = ingest(&[]).expect_err("empty upload must fail");
        assert_eq!(err, IngestError::UnsupportedType);
    }

    #[test]
    fn gif_magic_is_unsupported() {
        let err = ingest(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;").expect_err("gif must fail");
        assert_eq!(err, IngestError::UnsupportedType);
    }

    #[test]
    fn oversized_bytes_rejected_before_decode() {
        let big = vec![0u8; MAX_BYTES + 1];
        let err = ingest(&big).expect_err("oversize must fail");
        assert_eq!(err, IngestError::TooLargeBytes(big.len()));
    }

    #[test]
    fn too_many_pixels_rejected_after_decode() {
        // 3500x3500 solid PNG: ~12.25 MP, encodes to kilobytes.
        let png = encode_png(3500, 3500, &vec![200u8; 3500 * 3500 * 3]);
        assert!(png.len() < MAX_BYTES, "test file must fit the byte cap");
        let err = ingest(&png).expect_err("12.25 MP must fail");
        assert!(
            matches!(err, IngestError::TooManyPixels(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn png_roundtrip_keeps_pixels_and_format() {
        let png = encode_png(32, 24, &vec![180u8; 32 * 24 * 3]);
        let photo = ingest(&png).expect("valid png must ingest");
        assert_eq!(photo.format, PhotoFormat::Png);
        assert_eq!((photo.image.width(), photo.image.height()), (32, 24));
    }

    #[test]
    fn jpeg_magic_is_accepted() {
        let mut buf = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85)
            .encode_image(&DynamicImage::new_rgb8(16, 12))
            .expect("test jpeg must encode");
        let photo = ingest(&buf).expect("valid jpeg must ingest");
        assert_eq!(photo.format, PhotoFormat::Jpeg);
        assert_eq!((photo.image.width(), photo.image.height()), (16, 12));
    }

    #[test]
    fn downscale_only_shrinks_the_long_edge() {
        let big = DynamicImage::new_rgb8(2000, 1000);
        let small = downscale_long_edge(&big, 1024);
        assert_eq!((small.width(), small.height()), (1024, 512));
        let tiny = DynamicImage::new_rgb8(100, 80);
        let same = downscale_long_edge(&tiny, 1024);
        assert_eq!((same.width(), same.height()), (100, 80));
    }
}
