//! Slice 2 model spike: compare two U2Net-family ONNX segmenters via `ort`.
//!
//! Usage (from workspace root):
//! `cargo run -p techpack-vision --example segment_spike --
//! data/spike/weights data/spike data/spike/out`
//!
//! For each candidate model and each photo it prints preprocess / inference /
//! postprocess milliseconds plus the foreground fraction, and writes a mask
//! PNG and an RGBA cutout PNG per photo. Weights are local files and are
//! never committed (see `docs/models.md`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use image::DynamicImage;
use ort::session::{builder::GraphOptimizationLevel, Session};
use techpack_vision::{preprocess_u2net, Mask};

struct Candidate {
    name: &'static str,
    weights: &'static str,
    input_size: u32,
}

const CANDIDATES: [Candidate; 2] = [
    Candidate {
        name: "u2net-salient",
        weights: "u2net.onnx",
        input_size: 320,
    },
    Candidate {
        name: "u2net-cloth-seg",
        weights: "cloth-seg-u2net.onnx",
        input_size: 768,
    },
];

fn load_session(weights: &Path) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(weights)
}

fn run_candidate(
    session: &mut Session,
    input_name: &str,
    output_name: &str,
    img: &DynamicImage,
    input_size: u32,
) -> ort::Result<(Mask, u128, u128, u128)> {
    let t0 = Instant::now();
    let data = preprocess_u2net(img, input_size);
    let s = input_size as usize;
    let tensor = ort::value::Tensor::from_array(([1usize, 3, s, s], data))?;
    let preprocess_ms = t0.elapsed().as_millis();

    let t1 = Instant::now();
    let outputs = session.run(ort::inputs![input_name => tensor])?;
    let inference_ms = t1.elapsed().as_millis();

    let t2 = Instant::now();
    let (_shape, values) = outputs[output_name].try_extract_tensor::<f32>()?;
    let mask_small = postprocess(values, s);
    let mask = upscale(&mask_small, img.width(), img.height());
    let postprocess_ms = t2.elapsed().as_millis();

    Ok((mask, preprocess_ms, inference_ms, postprocess_ms))
}

/// Turn a raw NCHW model output at `size` x `size` into a small mask:
/// single-channel outputs are sigmoid scores thresholded at 0.5;
/// 4-channel outputs are argmaxed with class 0 as background.
fn postprocess(values: &[f32], size: usize) -> Mask {
    let pixels = size * size;
    let channels = values.len() / pixels;
    let foreground: Vec<bool> = match channels {
        1 => values.iter().map(|v| *v > 0.5).collect(),
        4 => (0..pixels)
            .map(|i| {
                let mut best = 0;
                for c in 1..4 {
                    if values[c * pixels + i] > values[best * pixels + i] {
                        best = c;
                    }
                }
                best != 0
            })
            .collect(),
        other => panic!("unexpected channel count {other}"),
    };
    Mask {
        width: size as u32,
        height: size as u32,
        foreground,
    }
}

fn upscale(small: &Mask, width: u32, height: u32) -> Mask {
    let luma = small.to_luma();
    let big = image::imageops::resize(&luma, width, height, image::imageops::FilterType::Triangle);
    let foreground = big.pixels().map(|p| p[0] > 127).collect();
    Mask {
        width,
        height,
        foreground,
    }
}

fn write_cutout(img: &DynamicImage, mask: &Mask, path: &Path) -> image::ImageResult<()> {
    let rgb = img.to_rgb8();
    let mut out = image::RgbaImage::new(mask.width, mask.height);
    for (x, y, px) in out.enumerate_pixels_mut() {
        let src = rgb.get_pixel(x, y);
        let alpha = if mask.foreground[(y * mask.width + x) as usize] {
            255
        } else {
            0
        };
        *px = image::Rgba([src[0], src[1], src[2], alpha]);
    }
    out.save(path)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let weights_dir = args
        .get(1)
        .map_or(PathBuf::from("data/spike/weights"), PathBuf::from);
    let photos_dir = args
        .get(2)
        .map_or(PathBuf::from("data/spike"), PathBuf::from);
    let out_dir = args
        .get(3)
        .map_or(PathBuf::from("data/spike/out"), PathBuf::from);

    ort::init().commit();

    let mut photos: Vec<PathBuf> = std::fs::read_dir(&photos_dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e == "jpg" || e == "jpeg" || e == "png")
        })
        .collect();
    photos.sort();
    assert!(!photos.is_empty(), "no photos in {}", photos_dir.display());

    for candidate in CANDIDATES {
        let weights = weights_dir.join(candidate.weights);
        println!("=== {} ({})", candidate.name, weights.display());
        let mut session = load_session(&weights)?;
        let input_name = session.inputs()[0].name().to_owned();
        let output_name = session.outputs()[0].name().to_owned();
        println!("input: {input_name} output: {output_name}");

        let model_out = out_dir.join(candidate.name);
        std::fs::create_dir_all(&model_out)?;
        // Warmup so the timed runs exclude one-time kernel selection.
        let warmup = image::open(&photos[0])?;
        let _ = run_candidate(
            &mut session,
            &input_name,
            &output_name,
            &warmup,
            candidate.input_size,
        )?;

        for photo in &photos {
            let stem = photo
                .file_stem()
                .expect("photo has a stem")
                .to_string_lossy();
            let img = image::open(photo)?;
            let (mask, pre_ms, infer_ms, post_ms) = run_candidate(
                &mut session,
                &input_name,
                &output_name,
                &img,
                candidate.input_size,
            )?;
            println!(
                "{stem}: pre={pre_ms}ms infer={infer_ms}ms post={post_ms}ms fg={:.3}",
                mask.foreground_fraction()
            );
            mask.to_luma()
                .save(model_out.join(format!("{stem}_mask.png")))?;
            write_cutout(&img, &mask, &model_out.join(format!("{stem}_cutout.png")))?;
        }
    }
    Ok(())
}
