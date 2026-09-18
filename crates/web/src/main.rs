//! Local web shell for smart-techpack.
//!
//! Slice 1: renders the checked-in fixture as a read-only project summary.
//! Slice 2: photo upload (`POST /projects/:id/photo`) with offline cloth-seg
//! inference, stored under `data/projects/:id/`, plus a comparison page.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use askama::Template;
use axum::{
    Router,
    extract::{Multipart, Path as UrlPath, State},
    http::StatusCode,
    response::{Html, IntoResponse, Redirect},
    routing::get,
};
use serde::{Deserialize, Serialize};
use techpack_core::{Project, load_project};
use techpack_vision::{ClothSeg, IngestError, Segmenter, cutout_rgba, downscale_long_edge, ingest};

const FIXTURE_JSON: &str = include_str!("../../../fixtures/jacket.json");
/// Weights location relative to the server's working directory.
/// Gitignored; see `docs/models.md` for the fetch step.
const WEIGHTS_PATH: &str = "data/weights/cloth-seg-u2net.onnx";
/// Extra headroom above the vision byte cap for multipart framing.
const UPLOAD_SLACK: usize = 1024 * 1024;

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    project: Project,
}

#[derive(Template)]
#[template(path = "photo.html")]
struct PhotoTemplate {
    id: String,
    meta: PhotoMeta,
}

/// Stored alongside each upload so the comparison page needs no inference.
#[derive(Debug, Serialize, Deserialize)]
struct PhotoMeta {
    format: String,
    width: u32,
    height: u32,
    fg_fraction: f64,
}

#[derive(Clone)]
struct AppState {
    data_dir: PathBuf,
    segmenter: Option<Arc<ClothSeg>>,
}

async fn index() -> Html<String> {
    let project = load_project(FIXTURE_JSON).expect("fixture must parse");
    let rendered = IndexTemplate { project }
        .render()
        .expect("template must render");
    Html(rendered)
}

/// Project ids are directory names under `data/projects/`: reject anything
/// that could escape (`..`, `/`) or surprise the filesystem.
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn project_dir(state: &AppState, id: &str) -> PathBuf {
    state.data_dir.join("projects").join(id)
}

async fn photo_page(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
) -> impl IntoResponse {
    if !valid_id(&id) {
        return (StatusCode::BAD_REQUEST, "invalid project id").into_response();
    }
    let meta_path = project_dir(&state, &id).join("meta.json");
    let bytes = match tokio::fs::read(&meta_path).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::NOT_FOUND, "no photo for this project yet").into_response(),
    };
    let meta: PhotoMeta = match serde_json::from_slice(&bytes) {
        Ok(m) => m,
        Err(_) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "unreadable meta.json").into_response();
        }
    };
    let rendered = match (PhotoTemplate { id, meta }).render() {
        Ok(html) => html,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "template failed").into_response(),
    };
    Html(rendered).into_response()
}

async fn photo_file(
    State(state): State<AppState>,
    UrlPath((id, file)): UrlPath<(String, String)>,
) -> impl IntoResponse {
    if !valid_id(&id) {
        return (StatusCode::BAD_REQUEST, "invalid project id").into_response();
    }
    let (name, mime): (&str, &str) = match file.as_str() {
        "original" => return original_file(&state, &id).await,
        "mask" => ("mask.png", "image/png"),
        "preview" => ("preview.png", "image/png"),
        _ => return (StatusCode::NOT_FOUND, "unknown file").into_response(),
    };
    match tokio::fs::read(project_dir(&state, &id).join(name)).await {
        Ok(bytes) => ([("content-type", mime)], bytes).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "no photo for this project yet").into_response(),
    }
}

/// The original keeps its upload encoding, so resolve the extension first.
async fn original_file(state: &AppState, id: &str) -> axum::response::Response {
    let dir = project_dir(state, id);
    for (name, mime) in [
        ("original.jpg", "image/jpeg"),
        ("original.png", "image/png"),
    ] {
        if let Ok(bytes) = tokio::fs::read(dir.join(name)).await {
            return ([("content-type", mime)], bytes).into_response();
        }
    }
    (StatusCode::NOT_FOUND, "no photo for this project yet").into_response()
}

async fn upload_photo(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    if !valid_id(&id) {
        return (StatusCode::BAD_REQUEST, "invalid project id".to_owned()).into_response();
    }
    // Collect the first file field, enforcing the vision byte cap while
    // streaming so a hostile Content-Length never fills memory.
    let mut bytes: Option<Vec<u8>> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() != Some("photo") {
            continue;
        }
        let mut buf = Vec::new();
        let mut too_big = false;
        let mut field = field;
        while let Ok(Some(chunk)) = field.chunk().await {
            buf.extend_from_slice(&chunk);
            if buf.len() > techpack_vision::ingest::MAX_BYTES + UPLOAD_SLACK {
                too_big = true;
                break;
            }
        }
        if too_big {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                "upload exceeds the 15 MB limit".to_owned(),
            )
                .into_response();
        }
        bytes = Some(buf);
        break;
    }
    let bytes = match bytes {
        Some(b) => b,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "missing 'photo' file field".to_owned(),
            )
                .into_response();
        }
    };

    let photo = match ingest(&bytes) {
        Ok(p) => p,
        Err(IngestError::TooLargeBytes(_) | IngestError::TooManyPixels(_)) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                "photo exceeds size limits".to_owned(),
            )
                .into_response();
        }
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    let segmenter = match &state.segmenter {
        Some(s) => Arc::clone(s),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "segmentation weights are not installed; see docs/models.md".to_owned(),
            )
                .into_response();
        }
    };

    // Inference is a blocking multi-second CPU call: keep it off the
    // async runtime. The image moves into the thread and back.
    let image = photo.image.clone();
    let mask = match tokio::task::spawn_blocking(move || segmenter.mask(&image)).await {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
        Err(_) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "inference task failed").into_response();
        }
    };

    if let Err(e) = store_project(&state, &id, &bytes, &photo, &mask).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
    }
    Redirect::to(&format!("/projects/{id}/photo")).into_response()
}

/// Persist original + mask + preview + meta under `data/projects/:id/`.
async fn store_project(
    state: &AppState,
    id: &str,
    raw: &[u8],
    photo: &techpack_vision::IngestedPhoto,
    mask: &techpack_vision::Mask,
) -> Result<(), String> {
    let dir = project_dir(state, id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("cannot create project dir: {e}"))?;
    let ext = photo.format.extension();
    tokio::fs::write(dir.join(format!("original.{ext}")), raw)
        .await
        .map_err(|e| format!("cannot store original: {e}"))?;
    write_png(
        &dir.join("mask.png"),
        mask.width,
        mask.height,
        image::ExtendedColorType::L8,
        &mask.to_luma().into_raw(),
    )
    .await?;
    let preview_img = downscale_long_edge(&photo.image, techpack_vision::ingest::PREVIEW_LONG_EDGE);
    let preview_mask = mask.upscale_to(preview_img.width(), preview_img.height());
    let preview = cutout_rgba(&preview_img, &preview_mask);
    let (pw, ph) = (preview.width(), preview.height());
    write_png(
        &dir.join("preview.png"),
        pw,
        ph,
        image::ExtendedColorType::Rgba8,
        &preview.into_raw(),
    )
    .await?;
    let meta = PhotoMeta {
        format: ext.to_owned(),
        width: photo.image.width(),
        height: photo.image.height(),
        fg_fraction: mask.foreground_fraction(),
    };
    let meta_json =
        serde_json::to_vec_pretty(&meta).map_err(|e| format!("cannot encode meta: {e}"))?;
    tokio::fs::write(dir.join("meta.json"), meta_json)
        .await
        .map_err(|e| format!("cannot store meta: {e}"))?;
    Ok(())
}

async fn write_png(
    path: &Path,
    width: u32,
    height: u32,
    color: image::ExtendedColorType,
    bytes: &[u8],
) -> Result<(), String> {
    use image::ImageEncoder;
    let mut buf = Vec::new();
    image::codecs::png::PngEncoder::new(&mut buf)
        .write_image(bytes, width, height, color)
        .map_err(|e| format!("cannot encode {}: {e}", path.display()))?;
    tokio::fs::write(path, buf)
        .await
        .map_err(|e| format!("cannot store {}: {e}", path.display()))
}

#[tokio::main]
async fn main() {
    let segmenter = match ClothSeg::load(Path::new(WEIGHTS_PATH)) {
        Ok(s) => {
            println!("loaded segmentation weights from {WEIGHTS_PATH}");
            Some(Arc::new(s))
        }
        Err(e) => {
            println!("no segmentation weights ({e}); uploads will return 503 — see docs/models.md");
            None
        }
    };
    let state = AppState {
        data_dir: PathBuf::from("data"),
        segmenter,
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/projects/{id}/photo", get(photo_page).post(upload_photo))
        .route("/projects/{id}/photo/{file}", get(photo_file))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("bind 127.0.0.1:3000");
    println!("serving on http://127.0.0.1:3000");
    axum::serve(listener, app).await.expect("serve");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> AppState {
        AppState {
            // Nonexistent dir: handlers must answer 404, never panic.
            data_dir: PathBuf::from("test-target-that-does-not-exist"),
            segmenter: None,
        }
    }

    #[test]
    fn project_ids_reject_traversal() {
        assert!(valid_id("demo-1_abc"));
        for bad in [
            "",
            "..",
            "../x",
            "a/b",
            "a b",
            "a.png",
            "x".repeat(65).as_str(),
        ] {
            assert!(!valid_id(bad), "{bad:?} must be rejected");
        }
    }

    #[tokio::test]
    async fn photo_page_rejects_bad_id() {
        let res = photo_page(State(test_state()), UrlPath("../x".to_owned()))
            .await
            .into_response();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn photo_page_missing_project_is_404() {
        let res = photo_page(State(test_state()), UrlPath("demo".to_owned()))
            .await
            .into_response();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn photo_file_rejects_bad_id_and_unknown_file() {
        let res = photo_file(
            State(test_state()),
            UrlPath(("../x".to_owned(), "mask".to_owned())),
        )
        .await
        .into_response();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let res = photo_file(
            State(test_state()),
            UrlPath(("demo".to_owned(), "nope".to_owned())),
        )
        .await
        .into_response();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn photo_file_missing_project_is_404() {
        let res = photo_file(
            State(test_state()),
            UrlPath(("demo".to_owned(), "mask".to_owned())),
        )
        .await
        .into_response();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }
}
