//! Local web shell for smart-techpack. Slice 1: renders the checked-in
//! fixture as a read-only project summary. No uploads, no editing yet.

use askama::Template;
use axum::{Router, response::Html, routing::get};
use techpack_core::{Project, load_project};

const FIXTURE_JSON: &str = include_str!("../../../fixtures/jacket.json");

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    project: Project,
}

async fn index() -> Html<String> {
    let project = load_project(FIXTURE_JSON).expect("fixture must parse");
    let rendered = IndexTemplate { project }
        .render()
        .expect("template must render");
    Html(rendered)
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(index));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("bind 127.0.0.1:3000");
    println!("serving on http://127.0.0.1:3000");
    axum::serve(listener, app).await.expect("serve");
}
