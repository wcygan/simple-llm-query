use std::{error::Error, fs, path::Path};

use async_trait::async_trait;
use base64::Engine;
use bytes::Bytes;
use clap::Parser;
use futures::{StreamExt, pin_mut};
use reqwest::{Client, Url};
use serde::Serialize;
use serde_json::Value;
use tokio::io::{self, AsyncWriteExt};
use tracing::{error, info};

// ------ CLI and Configuration ------

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// The prompt to send to the LLM
    #[arg(short, long)]
    prompt: String,

    /// Optional path to an image file for multimodal input
    #[arg(short, long)]
    image: Option<String>,

    /// Whether to perform a review step after the initial response
    #[arg(long)]
    review: bool,

    /// LLM endpoint (overridden by --llm-endpoint)
    #[arg(long, default_value = "http://localhost:8080/v1/chat/completions")]
    llm_endpoint: String,
}

// ------ Domain Types ------

#[derive(Serialize, Debug)]
#[serde(rename_all = "lowercase")]
enum Role {
    User,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum ContentType {
    Text,
    ImageUrl,
}

#[derive(Serialize)]
struct TextContent {
    #[serde(rename = "type")]
    content_type: ContentType,
    text: String,
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Serialize)]
struct ImageContent {
    #[serde(rename = "type")]
    content_type: ContentType,
    image_url: ImageUrl,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ContentPart {
    Text(TextContent),
    Image(ImageContent),
}

struct ChatMessage {
    role: Role,
    content: Vec<ContentPart>,
}

impl ChatMessage {
    fn new(role: Role, content: Vec<ContentPart>) -> Self {
        Self { role, content }
    }
    fn user(content: Vec<ContentPart>) -> Self {
        Self::new(Role::User, content)
    }
}

// ------ Chat Request Builder ------

struct ChatRequest {
    stream: bool,
    messages: Vec<ChatMessage>,
}

impl ChatRequest {
    fn new() -> Self {
        Self {
            stream: false,
            messages: vec![],
        }
    }

    fn stream(mut self, s: bool) -> Self {
        self.stream = s;
        self
    }

    fn with_messages(mut self, msgs: Vec<ChatMessage>) -> Self {
        self.messages = msgs;
        self
    }
}

// ------ Image Encoding Trait ------

trait ImageEncoder {
    fn encode_path(&self, path: &Path) -> Result<String, Box<dyn Error + Send + Sync>>;
}

struct DataUrlEncoder;

impl ImageEncoder for DataUrlEncoder {
    fn encode_path(&self, path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
        let bytes = fs::read(path)?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let mime = match path.extension().and_then(|s| s.to_str()) {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("webp") => "image/webp",
            Some("gif") => "image/gif",
            _ => "application/octet-stream",
        };
        Ok(format!("data:{};base64,{}", mime, encoded))
    }
}

// ------ Content Builder ------

fn build_request_content(
    prompt: &str,
    image_path: Option<&str>,
    encoder: &dyn ImageEncoder,
) -> Result<Vec<ContentPart>, Box<dyn Error + Send + Sync>> {
    let mut parts = vec![ContentPart::Text(TextContent {
        content_type: ContentType::Text,
        text: prompt.to_string(),
    })];
    if let Some(path_str) = image_path {
        let url = encoder.encode_path(Path::new(path_str))?;
        parts.push(ContentPart::Image(ImageContent {
            content_type: ContentType::ImageUrl,
            image_url: ImageUrl { url },
        }));
    }
    Ok(parts)
}

// ------ Transport Trait ------

#[async_trait]
trait LlmTransport {
    async fn send(
        &self,
        endpoint: &Url,
        request: &ChatRequest,
    ) -> Result<
        impl futures::Stream<Item = Result<Bytes, reqwest::Error>>,
        Box<dyn Error + Send + Sync>,
    >;
}

struct HttpTransport {
    client: Client,
}

impl HttpTransport {
    fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

#[async_trait]
impl LlmTransport for HttpTransport {
    async fn send(
        &self,
        endpoint: &Url,
        request: &ChatRequest,
    ) -> Result<
        impl futures::Stream<Item = Result<Bytes, reqwest::Error>>,
        Box<dyn Error + Send + Sync>,
    > {
        let body = serde_json::json!({
            "stream": request.stream,
            "messages": request.messages.iter().map(|m| {
                serde_json::json!({
                    "role": format!("{:?}", m.role).to_lowercase(),
                    "content": m.content,
                })
            }).collect::<Vec<_>>(),
        });
        let resp = self
            .client
            .post(endpoint.clone())
            .json(&body)
            .send()
            .await?;
        Ok(resp.bytes_stream())
    }
}

// ------ Client ------

struct LlmClient<T: LlmTransport> {
    transport: T,
    endpoint: Url,
}

impl<T: LlmTransport> LlmClient<T> {
    fn new(transport: T, endpoint: Url) -> Self {
        Self {
            transport,
            endpoint,
        }
    }

    async fn chat(
        &self,
        request: ChatRequest,
        capture: bool,
    ) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
        let stream = self.transport.send(&self.endpoint, &request).await?;
        pin_mut!(stream);
        let mut buffer = Vec::new();
        let mut captured = if capture { Some(String::new()) } else { None };
        let mut stdout = io::stdout();

        while let Some(item) = stream.next().await {
            let chunk = item?;
            buffer.extend_from_slice(&chunk);
            while let Some(pos) = buffer.windows(2).position(|w| w == b"\n\n") {
                let data = buffer.drain(..pos + 2).collect::<Vec<_>>();
                let text = String::from_utf8_lossy(&data);
                for line in text.lines() {
                    if let Some(stripped) = line.strip_prefix("data: ") {
                        if stripped.trim() == "[DONE]" {
                            stdout.write_all(b"\n").await?;
                            return Ok(captured);
                        }
                        if let Ok(json) = serde_json::from_str::<Value>(stripped) {
                            if let Some(delta) = json
                                .pointer("/choices/0/delta/content")
                                .and_then(Value::as_str)
                            {
                                stdout.write_all(delta.as_bytes()).await?;
                                stdout.flush().await?;
                                if let Some(ref mut cap) = captured {
                                    cap.push_str(delta);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(captured)
    }
}

// ------ Main ------

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let endpoint = Url::parse(&cli.llm_endpoint)?;
    let client = LlmClient::new(HttpTransport::new(), endpoint);
    let encoder = DataUrlEncoder;

    info!("Building initial request content...");
    let parts = build_request_content(&cli.prompt, cli.image.as_deref(), &encoder)?;
    let initial_req = ChatRequest::new()
        .stream(true)
        .with_messages(vec![ChatMessage::user(parts)]);

    let first = match client.chat(initial_req, cli.review).await {
        Ok(res) => res,
        Err(e) => {
            error!("LLM request failed: {}", e);
            return Err(e);
        }
    };

    if cli.review {
        let Some(text) = first else {
            error!("No response captured for review step.");
            return Ok(());
        };
        println!();
        info!("Building review request...");
        let review_prompt = format!(
            "Original prompt: \"{}\"\n\nFirst response: \"{}\"\n\nPlease review and revise.",
            cli.prompt, text
        );
        let review_parts = build_request_content(&review_prompt, cli.image.as_deref(), &encoder)?;
        let review_req = ChatRequest::new()
            .stream(true)
            .with_messages(vec![ChatMessage::user(review_parts)]);
        client.chat(review_req, false).await?;
    }

    Ok(())
}
