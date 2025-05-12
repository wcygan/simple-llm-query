use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use clap::Parser;
use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;
use std::error::Error;
use std::fs;
use tokio::io::{self, AsyncWriteExt};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The prompt to send to the LLM
    #[arg(short, long)]
    prompt: String,

    /// Optional path to an image file for multimodal input
    #[arg(short, long)]
    image: Option<String>,
}

#[derive(Serialize)]
struct TextContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Serialize)]
struct ImageContent {
    #[serde(rename = "type")]
    content_type: String,
    image_url: ImageUrl,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ContentPart {
    Text(TextContent),
    Image(ImageContent),
}

#[derive(Serialize)]
struct MultimodalChatMessage {
    role: String,
    content: Vec<ContentPart>,
}

fn build_request_content(
    prompt: &str,
    image_path: Option<&str>,
) -> Result<Vec<ContentPart>, Box<dyn Error + Send + Sync>> {
    let mut content_parts = vec![ContentPart::Text(TextContent {
        content_type: "text".into(),
        text: prompt.into(),
    })];

    if let Some(path) = image_path {
        println!("Reading and encoding image: {}", path);
        let image_bytes = fs::read(path)?;
        let encoded_image = BASE64_STANDARD.encode(&image_bytes);

        let mime_type = match path.split('.').last() {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("webp") => "image/webp",
            Some("gif") => "image/gif",
            _ => "application/octet-stream",
        };

        let data_url = format!("data:{};base64,{}", mime_type, encoded_image);

        content_parts.push(ContentPart::Image(ImageContent {
            content_type: "image_url".into(),
            image_url: ImageUrl { url: data_url },
        }));
    }
    Ok(content_parts)
}

async fn handle_streamed_response(
    response: reqwest::Response,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if !response.status().is_success() {
        let status = response.status();
        let text = response
            .text()
            .await
            .unwrap_or_else(|e| format!("Failed to read error body: {}", e));
        eprintln!("Error response from server ({}): {}", status, text);
        return Err(format!("Server returned error: {}", status).into());
    }

    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    let mut handle = io::stdout();

    handle.write_all(b"\n--- LLM Response ---\n").await?;

    while let Some(item) = stream.next().await {
        let chunk = item?;
        buffer.extend_from_slice(&chunk);

        while let Some(pos) = buffer.windows(2).position(|w| w == b"\n\n") {
            let event_data = buffer.drain(..pos + 2).collect::<Vec<u8>>();

            if event_data.is_empty() || event_data == b"\n\n" || event_data.starts_with(b":") {
                continue;
            }

            let text = String::from_utf8_lossy(&event_data);
            for line in text.lines() {
                if let Some(stripped) = line.strip_prefix("data: ") {
                    if stripped.trim() == "[DONE]" {
                        handle.write_all(b"\n[stream closed]\n").await?;
                        handle.flush().await?;
                        return Ok(());
                    }
                    match serde_json::from_str::<Value>(stripped) {
                        Ok(json) => {
                            if let Some(delta) = json
                                .pointer("/choices/0/delta/content")
                                .and_then(Value::as_str)
                            {
                                handle.write_all(delta.as_bytes()).await?;
                                handle.flush().await?;
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "\n[Warning: Failed to parse JSON chunk: '{}', Error: {}]\n",
                                stripped, e
                            );
                        }
                    }
                }
            }
        }
    }

    if !buffer.is_empty() {
        eprintln!(
            "\n[Warning: Stream ended unexpectedly. Remaining buffer: {:?}]\n",
            String::from_utf8_lossy(&buffer)
        );
    } else {
        eprintln!("\n[Warning: Stream ended without a final [DONE] message.]\n");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let cli = Cli::parse();

    let url = "http://localhost:8080/v1/chat/completions";

    let content = build_request_content(&cli.prompt, cli.image.as_deref())?;

    let messages = vec![MultimodalChatMessage {
        role: "user".into(),
        content,
    }];

    let body = serde_json::json!({
        "stream": true,
        "messages": messages,
    });

    let client = reqwest::Client::new();

    println!("Sending request to {}", url);
    let resp = client.post(url).json(&body).send().await?;

    handle_streamed_response(resp).await?;

    Ok(())
}
