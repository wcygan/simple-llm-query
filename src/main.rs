use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
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

    /// Whether to perform a review step after the initial response
    #[arg(long)]
    review: bool,
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

async fn handle_and_capture_streamed_response(
    response: reqwest::Response,
    capture: bool,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
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
    let mut captured_response = if capture { Some(String::new()) } else { None };

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
                        handle.write_all(b"\n").await?;
                        handle.write_all(b"[stream closed]\n").await?;
                        handle.flush().await?;
                        return Ok(captured_response);
                    }
                    match serde_json::from_str::<Value>(stripped) {
                        Ok(json) => {
                            if let Some(delta) = json
                                .pointer("/choices/0/delta/content")
                                .and_then(Value::as_str)
                            {
                                handle.write_all(delta.as_bytes()).await?;
                                handle.flush().await?;
                                if let Some(ref mut captured) = captured_response {
                                    captured.push_str(delta);
                                }
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

    Ok(captured_response)
}

// Helper function to send request and handle response
async fn send_llm_request(
    client: &reqwest::Client,
    url: &str,
    messages: Vec<MultimodalChatMessage>,
    capture_response: bool,
    response_title: &str, // e.g., "Initial Response", "Review Response"
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let body = serde_json::json!({
        "stream": true,
        "messages": messages,
    });

    // Print title before sending request
    io::stdout()
        .write_all(format!("\n--- {} ---\n", response_title).as_bytes())
        .await?;
    io::stdout().flush().await?;

    println!("Sending request to {}", url);
    let resp = client.post(url).json(&body).send().await?;

    handle_and_capture_streamed_response(resp, capture_response).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let cli = Cli::parse();

    let url = "http://localhost:8080/v1/chat/completions";
    let client = reqwest::Client::new();

    // --- Initial Request ---
    println!("Building initial request content...");
    let initial_content = build_request_content(&cli.prompt, cli.image.as_deref())?;
    let initial_messages = vec![MultimodalChatMessage {
        role: "user".into(),
        content: initial_content,
    }];

    let initial_response_result = send_llm_request(
        &client,
        url,
        initial_messages,
        cli.review, // Capture only if review flag is set
        "LLM Response",
    )
    .await;

    let first_response = match initial_response_result {
        Ok(response_opt) => response_opt,
        Err(e) => {
            eprintln!("Error during initial LLM request: {}", e);
            return Err(e);
        }
    };

    // --- Review Request (if applicable) ---
    if cli.review {
        if let Some(first_response_text) = first_response {
            println!("\nBuilding review request content...");

            let review_prompt_text = format!(
                "Original prompt: \"{}\"\n\nFirst response: \"{}\"\n\nPlease review the first response based on the original prompt (and image, if provided below). Provide a final, potentially revised response.",
                cli.prompt, first_response_text
            );

            // Start building content for the review request
            let mut review_content_parts = vec![ContentPart::Text(TextContent {
                content_type: "text".into(),
                text: review_prompt_text,
            })];

            // Re-add image if it was present in the initial request
            if let Some(image_path) = cli.image.as_deref() {
                println!("Re-encoding image for review request: {}", image_path);
                // Need to re-read and encode the image for the review step
                // (Could optimize by storing the data_url from the first step, but this is simpler for now)
                let image_bytes = fs::read(image_path)?;
                let encoded_image = BASE64_STANDARD.encode(&image_bytes);
                let mime_type = match image_path.split('.').last() {
                    Some("png") => "image/png",
                    Some("jpg") | Some("jpeg") => "image/jpeg",
                    Some("webp") => "image/webp",
                    Some("gif") => "image/gif",
                    _ => "application/octet-stream",
                };
                let data_url = format!("data:{};base64,{}", mime_type, encoded_image);

                review_content_parts.push(ContentPart::Image(ImageContent {
                    content_type: "image_url".into(),
                    image_url: ImageUrl { url: data_url },
                }));
            }

            let review_messages = vec![MultimodalChatMessage {
                role: "user".into(),
                content: review_content_parts,
            }];

            // Send the review request (don't need to capture this response)
            if let Err(e) = send_llm_request(
                &client,
                url,
                review_messages,
                false, // Don't capture review response
                "Review Response",
            )
            .await
            {
                eprintln!("Error during review LLM request: {}", e);
                // Decide if we should return Err here or just warn
                return Err(e);
            }
        } else {
            eprintln!(
                "[Warning: Review step requested, but failed to capture the initial response.]"
            );
            // Potentially return an error here if capturing was essential
            // For now, we just warn and continue (program finishes)
        }
    }

    Ok(())
}
