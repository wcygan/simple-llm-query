use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;
use std::error::Error;
use std::{env, fs};
use tokio::io::{self, AsyncWriteExt};

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
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

async fn stream_chat(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    messages: &[ChatMessage],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // build JSON body with stream:true
    let body = serde_json::json!({
        "model": model,
        "stream": true,
        "messages": messages,
    });

    let resp = client.post(url).json(&body).send().await?;
    let mut stream = resp.bytes_stream();

    // Accumulate partial chunks to handle SSE framing
    let mut buffer = Vec::new();
    let mut handle = io::stdout();

    while let Some(item) = stream.next().await {
        let chunk = item?;
        buffer.extend_from_slice(&chunk);

        // split on "\n\n" which delimits SSE events
        while let Some(pos) = buffer.windows(2).position(|w| w == b"\n\n") {
            let event = buffer.drain(..pos + 2).collect::<Vec<u8>>();
            // Skip empty events or comment lines
            if event.is_empty() || event.starts_with(b":") {
                continue;
            }
            let text = String::from_utf8_lossy(&event);
            for line in text.lines() {
                if let Some(stripped) = line.strip_prefix("data: ") {
                    if stripped.trim() == "[DONE]" {
                        handle.write_all(b"\n[stream closed]\n").await.unwrap();
                        return Ok(());
                    }
                    // parse the JSON chunk
                    if let Ok(json) = serde_json::from_str::<Value>(stripped) {
                        if let Some(delta) = json
                            .pointer("/choices/0/delta/content")
                            .and_then(Value::as_str)
                        {
                            handle.write_all(delta.as_bytes()).await.unwrap();
                            handle.flush().await.unwrap();
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

async fn chat_with_image(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    image_path: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // read image bytes and base64 encode
    let image_bytes = fs::read(image_path)?;
    let encoded_image = BASE64_STANDARD.encode(&image_bytes);

    // Determine mime type (basic inference based on extension)
    let mime_type = match image_path.split('.').last() {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        _ => "application/octet-stream", // Default or consider erroring
    };

    let data_url = format!("data:{};base64,{}", mime_type, encoded_image);

    // Construct multimodal message payload
    let messages = vec![MultimodalChatMessage {
        role: "user".into(),
        content: vec![
            ContentPart::Text(TextContent {
                content_type: "text".into(),
                text: prompt.into(),
            }),
            ContentPart::Image(ImageContent {
                content_type: "image_url".into(),
                image_url: ImageUrl { url: data_url },
            }),
        ],
    }];

    // build JSON body - set stream to true
    let body = serde_json::json!({
        "model": model,
        "stream": true, // Set to true for streaming response
        "messages": messages,
    });

    // Send JSON request
    let resp = client.post(url).json(&body).send().await?;

    // Check for errors before parsing JSON
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await?;
        eprintln!("Error response from server ({}): {}", status, text);
        return Err(format!("Server returned error: {}", status).into());
    }

    // --- Start Streaming Logic --- Modified from stream_chat
    let mut stream = resp.bytes_stream();
    let mut buffer = Vec::new();
    let mut handle = io::stdout();

    handle.write_all(b"\n\n=== multimodal stream ===\n").await?;

    while let Some(item) = stream.next().await {
        let chunk = item?;
        buffer.extend_from_slice(&chunk);

        // split on "\n\n" which delimits SSE events
        while let Some(pos) = buffer.windows(2).position(|w| w == b"\n\n") {
            let event = buffer.drain(..pos + 2).collect::<Vec<u8>>();
            // Skip empty events or comment lines
            if event.is_empty() || event.starts_with(b":") {
                continue;
            }
            let text = String::from_utf8_lossy(&event);
            for line in text.lines() {
                if let Some(stripped) = line.strip_prefix("data: ") {
                    if stripped.trim() == "[DONE]" {
                        handle.write_all(b"\n[stream closed]\n").await.unwrap();
                        return Ok(());
                    }
                    // parse the JSON chunk
                    if let Ok(json) = serde_json::from_str::<Value>(stripped) {
                        if let Some(delta) = json
                            .pointer("/choices/0/delta/content")
                            .and_then(Value::as_str)
                        {
                            handle.write_all(delta.as_bytes()).await.unwrap();
                            handle.flush().await.unwrap();
                        }
                    }
                }
            }
        }
    }
    // --- End Streaming Logic ---

    // This part is now unreachable if the stream ends correctly with [DONE]
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // URL and model name
    let url = "http://localhost:8080/v1/chat/completions";
    let model = "llava-v1.5-7b-Q4_K.gguf"; // Example multimodal model name

    // Define a simple text prompt
    let text_prompt = "Describe this image. Be succinct, and display calorie details in a JSON format. Do not reply with ANYTHING other than the JSON output. JSON keys are name, calories, protein, carbs, and fat. Combine the entire dish into one entry.";

    // Define simple text-only messages for the stream_chat case
    let text_messages = vec![ChatMessage {
        role: "user".into(),
        content: "Say hello in Hungarian".into(),
    }];

    let client = reqwest::Client::new();

    // if an image path is passed as first arg, do multimodal
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let image_path = &args[1];
        // Use the text_prompt for the multimodal request
        println!(
            "Sending image `{}` with prompt \"{}\" for multimodal inference...",
            image_path, text_prompt
        );
        chat_with_image(&client, url, model, text_prompt, image_path).await?;
    } else {
        println!("Starting text stream...");
        // Use text_messages for the text-only request
        stream_chat(&client, url, model, &text_messages).await?;
    }

    Ok(())
}
