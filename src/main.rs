use futures::StreamExt;
use reqwest::multipart::{Form, Part};
use serde::Serialize;
use serde_json::Value;
use std::{env, fs};
use tokio::io::{self, AsyncWriteExt};
use std::error::Error;

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
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
    messages: &[ChatMessage],
    image_path: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // read image bytes
    let image_bytes = fs::read(image_path)?;
    let messages_json = serde_json::to_string(messages).unwrap();

    // build multipart form
    let form = Form::new()
        .text("model", model.to_string())
        .text("stream", "false")
        .text("messages", messages_json)
        .part(
            "image_file",
            Part::bytes(image_bytes)
                .file_name("upload.png")
                .mime_str("image/png")?,
        );

    let resp = client.post(url).multipart(form).send().await?;
    let json: Value = resp.json().await?;
    println!("\n\n=== multimodal response ===\n{}", json);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // URL and model name
    let url = "http://localhost:8080/v1/chat/completions";
    let model = "my-model.gguf";

    // define a simple chat
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "Say hello in Hungarian".into(),
    }];

    let client = reqwest::Client::new();

    // if an image path is passed as first arg, do multimodal
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let image_path = &args[1];
        println!("Sending image `{}` for multimodal inference...", image_path);
        chat_with_image(&client, url, model, &messages, image_path).await?;
    } else {
        println!("Starting text stream...");
        stream_chat(&client, url, model, &messages).await?;
    }

    Ok(())
}