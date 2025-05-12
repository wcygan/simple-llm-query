# LLM Query in Rust

Goal: self-host an AI model, and write a Rust program to query it.

## Quickstart

Start an LLM with [llama.cpp](https://github.com/ggml-org/llama.cpp)

```bash
brew install llama.cpp
llama-server -hf ggml-org/gemma-3-12b-it-GGUF
```

Run the program:

```bash
cargo run -- --prompt "What is the capital of France?"
```

Output:

```bash
The capital of France is **Paris**.
```

Run with an image:

```bash
cargo run -- --prompt "What is in this image? Describe it in 5 words or less." --image food.jpeg
```

Output:

```bash
Colorful, healthy, Mediterranean-style bowl.
```

Run with a review step (have the model review its own response):

```bash
cargo run -- --prompt "What is in this image? Describe it in 5 words or less." --image food.jpeg --review
```

Output:

```bash
Okay, here's a review and some revised responses, keeping the 5-word limit in mind.

**Review of Original Response:**

"Colorful, fresh, and healthy bowl" is a decent start, but it's a bit wordy for the prompt's limit. "Healthy" is also a subjective assessment.

**Revised Responses (Choose one):**

*   **Vegetable and chickpea bowl.** (Concise and descriptive)
*   **Colorful Mediterranean-style salad bowl.** (Hints at the flavors)
*   **Fresh salad with avocado, tomatoes.** (Highlights key ingredients)

I think "Vegetable and chickpea bowl" is the strongest option.
```