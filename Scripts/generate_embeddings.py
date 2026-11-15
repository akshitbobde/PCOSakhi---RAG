import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "Output" / "chunks.json"
OUTPUT_FILE = PROJECT_DIR / "Output" / "embedded_chunks.json"

MODEL_NAME = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768

# Load API key from the project's .env file
load_dotenv(PROJECT_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input JSON not found: {INPUT_FILE}\n"
            "Check the JSON filename in the output folder."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("The input JSON must contain a non-empty list.")

    results = []
    total = len(chunks)

    print(f"Loaded {total} chunks.")
    print("Generating embeddings. Please wait...")

    for index, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()

        if not text:
            print(f"Skipping empty chunk: {chunk.get('chunk_id')}")
            continue

        # Retry transient errors up to 3 attempts
        for attempt in range(1, 4):
            try:
                response = client.models.embed_content(
                    model=MODEL_NAME,
                    contents=text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIMENSIONS
                    ),
                )

                embedding = response.embeddings[0].values

                if len(embedding) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Unexpected vector size: {len(embedding)}"
                    )

                # Preserve the original chunk metadata
                record = dict(chunk)
                record["embedding"] = embedding
                record["embedding_model"] = MODEL_NAME
                record["embedding_dimensions"] = EMBEDDING_DIMENSIONS

                results.append(record)
                break

            except Exception as error:
                if attempt == 3:
                    raise RuntimeError(
                        f"Failed on chunk {chunk.get('chunk_id')} "
                        f"after 3 attempts. Error: {error}"
                    ) from error

                wait_seconds = attempt * 2
                print(
                    f"Attempt {attempt} failed for chunk "
                    f"{chunk.get('chunk_id')}. "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)

        if index % 10 == 0 or index == total:
            print(f"Processed {index}/{total} chunks.")

        # Small pause to reduce request bursts
        time.sleep(0.3)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False)

    print("\nEmbedding generation complete.")
    print(f"Input chunks: {total}")
    print(f"Embeddings created: {len(results)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()