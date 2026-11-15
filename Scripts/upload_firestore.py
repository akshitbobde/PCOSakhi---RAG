
import json
from pathlib import Path

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "Output" / "embedded_chunks.json"
CREDENTIALS_FILE = PROJECT_DIR / "service-account.json"

COLLECTION_NAME = "chunks"
EXPECTED_DIMENSIONS = 768
BATCH_SIZE = 400


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"Missing credentials file: {CREDENTIALS_FILE}"
        )

    with CREDENTIALS_FILE.open("r", encoding="utf-8") as file:
        credentials_info = json.load(file)

    project_id = credentials_info.get("project_id")
    if not project_id:
        raise ValueError("project_id is missing from service-account.json")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("embedded_chunks.json must contain a non-empty list")

    # Validate everything before writing any documents.
    seen_ids = set()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        embedding = chunk.get("embedding")

        if not chunk_id or chunk_id in seen_ids:
            raise ValueError(f"Missing or duplicate chunk_id: {chunk_id}")

        seen_ids.add(chunk_id)

        if not isinstance(embedding, list):
            raise ValueError(f"Embedding missing for {chunk_id}")

        if len(embedding) != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"{chunk_id} has {len(embedding)} dimensions; "
                f"expected {EXPECTED_DIMENSIONS}"
            )

        if not chunk.get("text", "").strip():
            raise ValueError(f"Text missing for {chunk_id}")

    db = firestore.Client.from_service_account_json(
        str(CREDENTIALS_FILE),
        project=project_id,
    )

    collection = db.collection(COLLECTION_NAME)
    uploaded = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[start:start + BATCH_SIZE]
        batch = db.batch()

        for chunk in batch_chunks:
            document_data = {
                key: value
                for key, value in chunk.items()
                if key != "embedding"
            }

            document_data["embedding"] = Vector(
                [float(value) for value in chunk["embedding"]]
            )

            document_ref = collection.document(chunk["chunk_id"])
            batch.set(document_ref, document_data)

        batch.commit()
        uploaded += len(batch_chunks)
        print(f"Uploaded {uploaded}/{len(chunks)} chunks")

    print("\nUpload completed successfully.")
    print(f"Firebase project: {project_id}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents uploaded: {uploaded}")


if __name__ == "__main__":
    main()
