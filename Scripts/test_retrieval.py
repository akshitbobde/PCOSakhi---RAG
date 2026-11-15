
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

PROJECT_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = PROJECT_DIR / "service-account.json"

load_dotenv(PROJECT_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not CREDENTIALS_FILE.exists():
    raise FileNotFoundError("service-account.json not found")

client = genai.Client(api_key=api_key)

db = firestore.Client.from_service_account_json(
    str(CREDENTIALS_FILE)
)

MODEL_NAME = "gemini-embedding-2"
DIMENSIONS = 768


def main():
    question = input("Ask PCOSakhi a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # Embed the user's question using the same model and dimensions.
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=question,
        config=types.EmbedContentConfig(
            output_dimensionality=DIMENSIONS
        ),
    )

    query_vector = response.embeddings[0].values

    if len(query_vector) != DIMENSIONS:
        raise ValueError("Unexpected query embedding dimensions")

    # Retrieve the 5 closest chunks from Firestore.
    query = db.collection("chunks").find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_vector),
        distance_measure=DistanceMeasure.COSINE,
        limit=5,
        distance_result_field="vector_distance",
    )

    results = list(query.stream())

    if not results:
        print("\nNo matching chunks found.")
        return

    print(f"\nRetrieved {len(results)} chunks:\n")

    for rank, doc in enumerate(results, start=1):
        data = doc.to_dict()

        print("=" * 65)
        print(f"Rank: {rank}")
        print(f"Chunk ID: {doc.id}")
        print(f"Chapter: {data.get('chapter', 'Not available')}")
        print(f"Section: {data.get('section', 'Not available')}")
        print(f"PDF page: {data.get('page', 'Not available')}")
        print(f"Cosine distance: {data.get('vector_distance')}")
        print("\nRetrieved text:")
        print(data.get("text", "No text found"))

    print("\nRetrieval test completed.")


if __name__ == "__main__":
    main()
