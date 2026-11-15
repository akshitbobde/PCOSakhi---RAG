import json

CHUNKS_PATH = "Scripts/output/chunks.json"


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("================================")
    print("CHUNK QUALITY CHECK")
    print("================================")

    print("Total chunks:", len(chunks))

    if not chunks:
        print("ERROR: No chunks found.")
        return

    # Basic field check
    required_fields = [
        "chunk_id",
        "source_id",
        "source_title",
        "chapter",
        "section",
        "page",
        "content_type",
        "text"
    ]

    missing_fields = []

    for i, chunk in enumerate(chunks):
        for field in required_fields:
            if field not in chunk:
                missing_fields.append((i, field))

    if missing_fields:
        print("\nMissing fields found:")
        for item in missing_fields[:20]:
            print(item)
    else:
        print("\nAll required fields are present.")

    # Word count
    word_counts = [
        len(chunk["text"].split())
        for chunk in chunks
        if chunk.get("text")
    ]

    print("\nWord count:")
    print("Minimum:", min(word_counts))
    print("Maximum:", max(word_counts))
    print("Average:", round(sum(word_counts) / len(word_counts), 2))

    # Very small chunks
    small_chunks = [
        chunk
        for chunk in chunks
        if len(chunk.get("text", "").split()) < 50
    ]

    print("\nChunks below 50 words:", len(small_chunks))

    # Show first 5 chunks
    print("\n================================")
    print("FIRST 5 CHUNKS")
    print("================================")

    for chunk in chunks[:5]:
        print("\n--------------------------------")
        print("Chunk ID:", chunk["chunk_id"])
        print("Chapter:", chunk["chapter"])
        print("Section:", chunk["section"])
        print("Page:", chunk["page"])
        print("Words:", len(chunk["text"].split()))
        print("Text:")
        print(chunk["text"][:800])

        # Show unusually short chunks
    print("\n========== SHORT CHUNKS ==========")

    for chunk in chunks:
        word_count = len(chunk.get("text", "").split())

        if word_count < 50:
            print("\n------------------------------")
            print("Chunk ID:", chunk.get("chunk_id"))
            print("Page:", chunk.get("page"))
            print("Section:", chunk.get("section"))
            print("Words:", word_count)
            print("Text:", chunk.get("text"))

    # Show unusually large chunks
    print("\n========== LARGE CHUNKS ==========")

    for chunk in chunks:
        word_count = len(chunk.get("text", "").split())

        if word_count > 500:
            print("\n------------------------------")
            print("Chunk ID:", chunk.get("chunk_id"))
            print("Page:", chunk.get("page"))
            print("Section:", chunk.get("section"))
            print("Words:", word_count)
            print("Text:", chunk.get("text")[:500])

if __name__ == "__main__":
    main()
