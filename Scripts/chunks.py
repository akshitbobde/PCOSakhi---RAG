import fitz
import re
import json
import os


# ==========================================
# 1. FILE LOCATIONS
# ==========================================

PDF_PATH = "Evidence-Based-Guidelines-2023.pdf"
OUTPUT_PATH = "Output/chunks.json"


# ==========================================
# 2. BASIC SETTINGS
# ==========================================

SOURCE_ID = "PCOS_001"

SOURCE_TITLE = (
    "International Evidence-based Guideline for the "
    "assessment and management of polycystic ovary syndrome 2023"
)

# Approximate target size in words
TARGET_WORDS = 350

# Number of words carried into the next chunk
OVERLAP_WORDS = 50


# ==========================================
# 3. CHAPTER NAMES
# ==========================================

CHAPTERS = {
    "Chapter One": "Chapter One",
    "Chapter Two": "Chapter Two",
    "Chapter Three": "Chapter Three",
    "Chapter Four": "Chapter Four",
    "Chapter Five": "Chapter Five",
}


# ==========================================
# 4. CLEAN TEXT
# ==========================================

def clean_text(text):
    """
    Clean common PDF extraction problems.
    """

    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")

    # Remove form-feed characters
    text = text.replace("\f", " ")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ==========================================
# 5. DETECT CHAPTER
# ==========================================

def detect_chapter(text):
    """
    Check whether this page contains
    Chapter One through Chapter Five.
    """

    for chapter in CHAPTERS:

        if re.search(
            r"\b" + re.escape(chapter) + r"\b",
            text,
            re.IGNORECASE
        ):
            return CHAPTERS[chapter]

    return None


# ==========================================
# 6. DETECT SECTION
# ==========================================

def detect_section(text):
    """
    Detect section numbers such as:
    1.1
    1.2
    3.3
    4.11
    5.7.1
    """

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        match = re.match(
            r"^(\d+\.\d+(?:\.\d+)?)\s+(.+)$",
            line
        )

        if match:

            section_number = match.group(1)
            section_title = match.group(2).strip()

            # Ignore extremely long lines
            if len(section_title) < 180:

                return f"{section_number} {section_title}"

    return None


# ==========================================
# 7. SPLIT TEXT INTO PARAGRAPHS
# ==========================================

def split_into_paragraphs(text):

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    cleaned = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        # Ignore extremely short pieces
        if len(paragraph) < 50:
            continue

        cleaned.append(paragraph)

    return cleaned


# ==========================================
# 8. CREATE CHUNKS
# ==========================================

def create_chunks_from_paragraphs(
    paragraphs,
    page_number,
    chapter,
    section,
    start_chunk_number
):

    chunks = []

    current_words = []

    chunk_number = start_chunk_number

    for paragraph in paragraphs:

        words = paragraph.split()

        # Add paragraph if it keeps us within target size
        if len(current_words) + len(words) <= TARGET_WORDS:

            current_words.extend(words)

        else:

            # Save current chunk
            if current_words:

                chunk_text = " ".join(current_words)

                chunk_id = (
                    f"{SOURCE_ID}_"
                    f"P{page_number}_"
                    f"{chunk_number:04d}"
                )

                chunks.append({
                    "chunk_id": chunk_id,
                    "source_id": SOURCE_ID,
                    "source_title": SOURCE_TITLE,
                    "chapter": chapter,
                    "section": section,
                    "page": page_number,
                    "content_type": "guideline_content",
                    "text": chunk_text
                })

                chunk_number += 1

                # Keep overlap
                if len(current_words) > OVERLAP_WORDS:

                    current_words = current_words[
                        -OVERLAP_WORDS:
                    ]

                else:

                    current_words = []

            # Add current paragraph
            current_words.extend(words)

    # Save final chunk
    if current_words:

        chunk_text = " ".join(current_words)

        chunk_id = (
            f"{SOURCE_ID}_"
            f"P{page_number}_"
            f"{chunk_number:04d}"
        )

        chunks.append({
            "chunk_id": chunk_id,
            "source_id": SOURCE_ID,
            "source_title": SOURCE_TITLE,
            "chapter": chapter,
            "section": section,
            "page": page_number,
            "content_type": "guideline_content",
            "text": chunk_text
        })

        chunk_number += 1

    return chunks, chunk_number


# ==========================================
# 9. MAIN PROCESS
# ==========================================

def main():

    print("Opening PDF...")

    document = fitz.open(PDF_PATH)

    print("PDF opened successfully.")
    print(f"Total pages: {len(document)}")

    all_chunks = []

    current_chapter = None
    current_section = None

    chunk_counter = 1

    # ======================================
    # READ EVERY PAGE
    # ======================================

    for page_index in range(len(document)):

        page_number = page_index + 1

        page = document[page_index]

        raw_text = page.get_text()

        text = clean_text(raw_text)

        if not text:
            continue

        # ==================================
        # DETECT CHAPTER
        # ==================================

        detected_chapter = detect_chapter(text)

        if detected_chapter:

            current_chapter = detected_chapter

        # ==================================
        # DETECT SECTION
        # ==================================

        detected_section = detect_section(text)

        if detected_section:

            current_section = detected_section

        # ==================================
        # IGNORE PAGES BEFORE CHAPTER ONE
        # ==================================

        if current_chapter is None:
            continue

        # ==================================
        # ONLY PROCESS CHAPTERS 1-5
        # ==================================

        if current_chapter not in [
            "Chapter One",
            "Chapter Two",
            "Chapter Three",
            "Chapter Four",
            "Chapter Five"
        ]:

            continue

        # ==================================
        # SPLIT PAGE INTO PARAGRAPHS
        # ==================================

        paragraphs = split_into_paragraphs(text)

        if not paragraphs:
            continue

        # ==================================
        # CREATE CHUNKS
        # ==================================

        new_chunks, chunk_counter = (
            create_chunks_from_paragraphs(
                paragraphs,
                page_number,
                current_chapter,
                current_section,
                chunk_counter
            )
        )

        all_chunks.extend(new_chunks)

        print(
            f"Processed page {page_number} | "
            f"Chapter: {current_chapter} | "
            f"Section: {current_section}"
        )

    # ==========================================
    # SAVE JSON
    # ==========================================

    os.makedirs(
        "output",
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n================================")
    print("CHUNKING COMPLETE")
    print("================================")

    print(
        f"Total chunks created: {len(all_chunks)}"
    )

    print(
        f"Output file: {OUTPUT_PATH}"
    )


# ==========================================
# 10. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    main()