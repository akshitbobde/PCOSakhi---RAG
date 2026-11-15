"""Create page-aware chunks from the 2023 international PCOS guideline.

Run from the project root:
    py scripts/create_chunks.py

Input:  data/Evidence-Based-Guidelines-2023.pdf
Output: Output/chunks.json

The page field is the 1-based physical PDF page number.
"""
import json
import re
from pathlib import Path
import fitz

PDF_PATH = Path("data") / "Evidence-Based-Guidelines-2023.pdf"
OUTPUT_PATH = Path("output") / "chunks.json"
SOURCE_ID = "PCOS_001"
SOURCE_TITLE = "International Evidence-based Guideline for the assessment and management of polycystic ovary syndrome 2023"
TARGET_WORDS = 350
OVERLAP_WORDS = 50

# Verified 1-based physical PDF page ranges. Chapter Six (page 174 onward)
# is deliberately excluded because it describes guideline-development methods.
PAGE_RANGES = [
    (27, 56, "Recommendations Summary", "recommendation"),
    (57, 83, "Chapter One", "guideline_content"),
    (84, 100, "Chapter Two", "guideline_content"),
    (101, 117, "Chapter Three", "guideline_content"),
    (118, 144, "Chapter Four", "guideline_content"),
    (145, 173, "Chapter Five", "guideline_content"),
]

# Only a two-level section heading (e.g. 1.1 or 4.2) can update section metadata.
# Recommendation IDs such as 4.2.7 are intentionally not accepted.
SECTION_LINE = re.compile(r"^([1-5]\.\d{1,2})\s+([A-Z][^\n]{2,140})$")
SECTION_NUMBER_ONLY = re.compile(r"^([1-5]\.\d{1,2})$")


def chapter_for_page(page_number):
    for start, end, chapter, content_type in PAGE_RANGES:
        if start <= page_number <= end:
            return chapter, content_type
    return None, None


def clean_lines(raw_text):
    lines = []
    for raw in raw_text.replace("\xa0", " ").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            lines.append("")
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if re.fullmatch(r"[©\s]*Monash University[.,\s]*", line, re.I):
            continue
        if re.search(r"International Evidence-based Guideline for the", line, re.I):
            continue
        if re.search(r"assessment and management of polycystic ovary syndrome(?:\s+2023)?", line, re.I) and len(line) < 150:
            continue
        lines.append(line)
    # Join headings whose number and title are extracted on separate lines,
    # and create paragraph boundaries around two-level headings. This helps
    # section metadata survive PDFs that do not preserve visual spacing.
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if SECTION_NUMBER_ONLY.match(line):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip():
                line = f"{line} {lines[j].strip()}"
                i = j
        is_section_heading = bool(SECTION_LINE.match(line))
        if is_section_heading and merged and merged[-1].strip():
            merged.append("")
        merged.append(line)
        if is_section_heading:
            merged.append("")
        i += 1
    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text):
    paras = []
    for p in re.split(r"\n\s*\n", text):
        p = re.sub(r"\s+", " ", p).strip()
        if p and not re.fullmatch(r"[\W_]+", p):
            paras.append(p)
    return paras


def section_from_paragraph(paragraph, current_section, chapter):
    if chapter == "Recommendations Summary":
        return None
    p = paragraph.strip()
    match = SECTION_LINE.match(p)
    if match:
        title = match.group(2).strip()
        # Avoid accepting a full sentence or recommendation classification as a heading.
        if len(title.split()) <= 18 and not title.endswith((".", ";", ",", "?", "!")):
            return f"{match.group(1)} {title}"
    return current_section


def split_long_paragraph(paragraph):
    words = paragraph.split()
    if len(words) <= TARGET_WORDS:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“‘(])", paragraph)
    units = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sw = sentence.split()
        if len(sw) <= TARGET_WORDS:
            units.append(sentence)
        else:
            units.extend(" ".join(sw[i:i + TARGET_WORDS]) for i in range(0, len(sw), TARGET_WORDS))
    return units


def build_chunks(doc):
    chunks = []
    section_by_chapter = {f"Chapter {name}": None for name in ["One", "Two", "Three", "Four", "Five"]}
    sequence_by_page = {}

    for page_number in range(27, 174):
        chapter, content_type = chapter_for_page(page_number)
        if not chapter:
            continue
        text = clean_lines(doc[page_number - 1].get_text("text"))
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            continue

        if chapter == "Recommendations Summary":
            current_section = None
        else:
            current_section = section_by_chapter.get(chapter)

        # Track headings in reading order. Do not infer the chapter from recommendation IDs.
        tagged_paragraphs = []
        for paragraph in paragraphs:
            updated = section_from_paragraph(paragraph, current_section, chapter)
            if updated != current_section:
                current_section = updated
            tagged_paragraphs.append((paragraph, current_section))
        if chapter != "Recommendations Summary":
            section_by_chapter[chapter] = current_section

        # Pack paragraph units into chunks, keeping each chunk's metadata from its first unit.
        units = []
        for paragraph, section in tagged_paragraphs:
            for unit in split_long_paragraph(paragraph):
                units.append((unit, section))

        page_chunks = []
        current_parts = []
        current_words = 0
        current_section_for_chunk = None
        for unit, section in units:
            n = len(unit.split())
            if current_parts and (current_words + n > TARGET_WORDS or section != current_section_for_chunk):
                page_chunks.append(("\n\n".join(current_parts), current_section_for_chunk))
                current_parts = []
                current_words = 0
            if not current_parts:
                current_section_for_chunk = section
            # Oversized units are already split, but this guard keeps the contract safe.
            if n > TARGET_WORDS:
                unit = " ".join(unit.split()[:TARGET_WORDS])
                n = len(unit.split())
            current_parts.append(unit)
            current_words += n
        if current_parts:
            page_chunks.append(("\n\n".join(current_parts), current_section_for_chunk))

        for idx, (chunk_text, section) in enumerate(page_chunks, start=1):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            words = chunk_text.split()
            # Add overlap from the previous chunk on the same page only when useful.
            if idx > 1 and OVERLAP_WORDS:
                prev_words = page_chunks[idx - 2][0].split()
                overlap = prev_words[-OVERLAP_WORDS:]
                if overlap and len(words) + len(overlap) <= TARGET_WORDS:
                    words = overlap + words
                    chunk_text = " ".join(words)
            sequence_by_page[page_number] = sequence_by_page.get(page_number, 0) + 1
            chunks.append({
                "chunk_id": f"{SOURCE_ID}_P{page_number}_{sequence_by_page[page_number]:04d}",
                "source_id": SOURCE_ID,
                "source_title": SOURCE_TITLE,
                "chapter": chapter,
                "section": section,
                "page": page_number,
                "content_type": content_type,
                "word_count": len(chunk_text.split()),
                "text": chunk_text,
            })
    return chunks


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at {PDF_PATH}. Put Evidence-Based-Guidelines-2023.pdf in the data folder."
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(PDF_PATH) as doc:
        chunks = build_chunks(doc)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    ids = [c["chunk_id"] for c in chunks]
    chapter_counts = {}
    for c in chunks:
        chapter_counts[c["chapter"]] = chapter_counts.get(c["chapter"], 0) + 1
    print(f"Created {len(chunks)} chunks: {OUTPUT_PATH}")
    print(f"Unique IDs: {len(set(ids)) == len(ids)}")
    print(f"Maximum word count: {max(c['word_count'] for c in chunks)}")
    print(f"Word-count mismatches: {sum(c['word_count'] != len(c['text'].split()) for c in chunks)}")
    print("Chapter distribution:")
    for chapter, count in chapter_counts.items():
        print(f"  {chapter}: {count}")
    print("Reminder: inspect several chunks and section labels before generating embeddings.")

if __name__ == "__main__":
    main()
