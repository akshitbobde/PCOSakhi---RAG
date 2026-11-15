import fitz

pdf_path = "Data/Evidence-Based-Guidelines-2023.pdf"

document = fitz.open(pdf_path)

print("PDF opened successfully!")
print("Number of pages:", len(document))

for page_number in range(min(3, len(document))):
    page = document[page_number]

    text = page.get_text()

    print("\n==============================")
    print("PAGE:", page_number + 1)
    print("==============================")

    print(text[:2000])