import os
import pdfplumber
import fitz  # PyMuPDF
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Directory containing PDFs
pdf_folder = "./lpu_pdfs/"
output_file = "extracted_texts.txt"

# Function to extract text using pdfplumber


def extract_text_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to extract text using PyMuPDF (fitz)


def extract_text_pymupdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text() + "\n"
    return text

# Function to extract text using PyPDF2


def extract_text_pypdf2(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to extract text from images using OCR


def extract_text_ocr(pdf_path):
    text = ""
    images = convert_from_path(pdf_path)  # Convert PDF pages to images
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"  # OCR on each image
    return text


# Extract text from all PDFs in the folder
with open(output_file, "w", encoding="utf-8") as out_file:
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Extracting text from: {filename}")

            # Try normal extraction methods first
            extracted_text = extract_text_pdfplumber(pdf_path) or \
                extract_text_pymupdf(pdf_path) or \
                extract_text_pypdf2(pdf_path)

            # If no text is found, apply OCR
            if not extracted_text.strip():
                print(f"Applying OCR for {filename} (contains images)")
                extracted_text = extract_text_ocr(pdf_path)

            out_file.write(f"--- Extracted Text from {filename} ---\n")
            out_file.write(extracted_text)
            out_file.write("\n\n")

print(f"Extraction completed! Check {output_file} for results.")
