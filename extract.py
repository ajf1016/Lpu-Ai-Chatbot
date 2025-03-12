import os
import pdfplumber
import fitz  # PyMuPDF
import PyPDF2

# Directory containing PDFs
pdf_folder = "path/to/your/pdf/folder"

# Output file to store extracted text
output_file = "extracted_texts.txt"

# Function to extract text using pdfplumber


def extract_text_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
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
            text += page.extract_text() + "\n"
    return text


# Extract text from all PDFs in the folder
with open(output_file, "w", encoding="utf-8") as out_file:
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Extracting text from: {filename}")

            # Choose the best method for your PDFs
            extracted_text = extract_text_pdfplumber(
                pdf_path)  # Try pdfplumber
            # extracted_text = extract_text_pymupdf(pdf_path)   # Try PyMuPDF (fitz)
            # extracted_text = extract_text_pypdf2(pdf_path)    # Try PyPDF2

            out_file.write(f"--- Extracted Text from {filename} ---\n")
            out_file.write(extracted_text)
            out_file.write("\n\n")

print(f"Extraction completed! Check {output_file} for results.")
