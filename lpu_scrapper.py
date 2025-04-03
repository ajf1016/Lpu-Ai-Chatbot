import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def get_webpage_content(url):
    """Fetch webpage content"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    return None


def extract_links(soup, base_url, file_type="pdf"):
    """Extract specific file type links from a webpage"""
    links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(f".{file_type}"):
            full_url = urljoin(base_url, href)
            links.append(full_url)
    return links


def download_file(url, folder="downloads"):
    """Download a file from a given URL"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = os.path.join(folder, url.split("/")[-1])
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")


def scrape_lpu_admission(url):
    """Main function to scrape admission guidelines and download PDFs"""
    content = get_webpage_content(url)
    if content:
        soup = BeautifulSoup(content, "html.parser")

        # Extract and print text content
        text_content = soup.get_text(" ", strip=True)
        print("Extracted Text Content:")
        print(text_content[:1000])  # Print first 1000 characters

        # Extract and download PDFs
        pdf_links = extract_links(soup, url, "pdf")
        for pdf in pdf_links:
            download_file(pdf)
    else:
        print("Failed to fetch webpage content.")


if __name__ == "__main__":
    lpu_admission_url = "https://www.lpu.in/admission/"  # Change if needed
    scrape_lpu_admission(lpu_admission_url)
