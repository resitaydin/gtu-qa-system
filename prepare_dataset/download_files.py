import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os

def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def download_pdfs(url, download_dir="data"):
    """Downloads all PDFs from a given URL.

    Args:
        url (str): URL of the page containing links to PDFs
        download_dir (str, optional): Directory to save the PDF. Defaults to "downloads".
    """

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    pdf_links = []

    for a_tag in soup.find_all('a', href=True):
        link = a_tag['href']

        if link.lower().endswith('.pdf'):
            if not is_valid(link):
                # Make the URL absolute.
                link = urljoin(url, link)
                
            if is_valid(link):
               pdf_links.append(link)

    if not pdf_links:
        print(f"No PDF links found on {url}")
        return

    print(f"Found {len(pdf_links)} PDF links on {url}")

    os.makedirs(download_dir, exist_ok=True)  # Create directory if it does not exist

    for index, pdf_url in enumerate(pdf_links):
        try:
            print(f"Starting download: {index + 1} - {pdf_url}")
            pdf_response = requests.get(pdf_url, stream=True)
            pdf_response.raise_for_status()
            
            file_name = os.path.basename(urlparse(pdf_url).path)
            file_path = os.path.join(download_dir, file_name)


            with open(file_path, 'wb') as pdf_file:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    pdf_file.write(chunk)

            print(f"Downloaded: {index + 1} - {file_name}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {pdf_url}: {e}")
        
        time.sleep(1) # Add a delay of 1 second

if __name__ == "__main__":
    page_url = "https://www.gtu.edu.tr/kategori/3074/0/display.aspx?languageId=1"
    download_pdfs(page_url)
    print("Done.")