import os
import re
import unicodedata
import pdfplumber

def clean_text(text: str) -> str:
    """
    Applies strict text normalization for Legal NLP processing.
    """
    if not text:
        return ""
    
    # 1. Unicode Normalization: Resolves ligatures (e.g., 'ﬁ' -> 'fi') and standardizes encoding
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Fix end-of-line hyphenation (e.g., "Termi-\nnation" -> "Termination")
    text = re.sub(r'-\n+', '', text)
    
    # 3. Collapse all whitespaces, tabs, and newlines into a single space
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Remove any remaining non-printable or control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    return text.strip()

def parse_pdf(file_path: str) -> str:
    """
    Extracts text from PDF using pdfplumber's layout awareness.
    """
    full_text = []
    try:
        # pdfplumber respects visual column layouts much better than PyPDF2
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""
        
    # Join pages with a space to prevent arbitrary paragraph breaks
    raw_document = " ".join(full_text)
    return clean_text(raw_document)

def load_contracts(data_dir: str = "data/contracts") -> dict:
    """
    Scans the contracts directory and maps filename -> normalized string payload.
    """
    contracts = {}
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return contracts
        
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_dir, filename)
            contracts[filename] = parse_pdf(full_path)
    return contracts
