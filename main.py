import os
import gc
import torch
import time
import zipfile
import shutil
import urllib.request
import pandas as pd
from src.ingestion import load_contracts
from src.extraction import LLMExtractor

# Minimalist brutalist console logging colors
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

def setup_environment():
    """Ensure Hugging Face Token is pulled into the environment from Kaggle Secrets."""
    if "HF_TOKEN" not in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            print(f"{Colors.YELLOW}WARNING: HF_TOKEN not found in environment or Kaggle secrets.{Colors.RESET}")

def download_and_prepare_data(data_dir: str):
    """Downloads and extracts the CUAD dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if data already exists to avoid redundant heavy network requests
    if os.path.exists(data_dir) and len([f for f in os.listdir(data_dir) if f.endswith('.pdf')]) >= 50:
        print(f"      {Colors.GREEN}✓ Ground truth PDFs already present in '{data_dir}'. Skipping download.{Colors.RESET}")
        return True

    print(f"      {Colors.BLUE}-> Target data missing or incomplete. Downloading CUAD dataset from Zenodo...{Colors.RESET}")
    cuad_url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
    zip_path = "CUAD_v1.zip"

    try:
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(cuad_url, zip_path)

        print(f"      {Colors.BLUE}-> Extracting 50 core PDFs into '{data_dir}'...{Colors.RESET}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            pdf_files = [f for f in zip_ref.namelist() if f.endswith('.pdf') and 'full_contract_pdf' in f]
            
            for pdf_path in pdf_files[:50]:
                zip_ref.extract(pdf_path, "temp_extract")
                filename = os.path.basename(pdf_path)
                shutil.move(os.path.join("temp_extract", pdf_path), os.path.join(data_dir, filename))

        # Cleanup artifacts
        shutil.rmtree("temp_extract", ignore_errors=True)
        os.remove(zip_path)
        
        # Verify execution integrity
        downloaded_count = len([f for f in os.listdir(data_dir) if f.endswith('.pdf')])
        if downloaded_count >= 50:
            print(f"      {Colors.GREEN}✓ Success! {downloaded_count} PDFs safely stored.{Colors.RESET}")
            return True
        else:
            print(f"      {Colors.RED}❌ Extraction verification failed. Only found {downloaded_count} files.{Colors.RESET}")
            return False
            
    except Exception as e:
        print(f"      {Colors.RED}❌ Data Ingestion Error: {e}{Colors.RESET}")
        return False

def main():
    setup_environment()
    
    DATA_DIR = "data/contracts"
    OUTPUT_FILE = "extraction_results.csv"
    
    print(f"\n{Colors.BLUE}=================================================={Colors.RESET}")
    print(f"{Colors.BLUE}    LEGAL AI EXTRACTION PIPELINE (MAP-AGGREGATE)   {Colors.RESET}")
    print(f"{Colors.BLUE}=================================================={Colors.RESET}\n")
    
    print(f"{Colors.YELLOW}[1/4] Checking & Preparing Dataset...{Colors.RESET}")
    if not download_and_prepare_data(DATA_DIR):
        print(f"{Colors.RED}Pipeline aborted due to data setup failure.{Colors.RESET}")
        return
        
    print(f"{Colors.YELLOW}[2/4] Loading and Normalizing PDFs...{Colors.RESET}")
    contracts = load_contracts(DATA_DIR)
    print(f"      Loaded {len(contracts)} contracts into memory.\n")
    
    print(f"{Colors.YELLOW}[3/4] Initializing Quantized Llama 3.1 8B Engine...{Colors.RESET}")
    extractor = LLMExtractor()
    print("      Model and Tokenizer loaded successfully.\n")
    
    results = []
    
    print(f"{Colors.YELLOW}[4/4] Processing Contracts via Overlapping Windows...{Colors.RESET}")
    start_time = time.time()
    
    for idx, (filename, text) in enumerate(contracts.items(), 1):
        print(f"{Colors.BLUE}      -> Slicing & Extracting ({idx}/{len(contracts)}): {filename}{Colors.RESET}")
        
        # OOM Fix: Context management to free up graph execution memory
        with torch.no_grad():
            record = extractor.process_contract(filename, text)
            
        results.append(record.model_dump())
        
        # OOM Fix: Hard clearing VRAM fragmentations after every contract pass
        del record
        gc.collect()
        torch.cuda.empty_cache()
        
    elapsed = time.time() - start_time
    print(f"\n      All extractions completed in {elapsed:.2f} seconds.\n")
    
    print(f"{Colors.YELLOW}Serializing Results to CSV Schema...{Colors.RESET}")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"{Colors.GREEN}✅ SUCCESS: Pipeline run complete. File saved to: {OUTPUT_FILE}{Colors.RESET}\n")

if __name__ == "__main__":
    main()