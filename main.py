import os
import gc
import torch
import time
import zipfile
import shutil
import random
import urllib.request
import pandas as pd
from src.ingestion import load_contracts
from src.extraction import LLMExtractor

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
            pass

def check_hardware():
    """Validates if a CUDA-capable GPU is available for quantized inference."""
    print(f"{Colors.YELLOW}[1/5] Validating Hardware Environment...{Colors.RESET}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"      {Colors.GREEN}✓ GPU Acceleration Detected: {device_name}{Colors.RESET}\n")
        return True
    else:
        print(f"      {Colors.RED}❌ CRITICAL ERROR: No CUDA-capable GPU detected.{Colors.RESET}")
        print(f"        4-bit NF4 quantization requires a GPU accelerator (e.g., Kaggle T4).")
        print(f"        Pipeline execution halted to prevent CPU memory thrashing.\n")
        return False

def download_and_prepare_data(data_dir: str, target_count: int = 50, seed: int = 42):
    """Uses existing PDFs if present; falls back to downloading a deterministic CUAD subset if empty."""
    os.makedirs(data_dir, exist_ok=True)
    
    existing_pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if len(existing_pdfs) > 0:
        print(f"      {Colors.GREEN}✓ Found {len(existing_pdfs)} existing PDF(s) in '{data_dir}'. Bypassing download.{Colors.RESET}")
        return True

    print(f"      {Colors.BLUE}-> No local PDFs found. Fetching fallback CUAD dataset from Zenodo...{Colors.RESET}")
    cuad_url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
    zip_path = "CUAD_v1.zip"

    try:
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(cuad_url, zip_path)

        print(f"      {Colors.BLUE}-> Extracting {target_count} deterministic random samples (Seed: {seed})...{Colors.RESET}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            pdf_files = [f for f in zip_ref.namelist() if f.endswith('.pdf') and 'full_contract_pdf' in f]
            
            pdf_files.sort()
            random.seed(seed)
            random.shuffle(pdf_files)
            
            for pdf_path in pdf_files[:target_count]:
                zip_ref.extract(pdf_path, "temp_extract")
                filename = os.path.basename(pdf_path)
                shutil.move(os.path.join("temp_extract", pdf_path), os.path.join(data_dir, filename))

        shutil.rmtree("temp_extract", ignore_errors=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"      {Colors.GREEN}✓ Successfully staged default CUAD subset.{Colors.RESET}")
        return True
    except Exception as e:
        print(f"      {Colors.RED}❌ Data Ingestion Error: {e}{Colors.RESET}")
        return False

def main():
    setup_environment()
    
    DATA_DIR = "data/contracts"
    OUTPUT_FILE = "extraction_results.csv"
    
    print(f"\n{Colors.BLUE}=================================================={Colors.RESET}")
    print(f"{Colors.BLUE}    CUADPIPE: DOCUMENT AI EXTRACTION ENGINE       {Colors.RESET}")
    print(f"{Colors.BLUE}=================================================={Colors.RESET}\n")
    
    # 1. Hardware check
    if not check_hardware():
        return
        
    # 2. Data Check / Fallback Download
    print(f"{Colors.YELLOW}[2/5] Staging Contract Datasets...{Colors.RESET}")
    if not download_and_prepare_data(DATA_DIR):
        print(f"{Colors.RED}Pipeline execution aborted due to ingestion failure.{Colors.RESET}")
        return
    print("")
        
    # 3. Text Ingestion & Layout Normalization
    print(f"{Colors.YELLOW}[3/5] Executing Layout-Aware Normalization Pipeline...{Colors.RESET}")
    contracts = load_contracts(DATA_DIR)
    print(f"      Successfully normalized {len(contracts)} documents into plaintext cache.\n")
    
    # 4. Model Booting
    print(f"{Colors.YELLOW}[4/5] Loading Quantized Llama 3.1 8B Inference Engine...{Colors.RESET}")
    extractor = LLMExtractor()
    print(f"      {Colors.GREEN}✓ Local LLM engine initialized.{Colors.RESET}\n")
    
    results = []
    
    # 5. Pipeline Map-Aggregate Loop
    print(f"{Colors.YELLOW}[5/5] Processing Contract Extraction Stream...{Colors.RESET}")
    start_time = time.time()
    
    for idx, (filename, text) in enumerate(contracts.items(), 1):
        print(f"{Colors.BLUE}      -> [{idx}/{len(contracts)}] Segmenting & Extracting: {filename}{Colors.RESET}")
        
        with torch.no_grad():
            record = extractor.process_contract(filename, text)
            
        results.append(record.model_dump())
        
        # Micro-memory clearance loop
        del record
        gc.collect()
        torch.cuda.empty_cache()
        
    elapsed = time.time() - start_time
    print(f"\n{Colors.GREEN}✓ Core extraction stream completed in {elapsed:.2f} seconds.{Colors.RESET}\n")
    
    print(f"{Colors.YELLOW}Serializing Structured Results to Disks...{Colors.RESET}")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"{Colors.GREEN}✅ SUCCESS: Pipeline run complete. Structured file saved to: {OUTPUT_FILE}{Colors.RESET}\n")

if __name__ == "__main__":
    main()