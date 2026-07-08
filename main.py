import os
import pandas as pd
import time
from src.ingestion import load_contracts
from src.extraction import LLMExtractor

# Minimalist brutalist console logging colors
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def setup_environment():
    """Ensure Hugging Face Token is pulled into the environment from Kaggle Secrets."""
    if "HF_TOKEN" not in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            print("WARNING: HF_TOKEN not found in environment or Kaggle secrets.")

def main():
    setup_environment()
    
    DATA_DIR = "data/contracts"
    OUTPUT_FILE = "extraction_results.csv"
    
    print(f"\n{Colors.BLUE}=================================================={Colors.RESET}")
    print(f"{Colors.BLUE}    LEGAL AI EXTRACTION PIPELINE (MAP-AGGREGATE)   {Colors.RESET}")
    print(f"{Colors.BLUE}=================================================={Colors.RESET}\n")
    
    print(f"{Colors.YELLOW}[1/4] Loading and Normalizing PDFs...{Colors.RESET}")
    contracts = load_contracts(DATA_DIR)
    
    if not contracts:
        print("❌ Error: No contracts found in data/contracts/. Aborting pipeline.")
        return
        
    print(f"      Successfully loaded {len(contracts)} contracts into memory.\n")
    
    print(f"{Colors.YELLOW}[2/4] Initializing Quantized Llama 3.1 8B Engine...{Colors.RESET}")
    extractor = LLMExtractor()
    print("      Model and Tokenizer loaded successfully.\n")
    
    results = []
    
    print(f"{Colors.YELLOW}[3/4] Processing Contracts via Overlapping Windows...{Colors.RESET}")
    start_time = time.time()
    
    for idx, (filename, text) in enumerate(contracts.items(), 1):
        print(f"{Colors.BLUE}      -> Slicing & Extracting ({idx}/{len(contracts)}): {filename}{Colors.RESET}")
        
        # This processes the file through the overlapping chunk strategy we built
        record = extractor.process_contract(filename, text)
        results.append(record.model_dump())
        
    elapsed = time.time() - start_time
    print(f"\n      All extractions completed in {elapsed:.2f} seconds.\n")
    
    print(f"{Colors.YELLOW}[4/4] Serializing Results to CSV Schema...{Colors.RESET}")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"{Colors.GREEN}✅ SUCCESS: Pipeline run complete. File saved to: {OUTPUT_FILE}{Colors.RESET}\n")

if __name__ == "__main__":
    main()
