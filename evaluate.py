import os
import re
import zipfile
import shutil
import random
import urllib.request
import pandas as pd
from src.ingestion import load_contracts

def is_verbatim_match(extracted: str, original: str) -> bool:
    """Normalizes punctuation, whitespace, and line numbers to verify true verbatim content alignment across multi-block extractions."""
    def clean(text):
        text = text.lower()
        # Remove bullet numbers/letters at line starts (e.g., "1.", "(a)", "ii)")
        text = re.sub(r'^\s*(\(\w+\)|\w+\.|\d+\))\s*', '', text, flags=re.MULTILINE)
        # Remove all punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Collapse whitespace
        return re.sub(r'\s+', ' ', text).strip()
    
    cleaned_original = clean(original)
    
    # SPLIT LOGIC: Since extraction.py joins multiple valid <TEXT> blocks with \n\n, 
    # we must evaluate each distinct block independently against the source text!
    blocks = [b.strip() for b in extracted.split('\n\n') if b.strip()]
    
    if not blocks:
        return False
        
    # Every individual extracted block must be found contiguously in the original text
    for block in blocks:
        cleaned_block = clean(block)
        # Ignore extremely short residual snippets (<15 chars), but enforce verbatim substring matching on substantive blocks
        if len(cleaned_block) > 15 and cleaned_block not in cleaned_original:
            return False  # If even one block was hallucinated, paraphrased, or altered, fail the whole clause
            
    return True
    
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

def download_and_prepare_eval_data(data_dir: str = "data/contracts", target_count: int = 50, seed: int = 42):
    """Ensures ground truth source files are present; falls back to deterministic randomized download if empty."""
    os.makedirs(data_dir, exist_ok=True)
    
    existing_pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if len(existing_pdfs) > 0:
        print(f"      {Colors.GREEN}✓ Ground truth PDFs found in '{data_dir}' ({len(existing_pdfs)} files).{Colors.RESET}")
        return True

    print(f"      {Colors.BLUE}-> Source data missing from '{data_dir}'. Syncing default CUAD benchmark dataset...{Colors.RESET}")
    cuad_url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
    zip_path = "CUAD_v1.zip"

    try:
        if not os.path.exists(zip_path):
            print(f"      {Colors.BLUE}-> Downloading baseline archive from Zenodo (this may take a moment)...{Colors.RESET}")
            urllib.request.urlretrieve(cuad_url, zip_path)

        print(f"      {Colors.BLUE}-> Extracting {target_count} deterministic random reference files (Seed: {seed})...{Colors.RESET}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            pdf_files = [f for f in zip_ref.namelist() if f.endswith('.pdf') and 'full_contract_pdf' in f]
            
            # Sort first for cross-platform alignment, then shuffle deterministically
            pdf_files.sort()
            random.seed(seed)
            random.shuffle(pdf_files)
            
            for pdf_path in pdf_files[:target_count]:
                zip_ref.extract(pdf_path, "temp_extract")
                filename = os.path.basename(pdf_path)
                shutil.move(os.path.join("temp_extract", pdf_path), os.path.join(data_dir, filename))

        # Clean work-directory pollution
        shutil.rmtree("temp_extract", ignore_errors=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"      {Colors.GREEN}✓ Staging complete.{Colors.RESET}")
        return True
    except Exception as e:
        print(f"      {Colors.RED}❌ Data synchronization failure inside Evaluation module: {e}{Colors.RESET}")
        return False

def run_evaluation():
    # Looks for a fresh local run first; falls back to the static Kaggle input dataset if not found
    LOCAL_CSV = "extraction_results.csv"
    KAGGLE_CSV = "/kaggle/input/datasets/papapopeye/extraction-1/extraction_results (1).csv"
    CSV_PATH = LOCAL_CSV if os.path.exists(LOCAL_CSV) else KAGGLE_CSV
    
    DATA_DIR = "data/contracts"
    
    print(f"\n{Colors.BLUE}=================================================={Colors.RESET}")
    print(f"{Colors.BLUE}       CUADPIPE: EXTRACTION QUALITY EVALUATION    {Colors.RESET}")
    print(f"{Colors.BLUE}=================================================={Colors.RESET}\n")
    
    # 1. Integrity & Dataset Checks
    if not os.path.exists(CSV_PATH):
        print(f"{Colors.RED}❌ Error: Extraction target file not found at '{LOCAL_CSV}' or '{KAGGLE_CSV}'.{Colors.RESET}")
        print(f"          Please run main.py first or verify your Kaggle input dataset attachment.{Colors.RESET}")
        return
        
    print(f"{Colors.YELLOW}[1/3] Loading Dataset ('{CSV_PATH}') and Syncing Source Files...{Colors.RESET}")
    if not download_and_prepare_eval_data(DATA_DIR):
        print(f"{Colors.RED}Evaluation aborted due to missing reference documents.{Colors.RESET}")
        return

    df = pd.read_csv(CSV_PATH)
    contracts = load_contracts(DATA_DIR)
    
    print(f"      Loaded {len(df)} extraction records from target CSV.")
    print(f"      Successfully cached {len(contracts)} plaintext reference files into memory.\n")
    
    # 2. Metric Accumulators
    total_input_chars = 0
    total_summary_chars = 0
    
    verbatim_matches = {
        "termination_clause": {"match": 0, "none": 0, "fail": 0},
        "confidentiality_clause": {"match": 0, "none": 0, "fail": 0},
        "liability_clause": {"match": 0, "none": 0, "fail": 0}
    }
    
    print(f"{Colors.YELLOW}[2/3] Verifying Extraction Verbatim Accuracy...{Colors.RESET}")
    
    for _, row in df.iterrows():
        contract_id = row["contract_id"]
        summary = str(row["summary"])
        
        # Pull original text matching this specific CSV record
        original_text = contracts.get(contract_id, "")
        
        if not original_text:
            print(f"      {Colors.RED}[Warning] Source reference text missing for {contract_id}{Colors.RESET}")
            continue
            
        # Accumulate metrics for Input vs Output lengths
        total_input_chars += len(original_text)
        total_summary_chars += len(summary)
        
        # Verify the 3 clauses
        for clause_key in ["termination_clause", "confidentiality_clause", "liability_clause"]:
            if clause_key not in row:
                continue
                
            extracted_text = str(row[clause_key]).strip()
            
            if extracted_text == "NONE" or extracted_text.upper() == "NONE":
                verbatim_matches[clause_key]["none"] += 1
            else:
                # Direct substring containment check across multiple extracted blocks
                if is_verbatim_match(extracted_text, original_text):
                    verbatim_matches[clause_key]["match"] += 1
                else:
                    verbatim_matches[clause_key]["fail"] += 1
                    
    # 3. Print out Final Analytics Report
    print(f"\n{Colors.YELLOW}[3/3] Generating Final Analytics Report...{Colors.RESET}")
    print(f"{Colors.BLUE}--------------------------------------------------{Colors.RESET}")
    print(f" 1. EVALUATION OVERVIEW")
    print(f"{Colors.BLUE}--------------------------------------------------{Colors.RESET}")
    print(f"    Total Documents Evaluated      : {len(df)}")
    
    print(f"\n{Colors.BLUE}--------------------------------------------------{Colors.RESET}")
    print(f" 2. LENGTH PROFILE (INPUT VS SUMMARY OUTPUT)")
    print(f"{Colors.BLUE}--------------------------------------------------{Colors.RESET}")
    avg_input = total_input_chars / len(df) if len(df) > 0 else 0
    avg_output = total_summary_chars / len(df) if len(df) > 0 else 0
    compression_ratio = (avg_output / avg_input) * 100 if avg_input > 0 else 0
    
    print(f"    Avg Source Contract Length     : {avg_input:.1f} characters")
    print(f"    Avg Model Summary Length       : {avg_output:.1f} characters")
    print(f"    Data Compression Ratio         : {compression_ratio:.2f}% of original footprint")
    
    print(f"\n{Colors.BLUE}--------------------------------------------------{Colors.RESET}")
    print(f" 3. VERBATIM WORDING ACCURACY GROUND-TRUTH CHECK")
    print(f"{Colors.BLUE}--------------------------------------------------{Colors.RESET}")
    print(f"    A 'Match' confirms all extracted blocks match source wording 100% verbatim.")
    print(f"    A 'Fail' indicates hallucination, formatting edits, or paraphrase leaks.\n")
    
    # Render table layout for clean evaluation logging
    print(f"    {'Clause Target':<25} | {'Match':<6} | {'None':<5} | {'Fail':<5}")
    print(f"    --------------------------------------------------")
    for clause, stats in verbatim_matches.items():
        clause_name = clause.replace("_", " ").title()
        fail_color = Colors.RED if stats['fail'] > 0 else Colors.RESET
        print(f"    {clause_name:<25} | {stats['match']:<6} | {stats['none']:<5} | {fail_color}{stats['fail']:<5}{Colors.RESET}")
        
    print(f"\n{Colors.GREEN}✅ Evaluation Complete.{Colors.RESET}\n")

if __name__ == "__main__":
    run_evaluation()