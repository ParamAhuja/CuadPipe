import os
import random
import zipfile
import shutil
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from .ingestion import load_contracts

def download_and_prepare_eda_data(data_dir: str = "data/contracts", target_count: int = 50, seed: int = 42):
    """Uses existing PDFs if present; falls back to downloading a deterministic CUAD subset if empty."""
    os.makedirs(data_dir, exist_ok=True)
    
    existing_pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if len(existing_pdfs) > 0:
        print(f"      ✓ Found {len(existing_pdfs)} existing PDF(s) in '{data_dir}'. Using local files.")
        return True

    print(f"      -> No PDFs found in '{data_dir}'. Fetching default CUAD dataset from Zenodo...")
    cuad_url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
    zip_path = "CUAD_v1.zip"

    try:
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(cuad_url, zip_path)

        print(f"      -> Extracting {target_count} deterministic random PDFs (Seed: {seed})...")
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
        return True
    except Exception as e:
        print(f"      ❌ Data initialization failure inside EDA module: {e}")
        return False

def run_data_analysis(model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", output_dir: str = "plots"):
    """Performs data analysis on parsed contracts, computing character, word, and exact Llama-3.1 token counts."""
    DATA_DIR = "data/contracts"
    
    if not download_and_prepare_eda_data(DATA_DIR):
        print("Analysis aborted due to data generation failure.")
        return None
        
    print("1. Loading normalized text data...")
    contracts = load_contracts(DATA_DIR)
    
    if not contracts:
        print("No contracts found to analyze.")
        return None
        
    print("2. Downloading Llama 3.1 Tokenizer for exact token counting...")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    print("3. Analyzing text dimensions...")
    analysis_data = []
    
    for filename, text in contracts.items():
        char_count = len(text)
        word_count = len(text.split())
        token_count = len(tokenizer.encode(text))
        
        analysis_data.append({
            "Contract ID": filename,
            "Characters": char_count,
            "Words": word_count,
            "Tokens": token_count
        })
        
    df = pd.DataFrame(analysis_data)
    
    print("\n" + "="*40)
    print("       DATASET DESCRIPTIVE METRICS       ")
    print("="*40)
    print(df[["Characters", "Words", "Tokens"]].describe().round(1))
    print("="*40 + "\n")
    
    print(f"4. Generating visualization graphs and saving to '{output_dir}/'...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["Tokens"], bins=15, color="#007acc", edgecolor="#111111", alpha=0.85)
    ax.axvline(x=8000, color="#ff4500", linestyle="--", linewidth=2, label="Kaggle Safe Threshold (8k tokens)")
    
    ax.set_title("Contract Token Count Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Exact Llama 3.1 Token Count", fontsize=12)
    ax.set_ylabel("Number of Contracts", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "token_distribution.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    
    print(f"✅ Analysis complete! Visualization chart saved to: {plot_path}")
    return df