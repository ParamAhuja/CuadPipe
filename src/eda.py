import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from .ingestion import load_contracts

def run_data_analysis(model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", output_dir: str = "plots"):
    """Performs deep data analysis on parsed contracts, computing character, word, and exact Llama-3.1 token counts."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Loading normalized text data...")
    contracts = load_contracts()
    
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
    
    plt.style.use('dark_background')
    
    # Bypass plt.axes() and explicitly declare the figure and axes
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    
    for spine in ax.spines.values():
        spine.set_color("#444444")
    
    n, bins, patches = ax.hist(df["Tokens"], bins=15, color="#007acc", edgecolor="#111111", alpha=0.85)
    
    ax.axvline(x=8000, color="#ff4500", linestyle="--", linewidth=2, label="Kaggle Safe Threshold (8k tokens)")
    
    ax.set_title("Contract Token Count Distribution", fontsize=14, fontweight="bold", pad=15, color="#ffffff")
    ax.set_xlabel("Exact Llama 3.1 Token Count", fontsize=12, color="#cccccc")
    ax.set_ylabel("Number of Contracts", fontsize=12, color="#cccccc")
    
    ax.legend(loc="upper right", frameon=True, facecolor="#2d2d2d", edgecolor="#444444", labelcolor="#cccccc")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "token_distribution.png")
    fig.savefig(plot_path, dpi=300, facecolor="#1e1e1e")
    plt.close(fig)
    
    print(f"✅ Analysis complete! Visualization chart saved to: {plot_path}")
    return df
