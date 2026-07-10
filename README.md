contact: `paramahuja04@gmail.com`

# cuadpipe: A VRAM-Optimized Legal Document Extraction Engine

A local-first document AI pipeline that extracts verbatim legal clauses and generate summary for huge documents from complex corporate agreements. Built on top of the [Atticus Project (CUAD)](https://github.com/TheAtticusProject/cuad) benchmark dataset, `cuadpipe` leverages 4-bit quantized **Meta-Llama-3.1-8B-Instruct** with an exhaustive sliding-window Map-Aggregate chunking architecture to achieve maximum recall even on low compute power.

## 1. Quick Summary:

### Problem Understanding: 
Contract review is a task about "finding needles in a haystack.", requiring lawyers to spend hours or sometimes days reading through legal documents for scarce key clauses that matters for clients, buried among details often spanning large number of pages. Which is also subject to human error!

For our use case including 3 key clauses "termination_clause", "confidentiality_clause" and "liability_clause", it was found that only **about 16.28%** of the text was actually relevant. **That's only 16 pages of useful info in a 100 page pdf!**

Extracting exact, verbatim legal clauses from long-form commercial agreements requires a fundamental understanding: **probabilistic language models want to generate fluent prose, but legal compliance requires strict, deterministic copy-pasting.**

### My solution: 
`cuadpipe` resolves this by using **Exhaustive Targeted-Pass Map-Aggregate Architecture**. By breaking document analysis into isolated, chunks of text using overlapping context windows, the engine achieves:

* **100% Document Coverage:** Eliminates retrieval blindspots common in standard RAG pipelines.
* **Verbatim Fidelity:** Strips conversational formatting artifacts to guarantee character-for-character legal accuracy.
* **Zero API Dependency:** Runs entirely locally on consumer GPUs (e.g., 2x NVIDIA T4 on kaggle was used here) via 4-bit NormalFloat quantization.
* **Deterministic Reproducibility:** Features built-in seeded dataset bootstrapping (`seed=42`) for robust cross-environment benchmarking and batch execution.

## 2. System Architecture & Pipeline Flow

<img width="2458" height="2347" alt="mermaid-diagram-2026-07-09-123134" src="https://github.com/user-attachments/assets/5233b714-787a-4127-8d22-c5abaf41ce63" />

### Project Structure:

```text
cuadpipe/
├── plots/                         # exploratory data analysis visuals
├── src/                           # Inference engine, ingestion logic, and utilities
├── .gitignore
├── LICENSE                        # Project license
├── README.md                      # this file
├── cuadpipe.ipynb                 # complete notebook for pipeline execution and evaluation
├── extraction_results.csv         # Final merged CSV output of structured extractions
├── extraction_results.json        # Final merged JSON output of structured extractions
├── main.py                        # main execution driver, hardware check, and batch controller
└── requirements.txt               # dependencies for reproducible environments
```
### Project Flow: 
1. **Ingest & Normalize (ingestion.py):** Upload PDFs locally or automatically from Zenodo in data/contracts. It also normalizes layout into raw plaintext cache.
2. **Context Window Slicing**: Segments plaintext into rolling 4,000-token chunks with a 250-token overlap to ensure clauses are never truncated at a page boundary.
3. **Isolated Multi-Pass Inference (Map Phase):** Processes each chunk through a 4-bit quantized Llama 3.1 8B engine via 4 distinct, high-focus passes (1 multi-paragraph summary pass + 3 single-task verbatim clause passes).
4. **Smart Reconciliation (Aggregate Phase):** 
- Summaries: Chronologically stitched section-by-section.
- Clauses: Aggregated using a length-maximization (max(len)) strategy to discard minor mentions and lock onto the primary master clause.
5. **Serialization:** Structured outputs are written simultaneously to extraction_results.csv and a pretty-printed extraction_results.json.


## 3. Setup & Installation Guide

`cuadpipe` is built to run across Kaggle cloud environments (originally Kaggle was used with 2x t4 gpu), local development machines and Jupyter/Colab notebooks.

### System Requirements

* **GPU:** NVIDIA GPU with $\ge$ 8 GB VRAM (RTX 3060 / RTX 4060 Ti / NVIDIA T4 / A10G).
* **CUDA:** Version 11.8 or 12.1+.
* **Python:** Version 3.10 or 3.11.

### 1. Kaggle Notebook Setup (COMPLETE CODE IN NOTEBOOK GIVEN - cuadpipe.ipynb)

For running the entire pipeline on kaggle, download the original notebook and upload on kaggle.
`cuadpipe` natively supports Kaggle's dual-T4 or single-T4 GPU accelerators.

1. In your Kaggle notebook settings, set **Accelerator** to **GPU T4 x2**.
2. Set **Internet Access** to **On**.
3. Add your Hugging Face token to **Kaggle Secrets** under the label `HF_TOKEN`. The pipeline will automatically detect and import it via `kaggle_secrets.UserSecretsClient()`.

**NOTE:--** On Kaggle T4 GPUs, running 4 targeted inference passes across 50 long contracts takes approximately **12 hours**. To prevent interactive browser timeouts, use **Save Version $\rightarrow$ Save & Run All (Commit)**.

### 2. For using Local Setup - 

```bash
# Clone the repository
git clone https://github.com/YourUsername/cuadpipe.git
cd cuadpipe

# Create and activate an isolated virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt

# Set your Hugging Face authentication token (Required for Llama 3.1 access)
export HF_TOKEN="hf_your_personal_access_token_here"

```

To run the full extraction pipeline across your document directory, run the following command on cmd after setup:

```bash
python main.py
```


### 3. For Jupyter / Google Colab Notebook Setup - 

In the first code cell of your notebook, execute:

```python
!pip install -q -U torch transformers accelerate bitsandbytes pypdf pandas matplotlib

import os
# Securely inject token in Colab
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')

```

### data input:

* **Automatic Bootstrapping:** If `data/contracts/` is empty, `main.py` automatically connects to Zenodo, downloads the official CUAD archive, and extracts a deterministic, randomized subset of 50 contracts (`seed=42`).
* **Custom Documents:** Drop your own `.pdf` agreements into `data/contracts/`. The engine will detect existing files, bypass the Zenodo download, and process your custom corpus immediately.



## 4. Architectural Defense: Why Exactly Map-Aggregate?

A standard engineering instinct when dealing with long documents (such as 40,000-token corporate agreements) is to reach for **Retrieval-Augmented Generation (RAG)** using vector database embeddings (e.g., FAISS or ChromaDB). While RAG is optimal for real-time QA across massive document corpora, **it is an anti-pattern for verbatim legal clause extraction on CUAD.**

### The RAG Retrieval Blindspot

Standard RAG relies on semantic cosine similarity between the user's query and chunked document embeddings. In complex commercial agreements:

1. **Boilerplate Dilution:** Core legal mechanics are frequently buried inside miscellaneous sections or introductory definitions that do not trigger high semantic similarity scores against standard query prompts like *"extract termination rules."*
2. **Silent Data Loss:** If the embedding model ranks a critical carve-out or liability exception as the 4th or 5th most similar chunk, a top-3 RAG retriever never feeds it to the LLM. The system outputs an incomplete clause or a false negative (`NONE`), resulting in catastrophic compliance failure.

### The Exhaustive Map-Aggregate Advantage

`cuadpipe` implements a deterministic sliding window across 100% of the document token space. By executing targeted inference passes on every overlapping chunk, the system guarantees that **no section of the contract is skipped**.

| Architectural Dimension | Exhaustive Map-Aggregate (`cuadpipe`) | Standard RAG Pipeline |
| --- | --- | --- |
| **Recall / Coverage** | **100% Guaranteed**: Every token is evaluated by the LLM reasoning engine. | **70–85% Variable**: Highly dependent on embedding model alignment. |
| **Verbatim Precision** | **Very High**: Targeted prompts focus 100% attention on one legal concept. | **Moderate** — Retrieved chunks often contain fragmented context boundaries. |
| **Compute Overhead** | **Heavy**: Requires ~12 inference passes per 3-chunk document. | **Light**: Requires only 3–4 inference passes on top ranked chunks. |
| **Best Use Case** | **Compliance, auditing, and benchmark scoring** where missing a clause is unacceptable. | **High-throughput triage** or conversational chat across thousands of docs. |

## 5. Tech Stack Justifications and Tradeoffs

### Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`

* **Why 8B?** In local and edge deployments, 70B+ models exceed the VRAM limits of single-node accelerators. The Llama 3.1 8B Instruct variant represents the industry sweet spot for instruction-following, semantic boundaries, and syntactic precision, punching significantly above its weight class when guided by domain-specific system prompts.
* **Why Instruct?** Base foundation models require few-shot completion prompts. Instruct-tuned models obey negative semantic constraints (`"DO NOT extract..."`) and formatting boundaries natively.

### Hardware Optimization: 4-bit NormalFloat Quantization (`bitsandbytes`)

Running an 16-bit (`float16` or `bfloat16`) 8B model requires **~16 GB of raw VRAM just to store the model weights**, leaving zero memory for the Key-Value (KV) generation cache or activation buffers. This causes instant Out-Of-Memory (OOM) crashes on 16 GB accelerators like the NVIDIA T4.

`cuadpipe` integrates `BitsAndBytesConfig` with the following parameters:

* `load_in_4bit=True` using `bnb_4bit_quant_type="nf4"` (NormalFloat 4-bit): An information-theoretically optimal data type for normally distributed neural network weights.
* `bnb_4bit_compute_dtype=torch.bfloat16`: While weights are stored in 4-bit to compress footprint down to **~5.5 GB VRAM**, mathematical tensor multiplications are dynamically dequantized and executed in 16-bit precision, preserving generation accuracy without latency spikes.
* `bnb_4bit_use_double_quant=True`: Quantizes the quantization constants themselves, saving an additional ~0.4 bits per parameter (~400 MB of VRAM buffer).

### Hyperparameter Discipline

* `temperature=0.1`: Generative creativity is a liability in legal analysis. Near-zero temperature locks the model into greedy decoding, forcing it to copy text deterministically rather than attempting to paraphrase or embellish legalese.
* `return_full_text=False`: Strips the input prompt from the pipeline return payload, preventing memory bloat during multi-chunk serialization.
* `max_nex_tokens = 768`: length analysis was done for required output and 768 was determined to be sufficient.


## 6. Output structure and the "NONE" output Advantage - 

<img width="1284" height="167" alt="image" src="https://github.com/user-attachments/assets/bd50ebfd-e067-4259-b28f-8789ea512def" />

A core innovation in `cuadpipe` is adding negative constraints and treating model refusal not as an error, but as a **verifiable classification signal**.

In naive extraction pipelines, if a chunk does not contain a Termination clause, the LLM often hallucinates a generic explanation or attempts to summarize whatever text is present. To prevent this, our prompts enforce **Negative Semantic Constraints**:

```text
STRICT RULES:
1. VERBATIM COPY ONLY: Copy the text character-for-character. Do not paraphrase.
2. NEGATIVE CONSTRAINT: DO NOT extract 'Effects of Termination', survival rules, or post-termination logistics (like returning property or final billing).
3. NO POINTERS: Do not output 'See Section 10'. If the actual termination rules are not in this chunk, output exactly: NONE

```

### Why This Is an Advantage

1. **Eliminates False-Positive Boilerplate:** In commercial contracts, the word *"terminate"* appears dozens of times in non-substantive contexts (e.g., *"Upon termination, receiving party shall return all laptops"*). By explicitly instructing the model to reject post-termination logistics, we force it to output `NONE` on boilerplate chunks, reserving positive extractions strictly for the master clause defining *when and how* termination can be triggered.
2. **Automates Noise Filtering:** Our extraction engine runs a regex validation layer (`_is_valid_clause`) that traps informal conversational refusals (e.g., *"Not present in this text,"* *"No clause found,"* or *"See Section 5"*) and normalizes them to clean `NONE` strings. This ensures downstream databases receive pristine structured data.


## 7. Handling Large Texts: Context EDA & Windowing Strategy

### The Context Length Challenge

Exploratory Data Analysis (EDA) on the CUAD dataset reveals that commercial agreements exhibit an extremely wide token distribution. While the median document sits around 6,000 to 8,000 tokens, complex master service agreements and merger filings routinely scale past **30,000 to 45,000 tokens**.

<img width="474" height="519" alt="image" src="https://github.com/user-attachments/assets/08f75ef9-ca57-406a-871e-c3def66ac07b" />

<img width="474" height="519" alt="image" src="https://github.com/ParamAhuja/CuadPipe/blob/main/plots/token_distribution.png" />

Feeding a 40,000-token document directly into a local LLM creates two fatal bottlenecks:

1. **Quadratic Attention Degradation:** As context length scales, the attention mechanism's ability to isolate specific verbatim spans degrades (the "Lost in the Middle" phenomenon).
2. **VRAM Cache Thrashing:** The KV cache memory requirement scales linearly with sequence length. On a 16 GB GPU, sequences past 12,000 tokens trigger CUDA OOM exceptions.

### The Sliding-Window Solution

To process contracts of arbitrary length within a fixed memory envelope, `cuadpipe` employs a deterministic token window slicer:

$$\text{Chunk}_i = \text{Tokens}\left[i \cdot (W - O) : i \cdot (W - O) + W\right]$$


where window size $W = 4000$ tokens and overlap $O = 250$ tokens.

* **Why 4000 Tokens?** Fits comfortably inside Llama 3.1's native context window while leaving ample headroom for system instructions and output generation buffers.
* **Why 250 Tokens Overlap?** Legal clauses often span across paragraph breaks or page boundaries. A 250-token overlap ensures that if a critical sentence sits right at the edge of a window cut, the entire clause is fully captured intact inside the subsequent chunk.

## 8. Real-World Failures Found & Solutions Implemented:

During pipeline development and visual ground-truth inspection against raw PDFs, I uncovered 6 distinct failure modes that break standard LLM extractors. Here is how `cuadpipe` systematically engineers around each one:

| # | Real-World Failure Mode | Root Cause | Engineering Solution Implemented |
| --- | --- | --- | --- |
| **1** | **Meta-Reference Leaks** *(e.g., "See Section 5")* | Token starvation inside single multi-task prompts (512 token ceiling shared across 4 tasks). | **Targeted Single-Clause Passes:** Separate inference runs allocate dedicated token budgets per clause. |
| **2** | **Paraphrasing / Hallucinations** | Attention dilution when asking an 8B model to track 4 complex targets simultaneously. | **100% Task Isolation:** One target per prompt + strict `"VERBATIM COPY ONLY"` system directives. |
| **3** | **Grabbing Post-Termination Logistics** | Keyword triggering on the word *"terminate"* inside boilerplate "Effects of Termination" sections. | **Negative Prompt Constraints:** Explicit exclusion directives (`"DO NOT extract Effects..."`). |
| **4** | **Incomplete / Short Extractions** | Naive "First-Match" aggregation locks onto brief introductory mentions in early chunks. | **Length-Maximization Aggregation:** Compares extractions across all chunks and retains $\max(\text{len}(\text{chunk}_i))$. |
| **5** | **Punctuation Injection** | LLM grammar bias automatically appends punctuations to incomplete sentence spans. | **Evaluation Normalizer:** `evaluate.py` strips all punctuation before checking string containment. |
| **6** | **Stripped Bullet Numbers** | LLM internal formatter removes prefixes like `(a)` or `1.` to create "cleaner" paragraphs. | **Regex Prefix Stripper:** Normalizer strips alphanumeric bullet markers from both source and extraction prior to matching. |

**NOTE:** > For standard benchmark datasets like CUAD, a Length-Maximization aggregation strategy is used to isolate the primary master clause. For production deployment on heterogeneous contracts, this can be seamlessly swapped to an Appending + Fuzzy Deduplication pipeline to prevent dropping split clauses or post-execution amendments.



### Essential References & Context Links

* **Video Introduction to CUAD:** [YouTube: The Atticus Project - CUAD Overview](https://www.youtube.com/watch?v=hFUSdgryXyU)
* **Official Dataset Repository:** [GitHub: TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad/tree/main)
* **Kaggle Dataset Source:** [Atticus Open Contract Dataset (AOK) Beta](https://www.kaggle.com/datasets/theatticusproject/atticus-open-contract-dataset-aok-beta)
* **Original CUAD Research Paper:** [ArXiv: CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/pdf/2103.06268)
* **Hugging Face Dataset Portal:** [huggingface.co/datasets/theatticusproject/cuad](https://huggingface.co/datasets/theatticusproject/cuad)

*Built for high-precision document extraction with systems-level robustness over naive API wrappers.*
