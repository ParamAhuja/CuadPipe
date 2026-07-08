import os
import re
import gc
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .utils import ContractExtractionRecord

class LLMExtractor:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        hf_token = os.environ.get("HF_TOKEN")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        print("      Loading Llama 3.1 Tokenizer and Model (4-bit Targeted Inference Engine)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            token=hf_token
        )
        
        # INCREASED BUDGET: 768 tokens ensures long clauses and thorough summaries never truncate
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=768,  
            temperature=0.1,  
            return_full_text=False
        )
        
        self.chunk_size = 4000
        self.overlap = 250
        
    def _chunk_tokens(self, tokens: list) -> list:
        """Slides a window across the token array to create overlapping chunks."""
        chunks = []
        i = 0
        while i < len(tokens):
            chunks.append(tokens[i:i + self.chunk_size])
            i += self.chunk_size - self.overlap
        return chunks

    def _is_valid_clause(self, text: str) -> bool:
        """Filters out informal model refusals, pointers, or short hallucinated placeholders."""
        if not text or len(text.strip()) < 15:
            return False
            
        clean_text = text.strip().lower()
        invalid_patterns = [
            r"^none\.?$",
            r"^not present.*",
            r"^not found.*",
            r"^no .* clause.*found.*",
            r"^see section \d+.*",
            r"^as stated in.*",
            r"^n/a\.?$"
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, clean_text):
                return False
        return True

    def _run_inference(self, system_instruction: str, user_content: str) -> str:
        """Executes a single targeted inference pass with strict memory clearance."""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        
        with torch.no_grad():
            outputs = self.pipe(messages)
            
        result = outputs[0]["generated_text"].strip()
        
        # Strip conversational markdown artifacts if the model injects them
        if result.startswith("```"):
            result = re.sub(r"^```[a-zA-Z]*\n?|```$", "", result).strip()
            
        del outputs
        del messages
        gc.collect()
        torch.cuda.empty_cache()
        
        return result

    def process_contract(self, contract_id: str, contract_text: str) -> ContractExtractionRecord:
        tokens = self.tokenizer.encode(contract_text)
        chunks = self._chunk_tokens(tokens)
        
        print(f"      [Info] Sliced into {len(chunks)} chunk(s). Running 4 targeted passes per chunk...")
        
        # System prompts engineered with Negative Constraints and strict formatting rules
        prompts = {
            "summary": (
                "You are an expert legal analyst. Provide a comprehensive, highly detailed summary of the provided contract text. "
                "Highlight the core business purpose, key obligations, financial terms, and involved parties. "
                "Do not be overly brief. Write a thorough, professional synthesis."
            ),
            "termination_clause": (
                "You are a precise legal data extraction tool. Your task is to extract the EXACT verbatim text span from the contract "
                "that defines WHEN and HOW either party may terminate or end the agreement.\n\n"
                "STRICT RULES:\n"
                "1. VERBATIM COPY ONLY: Copy the text character-for-character. Do not paraphrase or remove bullet numbers.\n"
                "2. NEGATIVE CONSTRAINT: DO NOT extract 'Effects of Termination', survival rules, or post-termination logistics (like returning laptops or final billing).\n"
                "3. NO POINTERS: Do not output 'See Section 10'. If the actual termination rules are not in this chunk, output exactly: NONE"
            ),
            "confidentiality_clause": (
                "You are a precise legal data extraction tool. Your task is to extract the EXACT verbatim text span defining "
                "Confidentiality obligations, non-disclosure terms, or the protection of proprietary information.\n\n"
                "STRICT RULES:\n"
                "1. VERBATIM COPY ONLY: Copy the text character-for-character. Do not paraphrase or alter punctuation.\n"
                "2. If confidentiality terms are not present in this specific chunk, output exactly: NONE"
            ),
            "liability_clause": (
                "You are a precise legal data extraction tool. Your task is to extract the EXACT verbatim text span defining "
                "Limitations of Liability, indemnification rules, damages caps, or liability exclusions.\n\n"
                "STRICT RULES:\n"
                "1. VERBATIM COPY ONLY: Copy the text character-for-character. Do not paraphrase or summarize.\n"
                "2. If liability terms are not present in this specific chunk, output exactly: NONE"
            )
        }
        
        # Aggregation State across all chunks
        chunk_summaries = []
        best_termination = "NONE"
        best_confidentiality = "NONE"
        best_liability = "NONE"

        for chunk_idx, chunk_tokens in enumerate(chunks):
            safe_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            user_prompt = f"CONTRACT TEXT CHUNK:\n\n{safe_text}\n\nExecute the extraction task strictly following your instructions."
            
            # Pass 1: Comprehensive Summary
            sum_text = self._run_inference(prompts["summary"], user_prompt)
            if sum_text and len(sum_text) > 20:
                chunk_summaries.append(f"**Section {chunk_idx + 1}:** {sum_text}")
                
            # Pass 2: Termination Clause (Length-Maximization Aggregation)
            term_text = self._run_inference(prompts["termination_clause"], user_prompt)
            if self._is_valid_clause(term_text):
                if best_termination == "NONE" or len(term_text) > len(best_termination):
                    best_termination = term_text
                    
            # Pass 3: Confidentiality Clause (Length-Maximization Aggregation)
            conf_text = self._run_inference(prompts["confidentiality_clause"], user_prompt)
            if self._is_valid_clause(conf_text):
                if best_confidentiality == "NONE" or len(conf_text) > len(best_confidentiality):
                    best_confidentiality = conf_text
                    
            # Pass 4: Liability Clause (Length-Maximization Aggregation)
            liab_text = self._run_inference(prompts["liability_clause"], user_prompt)
            if self._is_valid_clause(liab_text):
                if best_liability == "NONE" or len(liab_text) > len(best_liability):
                    best_liability = liab_text

        # Assemble a rich, multi-part document summary
        final_summary = "\n\n".join(chunk_summaries) if chunk_summaries else "Summary extraction failed."

        return ContractExtractionRecord(
            contract_id=contract_id,
            summary=final_summary,
            termination_clause=best_termination,
            confidentiality_clause=best_confidentiality,
            liability_clause=best_liability
        )