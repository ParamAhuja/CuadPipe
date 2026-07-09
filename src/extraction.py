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
        
        # Globally locked to 512 tokens across the entire pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,  
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
        """Screens for conversational refusals only if the snippet is very short."""
        if not text or len(text.strip()) < 15:
            return False
            
        clean_text = text.strip().lower()
        
        if len(clean_text) < 100:
            invalid_patterns = [
                r"^none\.?$",
                r"^not present.*",
                r"^not found.*",
                r"^no .* clause.*",
                r"^see section \d+.*",
                r"^as stated in.*",
                r"^n/a\.?$"
            ]
            for pattern in invalid_patterns:
                if re.match(pattern, clean_text):
                    return False
                    
        return True

    def _run_inference(self, system_instruction: str, user_content: str, max_tokens: int = 512) -> str:
        """Executes targeted inference with strict 512-token budgets and clean XML tag parsing."""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        
        with torch.no_grad():
            outputs = self.pipe(messages, max_new_tokens=max_tokens)
            
        result = outputs[0]["generated_text"].strip()
        
        # Strip markdown code block wrappings if the model injects them
        if result.startswith("```"):
            result = re.sub(r"^```[a-zA-Z]*\n?|```$", "", result).strip()
            
        # Grab exact text inside <TEXT>...</TEXT> tags
        tag_match = re.search(r"<TEXT>(.*?)</TEXT>", result, re.DOTALL | re.IGNORECASE)
        if tag_match:
            result = tag_match.group(1).strip()
        else:
            # If tags were dropped, just strip dangling tags cleanly without guessing
            result = re.sub(r"</?TEXT>", "", result, flags=re.IGNORECASE).strip()
            
        del outputs
        del messages
        gc.collect()
        torch.cuda.empty_cache()
        
        return result

    def process_contract(self, contract_id: str, contract_text: str) -> ContractExtractionRecord:
        tokens = self.tokenizer.encode(contract_text)
        chunks = self._chunk_tokens(tokens)
        
        print(f"      [Info] Sliced into {len(chunks)} chunk(s). Running 4 targeted passes per chunk (512 tokens max)...")
        
        prompts = {
            "summary": (
                "You are an expert legal analyst. Provide a professional, thorough summary of the provided contract chunk. "
                "Synthesize any core business purpose, key obligations, OR financial terms present in this chunk into cohesive, readable paragraphs. "
                "Write clean prose without excessive bolding, markdown formatting, or bulleted lists."
            ),
            "termination_clause": (
                "You are a precise legal data extraction tool. Extract the EXACT verbatim text span from the contract "
                "that defines WHEN and HOW either party may terminate or end the agreement.\n\n"
                "STRICT OUTPUT STRUCTURE:\n"
                "1. VERBATIM COPY ONLY: Copy the text character-for-character from the source. Do not paraphrase or alter text.\n"
                "2. NEGATIVE CONSTRAINT: DO NOT extract 'Effects of Termination', survival rules, or post-termination logistics (like returning property).\n"
                "3. TAG WRAPPING: You MUST wrap your exact verbatim extraction inside <TEXT> and </TEXT> tags. Do not write introductory headings or labels outside the tags.\n"
                "4. If termination rules are not present in this specific chunk, output exactly: <TEXT>NONE</TEXT>"
            ),
            "confidentiality_clause": (
                "You are a precise legal data extraction tool. Extract the EXACT verbatim text span defining "
                "Confidentiality obligations, non-disclosure terms, or proprietary information protection.\n\n"
                "STRICT OUTPUT STRUCTURE:\n"
                "1. VERBATIM COPY ONLY: Copy the text character-for-character from the source. Do not paraphrase or alter text.\n"
                "2. TAG WRAPPING: You MUST wrap your exact verbatim extraction inside <TEXT> and </TEXT> tags. Do not write introductory headings or labels outside the tags.\n"
                "3. If confidentiality terms are not present in this specific chunk, output exactly: <TEXT>NONE</TEXT>"
            ),
            "liability_clause": (
                "You are a precise legal data extraction tool. Extract the EXACT verbatim text span defining "
                "Limitations of Liability, indemnification rules, damages caps, or liability exclusions.\n\n"
                "STRICT OUTPUT STRUCTURE:\n"
                "1. VERBATIM COPY ONLY: Copy the text character-for-character from the source. Do not paraphrase or alter text.\n"
                "2. TAG WRAPPING: You MUST wrap your exact verbatim extraction inside <TEXT> and </TEXT> tags. Do not write introductory headings or labels outside the tags.\n"
                "3. If liability terms are not present in this specific chunk, output exactly: <TEXT>NONE</TEXT>"
            )
        }
        
        chunk_summaries = []
        best_termination = "NONE"
        best_confidentiality = "NONE"
        best_liability = "NONE"

        for chunk_idx, chunk_tokens in enumerate(chunks):
            safe_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            user_prompt = f"CONTRACT TEXT CHUNK:\n\n{safe_text}\n\nExecute the extraction task strictly following your instructions."
            
            # Pass 1: Summary (Locked to 512 tokens)
            sum_text = self._run_inference(prompts["summary"], user_prompt, max_tokens=512)
            if sum_text and len(sum_text) > 20:
                chunk_summaries.append(f"**Section {chunk_idx + 1}:**\n{sum_text}")
                
            # Pass 2: Termination Clause (Locked to 512 tokens)
            term_text = self._run_inference(prompts["termination_clause"], user_prompt, max_tokens=512)
            if self._is_valid_clause(term_text):
                if best_termination == "NONE" or len(term_text) > len(best_termination):
                    best_termination = term_text
                    
            # Pass 3: Confidentiality Clause (Locked to 512 tokens)
            conf_text = self._run_inference(prompts["confidentiality_clause"], user_prompt, max_tokens=512)
            if self._is_valid_clause(conf_text):
                if best_confidentiality == "NONE" or len(conf_text) > len(best_confidentiality):
                    best_confidentiality = conf_text
                    
            # Pass 4: Liability Clause (Locked to 512 tokens)
            liab_text = self._run_inference(prompts["liability_clause"], user_prompt, max_tokens=512)
            if self._is_valid_clause(liab_text):
                if best_liability == "NONE" or len(liab_text) > len(best_liability):
                    best_liability = liab_text

        final_summary = "\n\n".join(chunk_summaries) if chunk_summaries else "Summary extraction failed."

        return ContractExtractionRecord(
            contract_id=contract_id,
            summary=final_summary,
            termination_clause=best_termination,
            confidentiality_clause=best_confidentiality,
            liability_clause=best_liability
        )