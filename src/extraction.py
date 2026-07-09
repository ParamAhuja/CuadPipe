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
        
        # Default ceiling set to 1024 to accommodate massive legal clauses without truncation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,  
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

    def _run_inference(self, system_instruction: str, user_content: str, max_tokens: int = 1024) -> str:
        """Executes targeted inference, parsing multiple potential XML tag blocks robustly."""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        
        with torch.no_grad():
            outputs = self.pipe(messages, max_new_tokens=max_tokens)
            
        result = outputs[0]["generated_text"].strip()
        
        # Strip markdown code blocks if the model injects them
        if result.startswith("```"):
            result = re.sub(r"^```[a-zA-Z]*\n?|```$", "", result).strip()
            
        # MULTI-TAG PARSING: Find all instances of <TEXT>...</TEXT> or <TXT>...</TXT>
        matches = re.findall(r"<T[E]?XT>(.*?)</T[E]?XT>", result, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Clean and filter matches, discarding empty blocks or accidental "NONE" markers inside strings
            cleaned_matches = [m.strip() for m in matches if m.strip()]
            if len(cleaned_matches) == 1:
                result = cleaned_matches[0]
            elif len(cleaned_matches) > 1:
                # If "NONE" is part of multiple blocks by mistake, prioritize actual content blocks
                actual_content = [m for m in cleaned_matches if m.upper() != "NONE"]
                result = "\n\n".join(actual_content) if actual_content else "NONE"
            else:
                result = "NONE"
        else:
            # Fallback block cleanup if the model strips tags but outputs content
            result = re.sub(r"</?T[E]?XT>", "", result, flags=re.IGNORECASE).strip()
            
        del outputs
        del messages
        gc.collect()
        torch.cuda.empty_cache()
        
        return result

    def process_contract(self, contract_id: str, contract_text: str) -> ContractExtractionRecord:
        tokens = self.tokenizer.encode(contract_text)
        chunks = self._chunk_tokens(tokens)
        
        print(f"      [Info] Sliced into {len(chunks)} chunk(s). Running passes (Summary: 512 max | Clauses: 1024 max)...")
        
        prompts = {
            "summary": (
                "You are an expert legal AI. Summarize the provided contract chunk accurately and concisely. "
                "Use only the information explicitly stated in the chunk. Do not infer, assume, or invent details from other parts of the contract. "
                "When present, include the purpose of the clause, key obligations, rights, responsibilities, important conditions or exceptions, deadlines, confidentiality provisions, termination terms, liabilities, risks, or penalties. "
                "If the chunk contains definitions or boilerplate language, briefly summarize its function. "
                "Write in clear, professional prose without markdown."
            ),
            "termination_clause": (
                "You are a legal text extraction system.\n\n"
                "Task:\n"
                "Locate and extract the exact contiguous text span(s) from the provided contract chunk that define when, why, or how either party may terminate, cancel, or end the agreement.\n\n"
                "Extraction Rules:\n"
                "1. Copy the text exactly as it appears in the source.\n"
                "2. The extracted text must be an exact substring of the input.\n"
                "3. Do NOT paraphrase, rewrite, summarize, correct grammar, or modify punctuation, capitalization, spacing, or wording.\n"
                "4. Extract only contiguous text. Never remove or rewrite individual sentences from within an extracted passage.\n"
                "5. Do NOT include sections whose primary purpose is only the effects of termination, survival clauses, or post-termination obligations (such as return of property), unless they are inseparable from the termination provision within the same contiguous clause.\n"
                "6. If multiple separate termination provisions appear in this chunk, return each in its own <TEXT>...</TEXT> block.\n\n"
                "Output Rules:\n"
                "- Output ONLY <TEXT>...</TEXT> blocks.\n"
                "- Do NOT include headings, labels, explanations, markdown, or any other text.\n"
                "- If no termination provision exists in this chunk, output exactly: <TEXT>NONE</TEXT>"
            ),
            "confidentiality_clause": (
                "You are a legal text extraction system.\n\n"
                "Task:\n"
                "Locate and extract the exact contiguous text span(s) from the provided contract chunk that define confidentiality obligations, non-disclosure obligations, confidential information, proprietary information, trade secrets, or restrictions on the use or disclosure of protected information.\n\n"
                "Extraction Rules:\n"
                "1. Copy the text exactly as it appears in the source.\n"
                "2. The extracted text must be an exact substring of the input.\n"
                "3. Do NOT paraphrase, rewrite, summarize, correct grammar, or modify punctuation, capitalization, spacing, or wording.\n"
                "4. Extract only contiguous text. Never remove or rewrite individual sentences from within an extracted passage.\n"
                "5. If multiple separate confidentiality provisions appear in this chunk, return each in its own <TEXT>...</TEXT> block.\n\n"
                "Output Rules:\n"
                "- Output ONLY <TEXT>...</TEXT> blocks.\n"
                "- Do NOT include headings, labels, explanations, markdown, or any other text.\n"
                "- If no confidentiality provision exists in this chunk, output exactly: <TEXT>NONE</TEXT>"
            ),
            "liability_clause": (
                "You are a legal text extraction system.\n\n"
                "Task:\n"
                "Locate and extract the exact contiguous text span(s) from the provided contract chunk that define limitations of liability, exclusions of liability, indemnification, hold harmless obligations, damages limitations, liability caps, or exclusions of specific types of damages.\n\n"
                "Extraction Rules:\n"
                "1. Copy the text exactly as it appears in the source.\n"
                "2. The extracted text must be an exact substring of the input.\n"
                "3. Do NOT paraphrase, rewrite, summarize, correct grammar, or modify punctuation, capitalization, spacing, or wording.\n"
                "4. Extract only contiguous text. Never remove or rewrite individual sentences from within an extracted passage.\n"
                "5. If multiple separate liability provisions appear in this chunk, return each in its own <TEXT>...</TEXT> block.\n\n"
                "Output Rules:\n"
                "- Output ONLY <TEXT>...</TEXT> blocks.\n"
                "- Do NOT include headings, labels, explanations, markdown, or any other text.\n"
                "- If no liability provision exists in this chunk, output exactly: <TEXT>NONE</TEXT>"
            )
        }
        
        chunk_summaries = []
        best_termination = "NONE"
        best_confidentiality = "NONE"
        best_liability = "NONE"

        for chunk_idx, chunk_tokens in enumerate(chunks):
            safe_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            user_prompt = f"CONTRACT TEXT CHUNK:\n\n{safe_text}\n\nExecute the extraction task strictly following your instructions."
            
            # Pass 1: Summary (Locked to 512 tokens for concise narrative density)
            sum_text = self._run_inference(prompts["summary"], user_prompt, max_tokens=512)
            if sum_text and len(sum_text) > 20:
                chunk_summaries.append(f"**Section {chunk_idx + 1}:**\n{sum_text}")
                
            # Pass 2: Termination Clause (Locked to 1024 tokens to prevent truncation of long provisions)
            term_text = self._run_inference(prompts["termination_clause"], user_prompt, max_tokens=1024)
            if self._is_valid_clause(term_text):
                if best_termination == "NONE" or len(term_text) > len(best_termination):
                    best_termination = term_text
                    
            # Pass 3: Confidentiality Clause (Locked to 1024 tokens)
            conf_text = self._run_inference(prompts["confidentiality_clause"], user_prompt, max_tokens=1024)
            if self._is_valid_clause(conf_text):
                if best_confidentiality == "NONE" or len(conf_text) > len(best_confidentiality):
                    best_confidentiality = conf_text
                    
            # Pass 4: Liability Clause (Locked to 1024 tokens)
            liab_text = self._run_inference(prompts["liability_clause"], user_prompt, max_tokens=1024)
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