import torch
import json
import os
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
        
        print("Loading Llama 3.1 Tokenizer and Model (4-bit)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            token=hf_token
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.1,  
            return_full_text=False
        )
        
        # Hard limits for Kaggle T4 VRAM
        self.chunk_size = 6000
        self.overlap = 500
        
    def _chunk_tokens(self, tokens: list) -> list:
        """Slides a window across the token array to create overlapping chunks."""
        chunks = []
        i = 0
        while i < len(tokens):
            chunks.append(tokens[i:i + self.chunk_size])
            i += self.chunk_size - self.overlap
        return chunks

    def process_contract(self, contract_id: str, contract_text: str) -> ContractExtractionRecord:
        system_prompt = """You are an expert legal AI. Your task is to analyze the provided legal contract chunk and extract specific information into a strict JSON format. 

REQUIREMENTS:
1. "summary": Provide a concise summary of this specific text chunk. If it contains the agreement purpose or obligations, state them.
2. "termination_clause": Extract the EXACT verbatim text span defining Termination. Do not summarize. If not present, return "NONE".
3. "confidentiality_clause": Extract the EXACT verbatim text span defining Confidentiality. If not present, return "NONE".
4. "liability_clause": Extract the EXACT verbatim text span defining Liability. If not present, return "NONE".

Output ONLY valid JSON matching this schema:
{
  "summary": "...",
  "termination_clause": "...",
  "confidentiality_clause": "...",
  "liability_clause": "..."
}"""

        tokens = self.tokenizer.encode(contract_text)
        chunks = self._chunk_tokens(tokens)
        
        print(f"      [Info] Document split into {len(chunks)} overlapping chunks.")
        
        # Aggregation state
        final_summary = ""
        final_termination = "NONE"
        final_confidentiality = "NONE"
        final_liability = "NONE"

        for chunk_idx, chunk_tokens in enumerate(chunks):
            safe_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTRACT TEXT CHUNK:\n\n{safe_text}\n\nExtract the requested fields in JSON format."}
            ]

            outputs = self.pipe(messages)
            generated_text = outputs[0]["generated_text"].strip()
            
            if generated_text.startswith("```json"):
                generated_text = generated_text[7:-3].strip()
            elif generated_text.startswith("```"):
                generated_text = generated_text[3:-3].strip()

            try:
                parsed_data = json.loads(generated_text)
                
                # Aggregate findings: Keep the first chunk's summary (usually contains the preamble/purpose)
                if chunk_idx == 0:
                    final_summary = parsed_data.get("summary", "")
                
                # If a clause is found and we don't already have one, store it
                if parsed_data.get("termination_clause") != "NONE" and final_termination == "NONE":
                    final_termination = parsed_data["termination_clause"]
                    
                if parsed_data.get("confidentiality_clause") != "NONE" and final_confidentiality == "NONE":
                    final_confidentiality = parsed_data["confidentiality_clause"]
                    
                if parsed_data.get("liability_clause") != "NONE" and final_liability == "NONE":
                    final_liability = parsed_data["liability_clause"]
                    
            except Exception as e:
                print(f"      [Error] JSON Parse Failure on chunk {chunk_idx}: {e}")

        return ContractExtractionRecord(
            contract_id=contract_id,
            summary=final_summary if final_summary else "Summary generation failed.",
            termination_clause=final_termination,
            confidentiality_clause=final_confidentiality,
            liability_clause=final_liability
        )
