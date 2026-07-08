from pydantic import BaseModel, Field
from typing import Optional

class ContractExtractionRecord(BaseModel):
    contract_id: str
    summary: str = Field(description="100-150 word summary of purpose, obligations, and risks")
    termination_clause: Optional[str] = Field(description="Exact verbatim text of the Termination clause, or 'NONE'")
    confidentiality_clause: Optional[str] = Field(description="Exact verbatim text of the Confidentiality clause, or 'NONE'")
    liability_clause: Optional[str] = Field(description="Exact verbatim text of the Liability clause, or 'NONE'")
