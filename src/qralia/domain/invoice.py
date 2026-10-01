"""Domain model describing the parameters of a synthetic ONEE invoice."""
from datetime import date

from pydantic import BaseModel


class InvoiceParams(BaseModel):
    """Parameters needed to generate one synthetic ONEE invoice."""

    invoice_number: str
    contract_number: str
    client_name: str
    client_address: str
    invoice_date: date
    due_date: date
    period_start: date
    period_end: date
    consumption_kwh: int      
    amount_before_tax: float  
    taxes: float               
    total_ttc: float           
    previous_unpaid: float     
    total_cash: float          
    total_check: float        
    tariff: str 
    tariff_tranche: int    
    index_previous: int
    index_current: int          
