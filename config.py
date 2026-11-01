import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Ollama Configuration (Local LLM)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # Default to available model
    
    # Data Source Configuration
    DATA_SOURCE = os.getenv("DATA_SOURCE", "rapidapi")  # Use RapidAPI for reliable access
    RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
    
    # Financial Configuration
    RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))  # 5% default
    DEFAULT_BENCHMARK = os.getenv("DEFAULT_BENCHMARK", "^GSPC")  # S&P 500 for Yahoo Finance
    

    
    # Calculation Parameters
    DEFAULT_PERIOD = os.getenv("DEFAULT_PERIOD", "1y")  # 1 year default
    TRADING_DAYS_PER_YEAR = 252
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        # No API key needed for local Ollama!
        return True 