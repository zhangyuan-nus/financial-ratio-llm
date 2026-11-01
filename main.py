#!/usr/bin/env python3
"""
Financial Ratio LLM System - Main Application

This is the main entry point for the Financial Ratio LLM system.
It provides command-line interface functionality.
"""

import asyncio
import argparse
import sys
import os
from typing import Optional
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialRatioLLM:
    """Main application class for the Financial Ratio LLM system"""
    
    def __init__(self):
        self.config = Config()
        self.llm_integration = None
    
    async def initialize(self):
        """Initialize the LLM integration"""
        try:
            self.config.validate()
            self.llm_integration = LLMIntegration()
            logger.info("Financial Ratio LLM system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            raise
    
    async def process_query(self, prompt: str) -> str:
        """Process a single query"""
        if not self.llm_integration:
            await self.initialize()
        
        try:
            response = await self.llm_integration.process_user_query(prompt)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error: {str(e)}"
    
    async def interactive_mode(self):
        """Run in interactive command-line mode"""
        print("=" * 70)
        print("🔥 Financial Ratio LLM System - Interactive Mode 🔥")
        print("=" * 70)
        print("Ask me to calculate financial ratios for any stock or portfolio!")
        print()
        print("📈 SINGLE STOCK ANALYSIS:")
        print("  • 'Calculate all ratios for AAPL'")
        print("  • 'What's the Sharpe ratio of Tesla?'")
        print()
        print("📊 STOCK COMPARISON:")
        print("  • 'Compare Sharpe ratios between AAPL and MSFT'")
        print("  • 'What are the ratios for AAPL and TSLA?'")
        print()
        print("💼 PORTFOLIO ANALYSIS:")
        print("  • 'Analyze my portfolio of 60% AAPL and 40% MSFT'")
        print("  • 'Calculate ratios for my 70% SPY, 30% QQQ portfolio'")
        print()
        print("💡 TIP: I can analyze any combination of stocks with real-time data!")
        print("Type 'quit' or 'exit' to stop.")
        print("=" * 70)
        
        await self.initialize()
        
        while True:
            try:
                # Get user input
                prompt = input("\n💰 Your query: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if not prompt:
                    continue
                
                print("\n🔄 Processing your query...")
                
                # Process the query
                response = await self.process_query(prompt)
                
                print(f"\n📊 Analysis Result:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                logger.error(f"Error in interactive mode: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Financial Ratio LLM System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py --query "Calculate ratios for AAPL"              # Single stock
  python main.py --query "Compare AAPL and MSFT"                  # Stock comparison  
  python main.py --query "Analyze my 60% AAPL, 40% MSFT portfolio" # Portfolio analysis
  python main.py --config                                          # Show configuration
        """
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Process a single query and exit"
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration"
    )
    
    args = parser.parse_args()
    
    # Create the main application
    app_instance = FinancialRatioLLM()
    
    try:
        if args.config:
            # Show configuration
            print("Current Configuration:")
            print(f"  Risk-free rate: {app_instance.config.RISK_FREE_RATE}")
            print(f"  Default benchmark: {app_instance.config.DEFAULT_BENCHMARK}")
            print(f"  Default period: {app_instance.config.DEFAULT_PERIOD}")
            print(f"  Trading days per year: {app_instance.config.TRADING_DAYS_PER_YEAR}")
            print(f"  Data source: {app_instance.config.DATA_SOURCE}")
            print(f"  Ollama model: {app_instance.config.OLLAMA_MODEL}")
            print(f"  Ollama URL: {app_instance.config.OLLAMA_BASE_URL}")
            return
        
        if args.query:
            # Process single query
            async def run_query():
                result = await app_instance.process_query(args.query)
                print(result)
            
            asyncio.run(run_query())
            
        else:
            # Interactive mode (default)
            asyncio.run(app_instance.interactive_mode())
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()