import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import asyncio
import ollama
from config import Config
from data_fetcher_rapidapi import RapidAPIDataFetcher
from ratio_calculator import RatioCalculator, RatioResults

@dataclass
class ParsedRequest:
    """Parsed user request for financial ratio calculations"""
    symbols: List[str]
    ratios_requested: List[str]  # ['sharpe', 'sortino', 'treynor']
    portfolio_weights: Optional[Dict[str, float]] = None
    time_period: Optional[str] = None
    custom_benchmark: Optional[str] = None
    risk_free_rate: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'symbols': self.symbols,
            'ratios_requested': self.ratios_requested,
            'portfolio_weights': self.portfolio_weights,
            'time_period': self.time_period,
            'custom_benchmark': self.custom_benchmark,
            'risk_free_rate': self.risk_free_rate
        }

class LLMIntegration:
    """Handles Ollama-based prompt parsing and natural language response generation"""
    
    def __init__(self):
        self.config = Config()
        self.client = ollama.AsyncClient(host=self.config.OLLAMA_BASE_URL)
        self.data_fetcher = RapidAPIDataFetcher(self.config)
        self.ratio_calculator = RatioCalculator()
    
    async def process_user_query(self, user_prompt: str) -> str:
        """
        Main entry point: process user query and return natural language response
        
        Args:
            user_prompt: User's natural language query
        
        Returns:
            Natural language response with ratio calculations
        """
        try:
            # Step 1: Parse the user prompt
            parsed_request = await self.parse_prompt(user_prompt)
            
            # Step 2: Fetch data and calculate ratios
            results = await self.calculate_ratios(parsed_request)
            
            # Step 3: Generate natural language response
            response = await self.generate_response(user_prompt, parsed_request, results)
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error processing your request: {str(e)}. Please try rephrasing your question or check if the symbols are valid."
    
    async def parse_prompt(self, user_prompt: str) -> ParsedRequest:
        """
        Parse user prompt to extract financial analysis requirements
        
        Args:
            user_prompt: Natural language prompt from user
        
        Returns:
            ParsedRequest object with extracted parameters
        """
        system_prompt = """You are a financial analysis assistant. Parse the user's request and extract the following information in JSON format:

{
  "symbols": ["list of stock symbols mentioned"],
  "ratios_requested": ["list of ratios requested: sharpe, sortino, treynor, or all"],
  "portfolio_weights": {"symbol": weight} or null if not specified,
  "time_period": "time period (1y, 2y, 6mo, etc.) or null for default",
  "custom_benchmark": "benchmark symbol if specified, null otherwise",
  "risk_free_rate": number or null if not specified
}

Rules:
1. If no specific ratios are mentioned, assume "all" ratios are requested
2. Extract stock symbols in uppercase (e.g., AAPL, TSLA, MSFT)
3. If portfolio weights aren't specified but multiple stocks are mentioned, assume equal weights
4. Time periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
5. Common benchmarks: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)

Examples:
- "Calculate Sharpe ratio for AAPL" → {"symbols": ["AAPL"], "ratios_requested": ["sharpe"], ...}
- "What are all ratios for my portfolio of TSLA and MSFT?" → {"symbols": ["TSLA", "MSFT"], "ratios_requested": ["all"], ...}
- "Compare Treynor ratios of SPY vs QQQ over 2 years" → {"symbols": ["SPY", "QQQ"], "ratios_requested": ["treynor"], "time_period": "2y", ...}

Return only the JSON object, no additional text."""

        try:
            response = await self.client.chat(
                model=self.config.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            content = response['message']['content']
            if not content:
                raise ValueError("Empty response from LLM")
            
            # Parse JSON response
            parsed_data = json.loads(content)
            
            # Validate and create ParsedRequest
            symbols = parsed_data.get('symbols', [])
            if not symbols:
                raise ValueError("No symbols found in the request")
            
            ratios_requested = parsed_data.get('ratios_requested', ['all'])
            if 'all' in ratios_requested:
                ratios_requested = ['sharpe', 'sortino', 'treynor']
            
            return ParsedRequest(
                symbols=symbols,
                ratios_requested=ratios_requested,
                portfolio_weights=parsed_data.get('portfolio_weights'),
                time_period=parsed_data.get('time_period'),
                custom_benchmark=parsed_data.get('custom_benchmark'),
                risk_free_rate=parsed_data.get('risk_free_rate')
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing prompt: {e}")
    
    async def calculate_ratios(self, request: ParsedRequest) -> Dict[str, Any]:
        """
        Calculate financial ratios based on parsed request
        
        Args:
            request: ParsedRequest object
        
        Returns:
            Dictionary with calculation results
        """
        results = {}
        
        try:
            # Fetch stock data
            if len(request.symbols) == 1:
                # Single stock analysis
                symbol = request.symbols[0]
                stock_data = await self.data_fetcher.fetch_stock_data(symbol, request.time_period or "1y")
                returns = self.data_fetcher.calculate_returns(stock_data)
                
                # Fetch benchmark data if needed for Treynor ratio
                market_returns = None
                if 'treynor' in request.ratios_requested:
                    benchmark = request.custom_benchmark or self.config.DEFAULT_BENCHMARK
                    market_data = await self.data_fetcher.fetch_benchmark_data(benchmark, request.time_period or "1y")
                    market_returns = self.data_fetcher.calculate_returns(market_data)
                
                # Calculate ratios
                ratio_results = self.ratio_calculator.calculate_all_ratios(
                    returns=returns,
                    market_returns=market_returns,
                    risk_free_rate=request.risk_free_rate or self.config.RISK_FREE_RATE,
                    period=request.time_period or "1y"
                )
                
                results[symbol] = {
                    'ratios': ratio_results,
                    'interpretations': self.ratio_calculator.interpret_ratios(ratio_results)
                }
                
            else:
                # Multiple stocks analysis (portfolio or comparison)
                stock_data = await self.data_fetcher.fetch_multiple_stocks(request.symbols, request.time_period or "1y")
                
                if not stock_data:
                    raise ValueError("No valid stock data found")
                
                # Fetch benchmark data if needed for Treynor ratio
                market_returns = None
                if 'treynor' in request.ratios_requested:
                    benchmark = request.custom_benchmark or self.config.DEFAULT_BENCHMARK
                    market_data = await self.data_fetcher.fetch_benchmark_data(benchmark, request.time_period or "1y")
                    market_returns = self.data_fetcher.calculate_returns(market_data)
                
                # If user specified portfolio weights, create a portfolio analysis
                if request.portfolio_weights is not None:
                    # Calculate portfolio returns with specified weights
                    portfolio_returns = self.data_fetcher.get_portfolio_returns(stock_data, request.portfolio_weights)
                    
                    # Calculate portfolio ratios
                    portfolio_ratios = self.ratio_calculator.calculate_all_ratios(
                        returns=portfolio_returns,
                        market_returns=market_returns,
                        risk_free_rate=request.risk_free_rate or self.config.RISK_FREE_RATE,
                        period=request.time_period or "1y"
                    )
                    
                    results['portfolio'] = {
                        'symbols': request.symbols,
                        'weights': request.portfolio_weights,
                        'ratios': portfolio_ratios,
                        'interpretations': self.ratio_calculator.interpret_ratios(portfolio_ratios)
                    }
                
                # Also calculate individual stock ratios for comparison
                for symbol, data in stock_data.items():
                    returns = self.data_fetcher.calculate_returns(data)
                    individual_ratios = self.ratio_calculator.calculate_all_ratios(
                        returns=returns,
                        market_returns=market_returns,
                        risk_free_rate=request.risk_free_rate or self.config.RISK_FREE_RATE,
                        period=request.time_period or "1y"
                    )
                    
                    results[symbol] = {
                        'ratios': individual_ratios,
                        'interpretations': self.ratio_calculator.interpret_ratios(individual_ratios)
                    }
            
            return results
            
        except Exception as e:
            raise Exception(f"Error calculating ratios: {str(e)}")
    
    async def generate_response(self, original_prompt: str, request: ParsedRequest, results: Dict[str, Any]) -> str:
        """
        Generate natural language response explaining the ratio calculations
        
        Args:
            original_prompt: Original user prompt
            request: ParsedRequest object
            results: Calculation results
        
        Returns:
            Natural language response
        """
        # Prepare data for LLM
        context = {
            'original_prompt': original_prompt,
            'request': request.to_dict(),
            'results': {}
        }
        
        # Convert results to serializable format
        for key, value in results.items():
            if isinstance(value, dict) and 'ratios' in value:
                context['results'][key] = {
                    'ratios': value['ratios'].to_dict(),
                    'interpretations': value['interpretations']
                }
                if 'symbols' in value:
                    context['results'][key]['symbols'] = value['symbols']
                if 'weights' in value:
                    context['results'][key]['weights'] = value['weights']
        
        system_prompt = """You are a financial analysis expert. Generate a comprehensive, natural language response explaining the financial ratio calculations.

Guidelines:
1. Start with a brief summary of what was analyzed
2. Explain each ratio clearly with its value and interpretation
3. Use the provided interpretations to explain what the ratios mean
4. Compare ratios between different stocks if multiple are analyzed
5. Provide actionable insights when possible
6. Use clear, professional language that's accessible to non-experts
7. Include the time period and any assumptions made
8. Format numbers clearly (e.g., "1.25" not "1.2500000")

Structure your response as:
- Summary of analysis
- Individual ratio explanations
- Comparative analysis (if applicable)
- Key insights and recommendations

Make the response informative but conversational."""

        try:
            response = await self.client.chat(
                model=self.config.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please analyze this financial data and provide a comprehensive explanation:\n\n{json.dumps(context, indent=2)}"}
                ]
            )
            
            content = response['message']['content']
            if not content:
                raise ValueError("Empty response from LLM")
            
            return content
            
        except Exception as e:
            # Fallback to basic response if LLM fails
            return self._generate_basic_response(request, results)
    
    def _generate_basic_response(self, request: ParsedRequest, results: Dict[str, Any]) -> str:
        """
        Generate a basic response if LLM fails
        
        Args:
            request: ParsedRequest object
            results: Calculation results
        
        Returns:
            Basic formatted response
        """
        response_parts = []
        
        if len(request.symbols) == 1:
            symbol = request.symbols[0]
            if symbol in results:
                ratios = results[symbol]['ratios']
                response_parts.append(f"Financial Ratio Analysis for {symbol}:")
                
                if ratios.sharpe_ratio is not None:
                    response_parts.append(f"• Sharpe Ratio: {ratios.sharpe_ratio:.3f}")
                if ratios.sortino_ratio is not None:
                    response_parts.append(f"• Sortino Ratio: {ratios.sortino_ratio:.3f}")
                if ratios.treynor_ratio is not None:
                    response_parts.append(f"• Treynor Ratio: {ratios.treynor_ratio:.3f}")
        else:
            response_parts.append(f"Portfolio Analysis for {', '.join(request.symbols)}:")
            
            if 'portfolio' in results:
                ratios = results['portfolio']['ratios']
                response_parts.append("Portfolio Ratios:")
                
                if ratios.sharpe_ratio is not None:
                    response_parts.append(f"• Sharpe Ratio: {ratios.sharpe_ratio:.3f}")
                if ratios.sortino_ratio is not None:
                    response_parts.append(f"• Sortino Ratio: {ratios.sortino_ratio:.3f}")
                if ratios.treynor_ratio is not None:
                    response_parts.append(f"• Treynor Ratio: {ratios.treynor_ratio:.3f}")
        
        return "\n".join(response_parts)
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate stock symbols
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            List of valid symbols
        """
        valid_symbols = []
        for symbol in symbols:
            if self.data_fetcher.validate_symbol(symbol):
                valid_symbols.append(symbol)
        return valid_symbols 