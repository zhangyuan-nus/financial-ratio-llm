import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import asyncio
import time
import requests
import json
from config import Config

class RapidAPIDataFetcher:
    """Handles fetching financial data from Yahoo Finance via RapidAPI"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_key = getattr(config, 'RAPIDAPI_KEY', '')
        # Use the correct apidojo Yahoo Finance API
        self.base_url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
        }
        self.request_count = 0
        self.last_request_time = 0
        
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY not found in configuration. Please add it to your .env file.")
    
    def _rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # RapidAPI typically allows 500 requests per month on free tier
        # Add small delay to be respectful
        if time_since_last < 1:  # 1 second between requests
            sleep_time = 1 - time_since_last
            print(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        print(f"RapidAPI Request #{self.request_count}")
    

    def _parse_apidojo_chart_response(self, data_json: dict, symbol: str) -> pd.DataFrame:
        """Parse apidojo Yahoo Finance chart API response format"""
        try:
            # Navigate through the apidojo chart response structure
            # Structure: chart -> result -> [0] -> indicators/timestamp
            if 'chart' not in data_json:
                print(f"No 'chart' key in response: {list(data_json.keys())}")
                return pd.DataFrame()
            
            chart = data_json['chart']
            
            if 'result' not in chart:
                print(f"No 'result' key in chart: {list(chart.keys())}")
                return pd.DataFrame()
            
            result = chart['result']
            
            if not result or len(result) == 0:
                print("Empty result array")
                return pd.DataFrame()
            
            # Get the first result (should contain our symbol data)
            stock_data = result[0]
            
            if 'timestamp' not in stock_data:
                print(f"No 'timestamp' key in stock_data: {list(stock_data.keys())}")
                return pd.DataFrame()
            
            timestamps = stock_data['timestamp']
            
            if 'indicators' not in stock_data:
                print(f"No 'indicators' key in stock_data: {list(stock_data.keys())}")
                return pd.DataFrame()
            
            indicators = stock_data['indicators']
            
            if 'quote' not in indicators:
                print(f"No 'quote' key in indicators: {list(indicators.keys())}")
                return pd.DataFrame()
            
            quote_data = indicators['quote'][0]  # First quote set
            
            # Create DataFrame from the quote data
            df_data = {
                'Open': quote_data.get('open', []),
                'High': quote_data.get('high', []),
                'Low': quote_data.get('low', []),
                'Close': quote_data.get('close', []),
                'Volume': quote_data.get('volume', [])
            }
            
            # Handle adjusted close if available
            if 'adjclose' in indicators and indicators['adjclose']:
                df_data['Adj Close'] = indicators['adjclose'][0]['adjclose']
            else:
                df_data['Adj Close'] = df_data['Close']
            
            # Create DataFrame
            df = pd.DataFrame(df_data)
            
            # Convert timestamps to datetime index
            df.index = pd.to_datetime(timestamps, unit='s')
            df.index.name = 'Date'
            
            # Remove NaN values and sort
            df = df.dropna()
            df = df.sort_index(ascending=True)
            
            return df
            
        except Exception as e:
            print(f"Error parsing apidojo chart response: {str(e)}")
            print(f"Response keys: {list(data_json.keys()) if isinstance(data_json, dict) else 'Not a dict'}")
            return pd.DataFrame()
    
    def _parse_apidojo_response(self, data_json: dict, symbol: str) -> pd.DataFrame:
        """Parse apidojo Yahoo Finance API response format"""
        try:
            # Navigate through the apidojo response structure
            # Actual structure: timeseries -> result -> [0] -> indicators/timestamp
            if 'timeseries' not in data_json:
                print(f"No 'timeseries' key in response: {list(data_json.keys())}")
                return pd.DataFrame()
            
            timeseries = data_json['timeseries']
            
            if 'result' not in timeseries:
                print(f"No 'result' key in timeseries: {list(timeseries.keys())}")
                return pd.DataFrame()
            
            result = timeseries['result']
            
            if not result or len(result) == 0:
                print("Empty result array")
                return pd.DataFrame()
            
            # Get the first result (should contain our symbol data)
            stock_data = result[0]
            
            if 'indicators' not in stock_data:
                print(f"No 'indicators' key in stock_data: {list(stock_data.keys())}")
                return pd.DataFrame()
            
            indicators = stock_data['indicators']
            
            if 'quote' not in indicators:
                print(f"No 'quote' key in indicators: {list(indicators.keys())}")
                return pd.DataFrame()
            
            quote_data = indicators['quote'][0]  # First quote set
            
            # Get timestamps
            if 'timestamp' not in stock_data:
                print(f"No 'timestamp' key in stock_data: {list(stock_data.keys())}")
                return pd.DataFrame()
            
            timestamps = stock_data['timestamp']
            
            # Create DataFrame from the quote data
            df_data = {
                'Open': quote_data.get('open', []),
                'High': quote_data.get('high', []),
                'Low': quote_data.get('low', []),
                'Close': quote_data.get('close', []),
                'Volume': quote_data.get('volume', [])
            }
            
            # Handle adjusted close if available
            if 'adjclose' in indicators and indicators['adjclose']:
                df_data['Adj Close'] = indicators['adjclose'][0]['adjclose']
            else:
                df_data['Adj Close'] = df_data['Close']
            
            # Create DataFrame
            df = pd.DataFrame(df_data)
            
            # Convert timestamps to datetime index
            df.index = pd.to_datetime(timestamps, unit='s')
            df.index.name = 'Date'
            
            # Remove NaN values and sort
            df = df.dropna()
            df = df.sort_index(ascending=True)
            
            return df
            
        except Exception as e:
            print(f"Error parsing apidojo response: {str(e)}")
            print(f"Response keys: {list(data_json.keys()) if isinstance(data_json, dict) else 'Not a dict'}")
            return pd.DataFrame()
    
    def _parse_response(self, data_json: dict, symbol: str) -> pd.DataFrame:
        """Parse different response formats from various APIs"""
        try:
            # Try Yahoo Finance chart format
            if 'chart' in data_json and data_json['chart']['result']:
                result = data_json['chart']['result'][0]
                
                if 'timestamp' not in result or not result['timestamp']:
                    return pd.DataFrame()
                
                timestamps = result['timestamp']
                indicators = result['indicators']['quote'][0]
                
                df_data = {
                    'Open': indicators.get('open', []),
                    'High': indicators.get('high', []),
                    'Low': indicators.get('low', []),
                    'Close': indicators.get('close', []),
                    'Volume': indicators.get('volume', [])
                }
                
                # Handle adjusted close
                if 'adjclose' in result['indicators'] and result['indicators']['adjclose']:
                    df_data['Adj Close'] = result['indicators']['adjclose'][0]['adjclose']
                else:
                    df_data['Adj Close'] = df_data['Close']
                
                df = pd.DataFrame(df_data)
                df.index = pd.to_datetime(timestamps, unit='s')
                df.index.name = 'Date'
                
            # Try direct array format
            elif isinstance(data_json, list) and len(data_json) > 0:
                df = pd.DataFrame(data_json)
                if 'date' in df.columns:
                    df['Date'] = pd.to_datetime(df['date'])
                    df.set_index('Date', inplace=True)
                elif 'timestamp' in df.columns:
                    df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('Date', inplace=True)
                
            # Try nested data format
            elif 'data' in data_json:
                df_data = data_json['data']
                if isinstance(df_data, list) and len(df_data) > 0:
                    df = pd.DataFrame(df_data)
                    if 'date' in df.columns:
                        df['Date'] = pd.to_datetime(df['date'])
                        df.set_index('Date', inplace=True)
                else:
                    return pd.DataFrame()
            
            # Try historical data format
            elif 'historical' in data_json:
                df = pd.DataFrame(data_json['historical'])
                if 'date' in df.columns:
                    df['Date'] = pd.to_datetime(df['date'])
                    df.set_index('Date', inplace=True)
            
            else:
                print(f"Unknown response format: {list(data_json.keys())}")
                return pd.DataFrame()
            
            # Standardize column names
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'volume': 'Volume', 'adjClose': 'Adj Close', 'adj_close': 'Adj Close'
            }
            
            df.rename(columns=column_mapping, inplace=True)
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Missing required columns. Available: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Add Adj Close if missing
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            # Remove NaN values and sort
            df = df.dropna()
            df = df.sort_index(ascending=True)
            
            return df
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return pd.DataFrame()
    
    async def fetch_stock_data(self, symbol: str, period: str = None) -> pd.DataFrame:
        """
        Fetch stock price data for a given symbol from Yahoo Finance via RapidAPI
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
        
Returns:
            DataFrame with OHLCV data
        """
        if period is None:
            period = self.config.DEFAULT_PERIOD
            
        try:
            self._rate_limit()
            
            print(f"Fetching {symbol} data from apidojo Yahoo Finance API...")
            
            # Use the correct apidojo endpoint for stock chart/price data
            url = f"{self.base_url}/stock/v2/get-chart"
            
            # Map period to range parameter
            period_to_range = {
                '1d': '1d',
                '5d': '5d',
                '1mo': '1mo',
                '3mo': '3mo',
                '6mo': '6mo',
                '1y': '1y',
                '2y': '2y',
                '5y': '5y',
                '10y': '10y',
                'max': '10y'
            }
            
            range_param = period_to_range.get(period, '1y')
            
            params = {
                'symbol': symbol,
                'interval': '1d',  # Daily intervals
                'range': range_param,
                'region': 'US'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                raise Exception(f"RapidAPI request failed with status {response.status_code}: {response.text}")
            
            data_json = response.json()
            
            # Parse the apidojo chart response format
            df = self._parse_apidojo_chart_response(data_json, symbol)
            
            if df.empty:
                raise ValueError(f"No valid data found for symbol: {symbol}")
            
            print(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            print(f"RapidAPI error for {symbol}: {str(e)}")
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    async def fetch_multiple_stocks(self, symbols: List[str], period: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period for data
        
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        stock_data = {}
        
        for symbol in symbols:
            try:
                # Add delay to respect rate limits
                await asyncio.sleep(1)
                data = await self.fetch_stock_data(symbol, period)
                stock_data[symbol] = data
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        return stock_data
    
    async def fetch_benchmark_data(self, benchmark: str = None, period: str = None) -> pd.DataFrame:
        """
        Fetch benchmark data (e.g., S&P 500)
        
        Args:
            benchmark: Benchmark symbol (defaults to ^GSPC for S&P 500)
            period: Time period for data
        
        Returns:
            DataFrame with benchmark data
        """
        if benchmark is None:
            benchmark = "^GSPC"  # S&P 500 index
            
        return await self.fetch_stock_data(benchmark, period)
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data"""
        if 'Adj Close' in price_data.columns:
            return price_data['Adj Close'].pct_change().dropna()
        elif 'Close' in price_data.columns:
            return price_data['Close'].pct_change().dropna()
        else:
            raise ValueError("No price column found in data")
    
    def get_portfolio_returns(self, stock_data: Dict[str, pd.DataFrame], weights: Dict[str, float] = None) -> pd.Series:
        """Calculate portfolio returns from individual stock data and weights"""
        # If no weights provided, assume equal weights
        if weights is None:
            num_stocks = len(stock_data)
            if num_stocks == 0:
                return pd.Series()
            equal_weight = 1.0 / num_stocks
            weights = {symbol: equal_weight for symbol in stock_data.keys()}
        
        portfolio_returns = None
        
        for symbol, weight in weights.items():
            if symbol in stock_data:
                stock_returns = self.calculate_returns(stock_data[symbol])
                weighted_returns = stock_returns * weight
                
                if portfolio_returns is None:
                    portfolio_returns = weighted_returns
                else:
                    portfolio_returns = portfolio_returns.add(weighted_returns, fill_value=0)
        
        return portfolio_returns.dropna() if portfolio_returns is not None else pd.Series()
    
    def get_risk_free_rate(self) -> float:
        """
        Get risk-free rate (using config default for now)
        Could be enhanced to fetch actual treasury rates via RapidAPI
        
        Returns:
            Risk-free rate as decimal
        """
        return self.config.RISK_FREE_RATE
    
    async def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            data = await self.fetch_stock_data(symbol, "5d")
            return not data.empty
        except:
            return False 