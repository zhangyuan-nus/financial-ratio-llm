import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from config import Config

@dataclass
class RatioResults:
    """Data class to hold ratio calculation results"""
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    treynor_ratio: Optional[float] = None
    
    # Additional metrics for context
    annualized_return: Optional[float] = None
    annualized_volatility: Optional[float] = None
    downside_deviation: Optional[float] = None
    beta: Optional[float] = None
    
    # Metadata
    period: Optional[str] = None
    risk_free_rate: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'treynor_ratio': self.treynor_ratio,
            'annualized_return': self.annualized_return,
            'annualized_volatility': self.annualized_volatility,
            'downside_deviation': self.downside_deviation,
            'beta': self.beta,
            'period': self.period,
            'risk_free_rate': self.risk_free_rate
        }

class RatioCalculator:
    """Calculates financial ratios (Sharpe, Sortino, Treynor) for investments"""
    
    def __init__(self):
        self.config = Config()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe Ratio
        
        Formula: (Portfolio Return - Risk-free Rate) / Portfolio Standard Deviation
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.config.RISK_FREE_RATE
        
        # Annualize returns and volatility
        annualized_return = self._annualize_returns(returns)
        annualized_volatility = self._annualize_volatility(returns)
        
        if annualized_volatility == 0:
            return np.inf if annualized_return > risk_free_rate else 0
        
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None, target_return: Optional[float] = None) -> float:
        """
        Calculate Sortino Ratio
        
        Formula: (Portfolio Return - Target Return) / Downside Deviation
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (used as target if target_return is None)
            target_return: Target return (defaults to risk-free rate)
        
        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.config.RISK_FREE_RATE
        
        if target_return is None:
            target_return = risk_free_rate
        
        annualized_return = self._annualize_returns(returns)
        downside_deviation = self._calculate_downside_deviation(returns, target_return)
        
        if downside_deviation == 0:
            return np.inf if annualized_return > target_return else 0
        
        sortino_ratio = (annualized_return - target_return) / downside_deviation
        return sortino_ratio
    
    def calculate_treynor_ratio(self, returns: pd.Series, market_returns: pd.Series, risk_free_rate: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate Treynor Ratio
        
        Formula: (Portfolio Return - Risk-free Rate) / Beta
        
        Args:
            returns: Series of portfolio returns
            market_returns: Series of market/benchmark returns
            risk_free_rate: Risk-free rate
        
        Returns:
            Tuple of (Treynor ratio, Beta)
        """
        if risk_free_rate is None:
            risk_free_rate = self.config.RISK_FREE_RATE
        
        # Calculate beta
        beta = self._calculate_beta(returns, market_returns)
        
        annualized_return = self._annualize_returns(returns)
        
        if beta == 0:
            return (np.inf if annualized_return > risk_free_rate else 0, beta)
        
        treynor_ratio = (annualized_return - risk_free_rate) / beta
        return treynor_ratio, beta
    
    def calculate_all_ratios(self, returns: pd.Series, market_returns: pd.Series = None, 
                           risk_free_rate: float = None, period: str = None) -> RatioResults:
        """
        Calculate all ratios at once
        
        Args:
            returns: Series of returns
            market_returns: Series of market returns (for Treynor ratio)
            risk_free_rate: Risk-free rate
            period: Time period string for metadata
        
        Returns:
            RatioResults object with all calculated ratios
        """
        if risk_free_rate is None:
            risk_free_rate = self.config.RISK_FREE_RATE
        
        results = RatioResults()
        results.risk_free_rate = risk_free_rate
        results.period = period
        
        # Calculate basic metrics
        results.annualized_return = self._annualize_returns(returns)
        results.annualized_volatility = self._annualize_volatility(returns)
        results.downside_deviation = self._calculate_downside_deviation(returns, risk_free_rate)
        
        # Calculate Sharpe ratio
        try:
            results.sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate)
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
        
        # Calculate Sortino ratio
        try:
            results.sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate)
        except Exception as e:
            print(f"Error calculating Sortino ratio: {e}")
        
        # Calculate Treynor ratio (if market data available)
        if market_returns is not None:
            try:
                treynor_ratio, beta = self.calculate_treynor_ratio(returns, market_returns, risk_free_rate)
                results.treynor_ratio = treynor_ratio
                results.beta = beta
            except Exception as e:
                print(f"Error calculating Treynor ratio: {e}")
        
        return results
    
    def _annualize_returns(self, returns: pd.Series) -> float:
        """Convert returns to annualized returns"""
        if len(returns) == 0:
            return 0
        
        # Calculate compound annual growth rate
        total_return = (1 + returns).prod() - 1
        num_periods = len(returns)
        
        # Assume daily returns, convert to annual
        periods_per_year = self.config.TRADING_DAYS_PER_YEAR
        years = num_periods / periods_per_year
        
        if years <= 0:
            return 0
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return
    
    def _annualize_volatility(self, returns: pd.Series) -> float:
        """Convert returns volatility to annualized volatility"""
        if len(returns) <= 1:
            return 0
        
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(self.config.TRADING_DAYS_PER_YEAR)
        return annualized_volatility
    
    def _calculate_downside_deviation(self, returns: pd.Series, target_return: float) -> float:
        """Calculate downside deviation for Sortino ratio"""
        if len(returns) == 0:
            return 0
        
        # Convert annual target return to daily
        daily_target = target_return / self.config.TRADING_DAYS_PER_YEAR
        
        # Calculate downside returns (returns below target)
        downside_returns = returns[returns < daily_target]
        
        if len(downside_returns) == 0:
            return 0
        
        # Calculate downside deviation
        downside_variance = ((downside_returns - daily_target) ** 2).mean()
        downside_deviation = np.sqrt(downside_variance)
        
        # Annualize
        annualized_downside_deviation = downside_deviation * np.sqrt(self.config.TRADING_DAYS_PER_YEAR)
        return annualized_downside_deviation
    
    def _calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta (systematic risk measure)"""
        if len(returns) != len(market_returns) or len(returns) <= 1:
            return 1.0  # Default beta
        
        # Align the series
        aligned_data = pd.DataFrame({'portfolio': returns, 'market': market_returns}).dropna()
        
        if len(aligned_data) <= 1:
            return 1.0
        
        portfolio_returns = aligned_data['portfolio']
        market_returns_aligned = aligned_data['market']
        
        # Calculate beta using covariance and variance
        covariance = np.cov(portfolio_returns, market_returns_aligned)[0, 1]
        market_variance = np.var(market_returns_aligned)
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
    
    def interpret_ratios(self, results: RatioResults) -> Dict[str, str]:
        """
        Provide interpretations for calculated ratios
        
        Args:
            results: RatioResults object
        
        Returns:
            Dictionary with ratio interpretations
        """
        interpretations = {}
        
        # Sharpe ratio interpretation
        if results.sharpe_ratio is not None:
            if results.sharpe_ratio > 2:
                interpretations['sharpe'] = "Excellent risk-adjusted performance"
            elif results.sharpe_ratio > 1:
                interpretations['sharpe'] = "Good risk-adjusted performance"
            elif results.sharpe_ratio > 0:
                interpretations['sharpe'] = "Positive risk-adjusted performance"
            else:
                interpretations['sharpe'] = "Poor risk-adjusted performance"
        
        # Sortino ratio interpretation
        if results.sortino_ratio is not None:
            if results.sortino_ratio > 2:
                interpretations['sortino'] = "Excellent downside risk-adjusted performance"
            elif results.sortino_ratio > 1:
                interpretations['sortino'] = "Good downside risk-adjusted performance"
            elif results.sortino_ratio > 0:
                interpretations['sortino'] = "Positive downside risk-adjusted performance"
            else:
                interpretations['sortino'] = "Poor downside risk-adjusted performance"
        
        # Treynor ratio interpretation
        if results.treynor_ratio is not None:
            if results.treynor_ratio > 0.1:
                interpretations['treynor'] = "Good systematic risk-adjusted performance"
            elif results.treynor_ratio > 0:
                interpretations['treynor'] = "Positive systematic risk-adjusted performance"
            else:
                interpretations['treynor'] = "Poor systematic risk-adjusted performance"
        
        # Beta interpretation
        if results.beta is not None:
            if results.beta > 1.2:
                interpretations['beta'] = "High systematic risk (more volatile than market)"
            elif results.beta > 0.8:
                interpretations['beta'] = "Moderate systematic risk (similar to market)"
            elif results.beta > 0:
                interpretations['beta'] = "Low systematic risk (less volatile than market)"
            else:
                interpretations['beta'] = "Negative correlation with market"
        
        return interpretations 