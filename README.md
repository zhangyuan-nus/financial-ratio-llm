# Financial Ratio LLM System

A tool-augmented LLM system that generates Sharpe, Sortino, and Treynor ratios for any financial instrument with natural language explanations.

## Features

- **Natural Language Processing**: Parse user prompts to understand what financial ratios to calculate
- **Real-time Data Fetching**: Automatically fetch financial data from various sources
- **Ratio Calculations**: Compute Sharpe, Sortino, and Treynor ratios
- **Natural Language Explanations**: Generate human-readable explanations of the results

## Architecture

```
User Prompt → LLM Parser → Data Fetcher → Ratio Calculator → Natural Language Response
```

## Setup

### Option 1: Free Local LLM (Recommended)

#### Step 1: Install Ollama
- **Windows**: Download from https://ollama.com/download and run the installer
- **Mac**: `brew install ollama` or download from https://ollama.com/download
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`

#### Step 2: Download a Model
Choose based on your hardware:

**Best Quality: Llama 3.3 70B** (48GB+ RAM required)
```bash
ollama pull llama3.3:70b
```

**Good Balance: Llama 3.1 8B** (8GB+ RAM required)
```bash
ollama pull llama3.1:8b
```

**Fastest: Mistral 7B** (4GB+ RAM required)
```bash
ollama pull mistral:7b
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Create `.env` file
```bash
# Ollama Configuration (No API key needed!)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Data Source Configuration
DATA_SOURCE=rapidapi
RAPIDAPI_KEY=your_rapidapi_key_here

# Financial Configuration
RISK_FREE_RATE=0.05
DEFAULT_BENCHMARK=^GSPC
DEFAULT_PERIOD=1y
TRADING_DAYS_PER_YEAR=252
```

#### Step 5: Start Ollama Server
```bash
# Start Ollama (runs in background)
ollama serve
```
*Note: On Windows/Mac, Ollama usually starts automatically after installation.*

#### Step 6: Test Your Setup
```bash
# Test Ollama directly
ollama run llama3.1:8b "What is the Sharpe ratio?"

# Test the financial system
python main.py --query "Calculate Sharpe ratio for AAPL"
```

### Option 2: OpenAI API (Paid)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   RISK_FREE_RATE=0.05
   DEFAULT_BENCHMARK=^GSPC
   DEFAULT_PERIOD=1y
   OPENAI_MODEL=gpt-4
   ```

## Analysis Types

The system supports three main types of financial analysis:

### 🔍 **Single Stock Analysis**
Analyze individual stocks with comprehensive ratio calculations and interpretations.
- **Example**: "Calculate all ratios for AAPL"
- **Output**: Sharpe, Sortino, Treynor ratios with detailed explanations

### 📊 **Stock Comparison**
Compare multiple stocks side-by-side to identify the best performers.
- **Example**: "Compare Sharpe ratios between AAPL and MSFT"
- **Output**: Individual analysis of each stock + comparative insights

### 💼 **Portfolio Analysis**
Analyze weighted portfolios with both portfolio-level and individual stock metrics.
- **Example**: "Analyze my portfolio of 60% AAPL and 40% MSFT"
- **Output**: Portfolio ratios + individual stock analysis + rebalancing insights

## Usage

### Interactive Mode
```bash
python main.py
```
Then ask questions like:

**Single Stock Analysis:**
- "Calculate the Sharpe ratio for AAPL"
- "What are all the ratios for Tesla stock?"
- "Analyze NVDA's risk-adjusted performance"

**Stock Comparison:**
- "Compare Sharpe ratios between AAPL and MSFT"
- "What are the ratios for AAPL and TSLA?"
- "Compare the risk-adjusted performance of SPY vs QQQ"

**Portfolio Analysis:**
- "Analyze my portfolio of 60% AAPL and 40% MSFT"
- "Calculate ratios for my 70% SPY, 30% QQQ portfolio"
- "What's the Sharpe ratio of my equal-weight portfolio with TSLA, MSFT, and GOOGL?"

**Advanced Queries:**
- "Compare Treynor ratios between tech stocks over 2 years"
- "How does my 50-50 AAPL-MSFT portfolio perform against the market?"



### Single Query Mode
```bash
# Single stock analysis
python main.py --query "Calculate all ratios for AAPL"

# Stock comparison
python main.py --query "Compare Sharpe ratios of AAPL and MSFT"

# Portfolio analysis
python main.py --query "Analyze my portfolio of 60% AAPL and 40% MSFT"
```

## Supported Ratios

- **Sharpe Ratio**: Risk-adjusted return measure
  - Formula: (Portfolio Return - Risk-free Rate) / Portfolio Standard Deviation
  - Measures excess return per unit of total risk
  
- **Sortino Ratio**: Downside risk-adjusted return
  - Formula: (Portfolio Return - Target Return) / Downside Deviation
  - Focuses only on downside volatility, ignoring upside volatility
  
- **Treynor Ratio**: Systematic risk-adjusted return
  - Formula: (Portfolio Return - Risk-free Rate) / Beta
  - Measures excess return per unit of systematic risk

## Ollama Model Comparison

| Model | Quality | Speed | RAM Usage | Best For |
|-------|---------|-------|-----------|----------|
| Llama 3.3 70B | ⭐⭐⭐⭐⭐ | ⭐⭐ | 48GB+ | Best results |
| Llama 3.1 8B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB+ | Balanced |
| Mistral 7B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4GB+ | Speed |

## Troubleshooting

### "Connection refused" Error
```bash
# Make sure Ollama is running
ollama serve

# Check if it's running
curl http://localhost:11434/api/tags
```

### Model Not Found
```bash
# List available models
ollama list

# Pull the model if missing
ollama pull llama3.1:8b
```

### Out of Memory
```bash
# Switch to smaller model in your .env file
OLLAMA_MODEL=mistral:7b
```

### Slow Performance
- Use a smaller model (`mistral:7b`)
- Close other applications
- Ensure you have enough RAM
- Consider using GPU acceleration if available

## Ollama vs OpenAI Comparison

| Feature | Ollama (Free) | OpenAI (Paid) |
|---------|---------------|---------------|
| **Cost** | $0 forever | $0.01-0.05 per query |
| **Privacy** | 100% local | Data sent to OpenAI |
| **Internet** | Not required | Required |
| **Setup** | One-time install | API key needed |
| **Performance** | Good-Excellent | Excellent |
| **Speed** | Depends on hardware | Fast |

## Advanced Configuration

### Custom Ollama Settings
```bash
# Set custom host/port
export OLLAMA_HOST=0.0.0.0:11435

# Use GPU acceleration (if available)
export OLLAMA_GPU=1
```

### Multiple Models
```bash
# Download multiple models for different use cases
ollama pull llama3.1:8b      # Balanced
ollama pull mistral:7b       # Fast
ollama pull codellama:7b     # Code-focused
```

## Features Summary

✅ **Natural Language Processing**: Ask questions in plain English  
✅ **Real-time Data**: Fetches live market data via RapidAPI (Yahoo Finance)  
✅ **Three Analysis Types**: Single stocks, stock comparisons, and portfolio analysis  
✅ **Smart Portfolio Handling**: Automatic equal weights or custom weight specifications  
✅ **Stock Comparison**: Side-by-side analysis of multiple stocks  
✅ **Multiple Time Periods**: Analyze over various time horizons (1d to 10y)  
✅ **Custom Benchmarks**: Compare against different market indices  
✅ **Command Line Interface**: Easy-to-use CLI for local analysis  
✅ **Comprehensive Explanations**: AI-generated insights and interpretations  
✅ **Free & Private**: Run completely locally with Ollama  
✅ **No API Costs**: No recurring charges or usage limits  

## Tips for Best Results

1. **Start Small**: Begin with `mistral:7b` to test, then upgrade
2. **Monitor Resources**: Use Task Manager/Activity Monitor to check RAM usage
3. **Close Other Apps**: Free up RAM for better performance
4. **Be Patient**: First run may be slower as model loads
5. **Experiment**: Try different models to find your sweet spot