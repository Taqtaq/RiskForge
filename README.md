# RiskForge

RiskForge is a multi-agent crypto risk analysis system built in Complete.dev.

## Architecture

### Agent 1 – Risk Analysis Engine
- Liquidity scoring
- Holder concentration analysis
- Contract verification checks
- Volatility detection
- Produces structured 0–100 risk score

### Agent 2 – Explanation Agent
- Consumes structured JSON
- Generates human-readable explanation
- Produces targeted recommendations
- Adds responsible disclaimer

## Usage

from crypto_explanation_agent import analyze_and_explain

result = analyze_and_explain("BTC")
print(result)

## Notes

Designed as a modular, API-ready multi-agent system.
Mock providers can be replaced with real APIs (CoinGecko, Etherscan, etc.).