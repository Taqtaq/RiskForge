# 🛡️ RiskForge

RiskForge is a multi-agent crypto risk analysis system built in Complete.dev.

It evaluates token risk using structured on-chain-style signals and converts that analysis into clear, human-readable investment insights.

---

## 🚀 What Problem Does It Solve?

Retail investors often struggle to interpret raw on-chain metrics such as liquidity depth, whale concentration, contract verification status, and volatility.

Most tools provide data — not structured intelligence.

RiskForge introduces a modular multi-agent architecture that bridges this gap.

---

## 🧠 Architecture Overview

### Agent 1 — Risk Analysis Engine (LangGraph Pipeline)

Evaluates:
- Liquidity depth
- Holder concentration
- Contract verification status
- Volatility patterns

Outputs:
- Structured 0–100 risk score
- Risk category (Low / Medium / High)
- Triggered risk signals

---

### Agent 2 — Explanation Agent

Consumes structured JSON and:
- Generates clear, human-readable explanations
- Produces targeted recommendations
- Adds responsible investment disclaimer
- Supports optional LLM mode

---

## 📊 Example Usage

```python
from crypto_explanation_agent import analyze_and_explain

result = analyze_and_explain("BTC")
print(result)
```
---

## 🔧 Technical Notes

- Built in Complete.dev
- Modular multi-agent architecture
- API-ready design (mock providers can be swapped with CoinGecko, Etherscan, etc.)
- Fully reproducible demo pipeline
