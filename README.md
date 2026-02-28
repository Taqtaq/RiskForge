# 🛡️ RiskForge

RiskForge is a multi-agent crypto risk analysis system built in Complete.dev.

It transforms fragmented on-chain-style signals into a structured 0–100 risk score and a clear, human-readable explanation with actionable recommendations.

---

## 🚀 Problem

Retail investors are exposed to raw blockchain metrics such as liquidity depth, whale concentration, contract verification flags, and volatility.

However, most tools provide data — not structured intelligence.

RiskForge bridges this gap through a modular multi-agent architecture that evaluates risk dimensions and explains them in plain language.

---

## 🧠 Architecture Overview

RiskForge uses two collaborating agents.

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

Consumes structured JSON from the Risk Engine and:

- Generates clear, human-readable explanations
- Produces targeted recommendations based on detected signals
- Adds responsible investment disclaimer
- Supports optional LLM mode (via environment credentials)

---

## 🛠 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/RiskForge.git
cd RiskForge
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(Optional) Enable LLM mode:

```bash
export CLIENT_ID=your_client_id
export CLIENT_SECRET=your_client_secret
```

On Windows:

```bash
set CLIENT_ID=your_client_id
set CLIENT_SECRET=your_client_secret
```

---

## ▶️ Run Demo

From terminal:

```bash
python crypto_explanation_agent.py
```

Or inside Python:

```python
from crypto_explanation_agent import analyze_and_explain

result = analyze_and_explain("BTC")
print(result)
```

---

## 📊 Example Output

### BTC (Low Risk)
- High liquidity
- Distributed holder base
- Verified contract
- Stable volatility
- Risk score: Low

### PEPE (High Risk)
- Lower liquidity
- High whale concentration
- Unverified contract flags
- Extreme volatility
- Risk score: High

---

## 🔧 Technical Highlights

- Built using LangGraph multi-agent workflow
- Modular and API-ready architecture
- Mock providers replaceable with real APIs (CoinGecko, Etherscan, Glassnode)
- Fully reproducible demo pipeline
- Separation of scoring logic and explanation logic

---

## 📈 Scalability Potential

RiskForge can scale into:

- Web-based investor dashboard
- Wallet risk scanner
- Portfolio-level risk analyzer
- DeFi protocol evaluation engine
- Institutional crypto risk monitoring tool

---

## 📜 License

MIT License