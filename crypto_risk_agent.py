"""
Crypto Risk Analysis Agent — LangGraph Implementation
======================================================
Analyzes a token for four risk dimensions:
  1. Liquidity Risk
  2. Holder Concentration Risk
  3. Contract Verification Risk
  4. Volatility Risk

Data layer uses mock data by default.
Swap MockDataProvider methods to integrate CoinGecko, Glassnode, Etherscan, etc.
"""

import json
import random
import logging
import os
from datetime import datetime, timezone
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END

# ─── Logging Setup ────────────────────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "crypto_risk_agent.json")),
        logging.StreamHandler(),
    ],
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger("crypto_risk_agent")


# ─── State Definition ─────────────────────────────────────────────────────────

class RiskSignal(TypedDict):
    type: str
    severity: str        # "low" | "medium" | "high"
    description: str
    value: float


class CryptoRiskState(TypedDict):
    token: str
    raw_data: dict
    liquidity_risk: Optional[dict]
    concentration_risk: Optional[dict]
    contract_risk: Optional[dict]
    volatility_risk: Optional[dict]
    signals: List[RiskSignal]
    risk_score: float
    risk_category: str
    output: dict


# ─── Mock Data Provider ───────────────────────────────────────────────────────

class MockDataProvider:
    """
    Simulates on-chain and market data.

    To integrate real APIs, replace each method:
      - get_liquidity_data(token) → CoinGecko  /coins/{id}/market_chart
      - get_holder_data(token)    → Glassnode   /v1/metrics/distribution/balance_1pct
      - get_contract_data(token)  → Etherscan   /api?module=contract&action=getsourcecode
      - get_ohlcv_data(token)     → CoinGecko   /coins/{id}/ohlc or Binance /api/v3/klines
    """

    _DB: dict = {
        "BTC": {
            "liquidity_usd": 2_500_000_000,
            "volume_24h": 30_000_000_000,
            "top10_holder_pct": 8.5,
            "is_contract_verified": True,
            "is_proxy": False,
            "price_changes_7d": [0.5, -1.2, 2.1, -0.8, 3.5, -2.3, 1.1],
        },
        "ETH": {
            "liquidity_usd": 800_000_000,
            "volume_24h": 15_000_000_000,
            "top10_holder_pct": 12.3,
            "is_contract_verified": True,
            "is_proxy": True,
            "price_changes_7d": [1.2, -2.5, 3.8, -1.5, 4.2, -3.1, 2.0],
        },
        "PEPE": {
            "liquidity_usd": 1_200_000,
            "volume_24h": 45_000_000,
            "top10_holder_pct": 52.7,
            "is_contract_verified": False,
            "is_proxy": False,
            "price_changes_7d": [15.2, -22.5, 38.8, -18.5, 42.2, -35.1, 28.0],
        },
        "SOL": {
            "liquidity_usd": 120_000_000,
            "volume_24h": 2_500_000_000,
            "top10_holder_pct": 33.4,
            "is_contract_verified": True,
            "is_proxy": False,
            "price_changes_7d": [3.1, -5.2, 7.4, -4.8, 8.1, -6.2, 4.5],
        },
    }

    def get_data(self, token: str) -> dict:
        key = token.upper()
        if key in self._DB:
            return dict(self._DB[key])
        # Unknown token → randomized mock (higher risk profile)
        logger.warning(f"Unknown token '{key}' — generating randomized mock data")
        return {
            "liquidity_usd": random.uniform(50_000, 5_000_000),
            "volume_24h": random.uniform(100_000, 20_000_000),
            "top10_holder_pct": random.uniform(30, 85),
            "is_contract_verified": random.choice([True, False]),
            "is_proxy": random.choice([True, False]),
            "price_changes_7d": [random.uniform(-40, 40) for _ in range(7)],
        }


# ─── Node: Fetch Data ─────────────────────────────────────────────────────────

def fetch_data_node(state: CryptoRiskState) -> CryptoRiskState:
    logger.info(f"Fetching data for token: {state['token']}")
    provider = MockDataProvider()
    state["raw_data"] = provider.get_data(state["token"])
    state["signals"] = []
    return state


# ─── Node: Liquidity Risk ─────────────────────────────────────────────────────

def liquidity_risk_node(state: CryptoRiskState) -> CryptoRiskState:
    """
    Evaluates available on-chain / DEX liquidity.
    Swap trigger: replace MockDataProvider.get_liquidity_data() with CoinGecko pools API.
    """
    liquidity = state["raw_data"]["liquidity_usd"]
    volume = state["raw_data"]["volume_24h"]
    liquidity_ratio = round(liquidity / volume, 4) if volume > 0 else 0

    if liquidity < 500_000:
        score, severity = 90, "high"
        desc = f"Critically low liquidity: ${liquidity:,.0f}"
    elif liquidity < 5_000_000:
        score, severity = 65, "medium"
        desc = f"Low liquidity: ${liquidity:,.0f}"
    elif liquidity < 50_000_000:
        score, severity = 30, "low"
        desc = f"Moderate liquidity: ${liquidity:,.0f}"
    else:
        score, severity = 5, "low"
        desc = f"Strong liquidity: ${liquidity:,.0f}"

    state["liquidity_risk"] = {
        "score": score,
        "liquidity_usd": liquidity,
        "liquidity_to_volume_ratio": liquidity_ratio,
    }

    if severity in ("medium", "high"):
        state["signals"].append({
            "type": "liquidity_risk",
            "severity": severity,
            "description": desc,
            "value": liquidity,
        })

    logger.info(f"[{state['token']}] Liquidity risk score: {score} ({severity})")
    return state


# ─── Node: Holder Concentration Risk ─────────────────────────────────────────

def concentration_risk_node(state: CryptoRiskState) -> CryptoRiskState:
    """
    Evaluates supply concentration among top holders.
    Swap trigger: replace with Glassnode /v1/metrics/distribution/balance_1pct.
    """
    top10_pct = state["raw_data"]["top10_holder_pct"]

    if top10_pct > 50:
        score, severity = 90, "high"
        desc = f"Extreme concentration: top 10 holders own {top10_pct:.1f}% of supply"
    elif top10_pct > 30:
        score, severity = 60, "medium"
        desc = f"Elevated concentration: top 10 holders own {top10_pct:.1f}%"
    elif top10_pct > 15:
        score, severity = 30, "low"
        desc = f"Moderate concentration: top 10 holders own {top10_pct:.1f}%"
    else:
        score, severity = 5, "low"
        desc = f"Well-distributed supply: top 10 at {top10_pct:.1f}%"

    state["concentration_risk"] = {
        "score": score,
        "top10_holder_pct": top10_pct,
    }

    if severity in ("medium", "high"):
        state["signals"].append({
            "type": "holder_concentration_risk",
            "severity": severity,
            "description": desc,
            "value": top10_pct,
        })

    logger.info(f"[{state['token']}] Concentration risk score: {score} ({severity})")
    return state


# ─── Node: Contract Verification Risk ────────────────────────────────────────

def contract_risk_node(state: CryptoRiskState) -> CryptoRiskState:
    """
    Checks contract verification status and proxy pattern usage.
    Swap trigger: replace with Etherscan API /api?module=contract&action=getsourcecode.
    """
    is_verified = state["raw_data"]["is_contract_verified"]
    is_proxy = state["raw_data"]["is_proxy"]
    score = 0

    if not is_verified:
        score += 55
        state["signals"].append({
            "type": "contract_not_verified",
            "severity": "high",
            "description": "Contract source code is NOT verified on-chain — cannot audit logic",
            "value": 1.0,
        })

    if is_proxy:
        score += 25
        state["signals"].append({
            "type": "proxy_contract_detected",
            "severity": "medium",
            "description": "Proxy pattern detected — contract logic can be silently upgraded",
            "value": 1.0,
        })

    score = min(score, 100)
    state["contract_risk"] = {
        "score": score,
        "is_verified": is_verified,
        "is_proxy": is_proxy,
    }

    logger.info(f"[{state['token']}] Contract risk score: {score}")
    return state


# ─── Node: Volatility Risk ────────────────────────────────────────────────────

def volatility_risk_node(state: CryptoRiskState) -> CryptoRiskState:
    """
    Computes average absolute daily price change over 7 days.
    Swap trigger: replace with CoinGecko /coins/{id}/ohlc or Binance klines API.
    """
    changes = state["raw_data"]["price_changes_7d"]
    avg_abs_change = round(sum(abs(c) for c in changes) / len(changes), 2)

    if avg_abs_change > 20:
        score, severity = 90, "high"
        desc = f"Extreme volatility: avg daily move {avg_abs_change:.1f}%"
    elif avg_abs_change > 10:
        score, severity = 65, "medium"
        desc = f"High volatility: avg daily move {avg_abs_change:.1f}%"
    elif avg_abs_change > 3:
        score, severity = 30, "low"
        desc = f"Moderate volatility: avg daily move {avg_abs_change:.1f}%"
    else:
        score, severity = 10, "low"
        desc = f"Low volatility: avg daily move {avg_abs_change:.1f}%"

    state["volatility_risk"] = {
        "score": score,
        "avg_daily_change_pct": avg_abs_change,
    }

    if severity in ("medium", "high"):
        state["signals"].append({
            "type": "volatility_risk",
            "severity": severity,
            "description": desc,
            "value": avg_abs_change,
        })

    logger.info(f"[{state['token']}] Volatility risk score: {score} ({severity})")
    return state


# ─── Node: Aggregate Risk ─────────────────────────────────────────────────────

def aggregate_risk_node(state: CryptoRiskState) -> CryptoRiskState:
    """
    Weighted aggregation of all risk dimension scores into a final 0–100 score.

    Weights (tunable):
      - Liquidity:     30%
      - Concentration: 25%
      - Contract:      25%
      - Volatility:    20%
    """
    WEIGHTS = {
        "liquidity":     0.30,
        "concentration": 0.25,
        "contract":      0.25,
        "volatility":    0.20,
    }

    raw_score = (
        state["liquidity_risk"]["score"]    * WEIGHTS["liquidity"]
        + state["concentration_risk"]["score"] * WEIGHTS["concentration"]
        + state["contract_risk"]["score"]      * WEIGHTS["contract"]
        + state["volatility_risk"]["score"]    * WEIGHTS["volatility"]
    )

    risk_score = round(min(max(raw_score, 0), 100), 2)

    if risk_score >= 65:
        risk_category = "High"
    elif risk_score >= 35:
        risk_category = "Medium"
    else:
        risk_category = "Low"

    state["risk_score"]    = risk_score
    state["risk_category"] = risk_category

    state["output"] = {
        "token": state["token"].upper(),
        "risk_score": risk_score,
        "risk_category": risk_category,
        "signals": state["signals"],
        "breakdown": {
            "liquidity_risk":     state["liquidity_risk"],
            "concentration_risk": state["concentration_risk"],
            "contract_risk":      state["contract_risk"],
            "volatility_risk":    state["volatility_risk"],
        },
        "weights_used": WEIGHTS,
        "metadata": {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "mock",
            "agent_version": "1.0.0",
        },
    }

    logger.info(
        f"[{state['token']}] Final risk score: {risk_score} — Category: {risk_category} "
        f"— Signals detected: {len(state['signals'])}"
    )
    return state


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_crypto_risk_graph():
    """
    Assembles the LangGraph StateGraph for crypto risk analysis.

    Flow:
      fetch_data → liquidity_risk → concentration_risk → contract_risk
                 → volatility_risk → aggregate_risk → END

    Extension points:
      - Add 'sentiment_risk' node after volatility_risk
      - Add 'on_chain_activity' node for tx volume anomaly detection
      - Add LLM 'explain_risk' node at the end for natural-language summaries
      - Enable parallel execution of dimension nodes via LangGraph's Send API
    """
    graph = StateGraph(CryptoRiskState)

    graph.add_node("fetch_data",         fetch_data_node)
    graph.add_node("liquidity_risk",     liquidity_risk_node)
    graph.add_node("concentration_risk", concentration_risk_node)
    graph.add_node("contract_risk",      contract_risk_node)
    graph.add_node("volatility_risk",    volatility_risk_node)
    graph.add_node("aggregate_risk",     aggregate_risk_node)

    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data",         "liquidity_risk")
    graph.add_edge("liquidity_risk",     "concentration_risk")
    graph.add_edge("concentration_risk", "contract_risk")
    graph.add_edge("contract_risk",      "volatility_risk")
    graph.add_edge("volatility_risk",    "aggregate_risk")
    graph.add_edge("aggregate_risk",     END)

    return graph.compile()


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_token(token: str) -> dict:
    """
    Main entry point. Accepts a token symbol, returns structured risk JSON.

    Args:
        token (str): Cryptocurrency symbol, e.g. "BTC", "ETH", "PEPE"

    Returns:
        dict: {
            token, risk_score, risk_category, signals, breakdown, metadata
        }
    """
    app = build_crypto_risk_graph()

    initial_state: CryptoRiskState = {
        "token": token.upper(),
        "raw_data": {},
        "liquidity_risk": None,
        "concentration_risk": None,
        "contract_risk": None,
        "volatility_risk": None,
        "signals": [],
        "risk_score": 0.0,
        "risk_category": "",
        "output": {},
    }

    result = app.invoke(initial_state)
    return result["output"]


# ─── CLI Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_tokens = ["BTC", "ETH", "PEPE", "SOL", "UNKNOWN"]

    for token in test_tokens:
        print(f"\n{'═' * 62}")
        print(f"  CRYPTO RISK ANALYSIS — {token}")
        print(f"{'═' * 62}")
        result = analyze_token(token)
        print(json.dumps(result, indent=2))
