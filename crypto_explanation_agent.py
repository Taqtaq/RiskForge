"""
Crypto Risk Explanation Agent — LangGraph Implementation
=========================================================
Takes structured JSON output from crypto_risk_agent.py and produces:
  1. Plain-text explanation for retail investors
  2. 3 bullet-point recommendations
  3. Risk disclaimer

LLM: Deploy AI (GPT-4o) when CLIENT_ID / CLIENT_SECRET are set in os.environ.
Fallback: Template-based generation when credentials are unavailable.
"""

import os
import re
import json
import logging
import requests
from typing import TypedDict, List, Optional
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END

from crypto_risk_agent import analyze_token

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "risk_explanation_agent.json")),
        logging.StreamHandler(),
    ],
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger("risk_explanation_agent")


# ─── State ────────────────────────────────────────────────────────────────────

class ExplanationState(TypedDict):
    risk_report:      dict
    prompt:           str
    raw_llm_response: str
    explanation:      str
    recommendations:  List[str]
    disclaimer:       str
    use_llm:          bool
    final_output:     dict


# ─── Deploy AI Client ─────────────────────────────────────────────────────────

_AUTH_URL = "https://api-auth.deploy.ai/oauth2/token"
_API_URL  = "https://core-api.deploy.ai"
_ORG_ID   = "85ba5044-2ddb-4e23-b994-de643188a875"


def _get_access_token() -> Optional[str]:
    client_id     = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    try:
        resp = requests.post(
            _AUTH_URL,
            data={
                "grant_type":    "client_credentials",
                "client_id":     client_id,
                "client_secret": client_secret,
            },
            timeout=10,
        )
        return resp.json().get("access_token")
    except Exception as e:
        logger.warning(f"Access token error: {e}")
        return None


def _create_chat(access_token: str) -> Optional[str]:
    try:
        resp = requests.post(
            f"{_API_URL}/chats",
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-Org":         _ORG_ID,
                "Content-Type":  "application/json",
            },
            json={"agentId": "GPT_4O", "stream": False},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()["id"]
        logger.warning(f"Chat creation failed: {resp.status_code} {resp.text}")
        return None
    except Exception as e:
        logger.warning(f"Chat creation error: {e}")
        return None


def _call_llm(access_token: str, chat_id: str, prompt: str) -> Optional[str]:
    try:
        resp = requests.post(
            f"{_API_URL}/messages",
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-Org":         _ORG_ID,
                "Content-Type":  "application/json",
            },
            json={
                "chatId":  chat_id,
                "stream":  False,
                "content": [{"type": "text", "value": prompt}],
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["value"]
        logger.warning(f"LLM call failed: {resp.status_code}")
        return None
    except Exception as e:
        logger.warning(f"LLM call error: {e}")
        return None


# ─── Template Engine (fallback) ───────────────────────────────────────────────

# Human-readable signal explanations by type + severity
_SIGNAL_EXPLANATIONS: dict = {
    "liquidity_risk": {
        "medium": (
            "The token has relatively limited trading liquidity, meaning it may be harder "
            "to buy or sell without moving the price. This is more noticeable with larger amounts."
        ),
        "high": (
            "This token has very low liquidity. Selling even a modest position could cause "
            "the price to drop sharply, and finding a willing buyer during a market downturn may be difficult."
        ),
    },
    "holder_concentration_risk": {
        "medium": (
            "A notable portion of the token's total supply is held by a small group of wallets. "
            "These large holders can have a meaningful impact on the token's price if they decide to sell."
        ),
        "high": (
            "A very small number of wallets control the majority of this token's supply. "
            "If any of these 'whale' wallets decides to sell, it could cause a rapid and dramatic price drop."
        ),
    },
    "contract_not_verified": {
        "high": (
            "The smart contract powering this token has not been publicly verified. "
            "Without verification, independent experts cannot review the code for hidden risks, bugs, or malicious features."
        ),
    },
    "proxy_contract_detected": {
        "medium": (
            "This token runs on an upgradeable smart contract. While that allows developers to fix issues, "
            "it also means the underlying rules governing the token could be changed at any time without your consent."
        ),
    },
    "volatility_risk": {
        "medium": (
            "The token has shown significant price swings in recent days. "
            "While this can create profit opportunities, it also means your investment could lose value quickly."
        ),
        "high": (
            "This token is extremely volatile, with very large daily price swings observed. "
            "Such extreme volatility means the value of your investment can drop sharply in a very short period."
        ),
    },
}

# Recommendations mapped to each signal type
_SIGNAL_RECOMMENDATIONS: dict = {
    "liquidity_risk": [
        "Avoid committing large amounts at once — low liquidity can make it difficult to exit your position when needed.",
        "Use limit orders instead of market orders to avoid paying an unfair price due to low trading volume.",
    ],
    "holder_concentration_risk": [
        "Monitor large wallet activity using on-chain explorer tools — sudden sell-offs by major holders can sharply impact price.",
        "Be mindful of the concentration risk and consider limiting your position size accordingly.",
    ],
    "contract_not_verified": [
        "Only invest what you can afford to lose entirely — unverified contracts carry unknown and potentially serious code risks.",
        "Look for independent security audit reports from trusted firms before considering any investment.",
    ],
    "proxy_contract_detected": [
        "Stay up to date with official project announcements, especially regarding any contract upgrades or changes.",
        "Understand that the token's behavior could change after upgrades — make sure you trust the development team.",
    ],
    "volatility_risk": [
        "Consider using stop-loss orders to cap your downside exposure given the high price fluctuations.",
        "Avoid investing money you may need in the near future — sharp drops can happen suddenly and without warning.",
    ],
}

# Default recommendations when no signals match or to pad to 3
_DEFAULT_RECOMMENDATIONS: dict = {
    "Low": [
        "Periodically reassess the token's risk profile — market conditions and project status can change over time.",
        "Diversify your holdings — even low-risk tokens are subject to broader crypto market volatility.",
        "Stay informed about the project's development activity, team updates, and any changes to tokenomics.",
    ],
    "Medium": [
        "Limit your exposure to a portion of your portfolio that you are comfortable potentially losing.",
        "Set price alerts to stay aware of any unusual market movements around this token.",
        "Research the project team, roadmap, and community engagement before making any investment decision.",
    ],
    "High": [
        "Proceed with extreme caution — the multiple risk signals detected suggest this token carries substantial risk.",
        "If you choose to engage with this token, only use funds you can afford to lose entirely.",
        "Consider waiting for risk factors to improve — such as contract verification or better liquidity — before investing.",
    ],
}

_DISCLAIMER = (
    "This analysis is generated by an automated AI system for educational purposes only "
    "and does not constitute financial advice. Cryptocurrency investments carry significant risk, "
    "and past performance is not indicative of future results. Always conduct your own research "
    "and consult a qualified financial adviser before making investment decisions."
)


def _build_template_explanation(
    risk_report: dict,
) -> tuple:
    """Generates explanation, recommendations, and disclaimer using templates."""
    token    = risk_report["token"]
    score    = risk_report["risk_score"]
    category = risk_report["risk_category"]
    signals  = risk_report["signals"]

    # ── Intro sentence ──
    intro = (
        f"{token} received an overall risk score of {score} out of 100, "
        f"placing it in the {category} risk category."
    )

    # ── Signal descriptions ──
    if not signals:
        body = (
            "No significant risk signals were detected. "
            "The token shows healthy liquidity, a well-distributed token supply, "
            "a verified smart contract, and relatively stable price behavior."
        )
    else:
        sentences = []
        for sig in signals:
            sig_type = sig["type"]
            severity = sig["severity"]
            bank = _SIGNAL_EXPLANATIONS.get(sig_type, {})
            if severity in bank:
                sentences.append(bank[severity])
            elif bank:
                sentences.append(next(iter(bank.values())))
        body = " ".join(sentences)

    # ── Closing sentence ──
    if category == "Low":
        closing = (
            f"Overall, {token} presents a relatively low risk profile compared "
            f"to many tokens in the crypto market."
        )
    elif category == "Medium":
        closing = (
            f"Investors should carefully evaluate these risk factors and ensure "
            f"they are comfortable with the risks before committing funds to {token}."
        )
    else:
        closing = (
            f"Given the multiple risk factors detected, {token} should be approached "
            f"with significant caution and thorough research."
        )

    explanation = f"{intro} {body} {closing}"

    # ── Recommendations ──
    recs: List[str] = []
    for sig in signals:
        sig_type = sig["type"]
        if sig_type in _SIGNAL_RECOMMENDATIONS:
            for r in _SIGNAL_RECOMMENDATIONS[sig_type]:
                if r not in recs and len(recs) < 3:
                    recs.append(r)

    # Pad to 3 with category defaults
    defaults = _DEFAULT_RECOMMENDATIONS.get(category, _DEFAULT_RECOMMENDATIONS["Low"])
    for d in defaults:
        if len(recs) >= 3:
            break
        if d not in recs:
            recs.append(d)

    return explanation, recs[:3], _DISCLAIMER


def _build_llm_prompt(risk_report: dict) -> str:
    """Builds a structured prompt for the Deploy AI LLM."""
    token     = risk_report["token"]
    score     = risk_report["risk_score"]
    category  = risk_report["risk_category"]
    signals   = risk_report["signals"]
    breakdown = risk_report["breakdown"]

    liq  = breakdown["liquidity_risk"]
    conc = breakdown["concentration_risk"]
    cont = breakdown["contract_risk"]
    vol  = breakdown["volatility_risk"]

    signal_lines = (
        "\n".join(
            f"  - [{s['severity'].upper()}] {s['description']}"
            for s in signals
        )
        if signals
        else "  - No significant risk signals detected."
    )

    return f"""You are a neutral, educational crypto risk analyst explaining analysis results to retail investors with no technical background.

RULES:
- Use clear, jargon-free language that a non-expert can easily understand.
- Translate technical terms into plain English (e.g. "liquidity" = "how easy it is to buy or sell").
- Be neutral and informative. For Medium or High risk tokens, be slightly cautionary but never alarmist.
- Never give direct financial advice. Do not tell users to buy, sell, or hold.
- Do not repeat raw field names or numbers directly from the report — interpret them in plain language.

RISK REPORT FOR {token}:
- Overall Risk Score: {score}/100
- Risk Category: {category}
- Liquidity Available: ${liq['liquidity_usd']:,.0f} (risk score: {liq['score']}/100)
- Top 10 Holder Share: {conc['top10_holder_pct']}% of total supply (risk score: {conc['score']}/100)
- Smart Contract: Verified={cont['is_verified']}, Upgradeable={cont['is_proxy']} (risk score: {cont['score']}/100)
- Price Volatility: Avg daily move = {vol['avg_daily_change_pct']}% (risk score: {vol['score']}/100)

Detected Risk Signals:
{signal_lines}

Respond in EXACTLY this format — no extra text before EXPLANATION: or after the disclaimer:

EXPLANATION:
[3–4 sentence plain-language summary of what these results mean for a retail investor]

RECOMMENDATIONS:
• [recommendation 1]
• [recommendation 2]
• [recommendation 3]

DISCLAIMER:
[one sentence risk disclaimer]"""


def _parse_llm_response(response: str) -> tuple:
    """Parses structured LLM response into (explanation, recommendations, disclaimer)."""
    explanation     = ""
    recommendations = []
    disclaimer      = ""

    try:
        exp_match = re.search(
            r"EXPLANATION:\s*(.+?)(?=RECOMMENDATIONS:|$)", response, re.DOTALL
        )
        if exp_match:
            explanation = exp_match.group(1).strip()

        rec_match = re.search(
            r"RECOMMENDATIONS:\s*(.+?)(?=DISCLAIMER:|$)", response, re.DOTALL
        )
        if rec_match:
            bullets = re.findall(r"[•\-\*]\s*(.+)", rec_match.group(1))
            recommendations = [b.strip() for b in bullets[:3]]

        disc_match = re.search(r"DISCLAIMER:\s*(.+?)$", response, re.DOTALL)
        if disc_match:
            disclaimer = disc_match.group(1).strip()
    except Exception as e:
        logger.warning(f"LLM response parsing error: {e}")

    return explanation, recommendations, disclaimer


# ─── Node: Prepare Prompt ─────────────────────────────────────────────────────

def prepare_prompt_node(state: ExplanationState) -> ExplanationState:
    """Builds the LLM prompt and checks whether Deploy AI credentials are available."""
    token     = state["risk_report"].get("token", "UNKNOWN")
    token_creds = bool(os.getenv("CLIENT_ID") and os.getenv("CLIENT_SECRET"))

    state["prompt"]  = _build_llm_prompt(state["risk_report"])
    state["use_llm"] = token_creds

    logger.info(
        f"[{token}] Prompt prepared. LLM mode: {'Deploy AI' if token_creds else 'template fallback'}"
    )
    return state


# ─── Node: Generate Explanation ───────────────────────────────────────────────

def generate_explanation_node(state: ExplanationState) -> ExplanationState:
    """Calls Deploy AI LLM or falls back to template generation."""
    token = state["risk_report"].get("token", "UNKNOWN")

    if state["use_llm"]:
        logger.info(f"[{token}] Calling Deploy AI LLM...")
        access_token = _get_access_token()
        chat_id      = _create_chat(access_token) if access_token else None

        if chat_id:
            raw = _call_llm(access_token, chat_id, state["prompt"])
            if raw:
                state["raw_llm_response"] = raw
                exp, recs, disc = _parse_llm_response(raw)
                # Validate we got all three parts; fall back if parsing failed
                if exp and len(recs) == 3 and disc:
                    state["explanation"]     = exp
                    state["recommendations"] = recs
                    state["disclaimer"]      = disc
                    logger.info(f"[{token}] LLM explanation generated successfully.")
                    return state
                else:
                    logger.warning(f"[{token}] LLM response incomplete — falling back to template.")
            else:
                logger.warning(f"[{token}] LLM returned no content — falling back to template.")
        else:
            logger.warning(f"[{token}] Could not create chat — falling back to template.")

    # Template fallback
    logger.info(f"[{token}] Using template-based explanation.")
    exp, recs, disc = _build_template_explanation(state["risk_report"])
    state["explanation"]     = exp
    state["recommendations"] = recs
    state["disclaimer"]      = disc
    state["raw_llm_response"] = ""
    return state


# ─── Node: Format Output ──────────────────────────────────────────────────────

def format_output_node(state: ExplanationState) -> ExplanationState:
    """Assembles the final structured output."""
    risk = state["risk_report"]
    token = risk.get("token", "UNKNOWN")

    state["final_output"] = {
        "token":           token,
        "risk_score":      risk.get("risk_score"),
        "risk_category":   risk.get("risk_category"),
        "explanation":     state["explanation"],
        "recommendations": [f"• {r}" for r in state["recommendations"]],
        "disclaimer":      state["disclaimer"],
        "metadata": {
            "explained_at":         datetime.now(timezone.utc).isoformat(),
            "explanation_source":   "llm" if state["use_llm"] and state["raw_llm_response"] else "template",
            "agent_version":        "1.0.0",
        },
    }

    logger.info(
        f"[{token}] Output formatted. Source: {state['final_output']['metadata']['explanation_source']}"
    )
    return state


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_explanation_graph():
    """
    Assembles the LangGraph StateGraph for risk explanation.

    Flow:
      prepare_prompt → generate_explanation → format_output → END

    Extension points:
      - Add a 'translate' node after format_output to support multiple languages
      - Add a 'personalize' node that adjusts tone based on user risk appetite
      - Route to different LLMs via conditional_edge (e.g. GPT-4o vs Claude vs local)
      - Add a 'summarize' node to condense the explanation to a single sentence
    """
    graph = StateGraph(ExplanationState)

    graph.add_node("prepare_prompt",        prepare_prompt_node)
    graph.add_node("generate_explanation",  generate_explanation_node)
    graph.add_node("format_output",         format_output_node)

    graph.set_entry_point("prepare_prompt")
    graph.add_edge("prepare_prompt",       "generate_explanation")
    graph.add_edge("generate_explanation", "format_output")
    graph.add_edge("format_output",        END)

    return graph.compile()


# ─── Public API ───────────────────────────────────────────────────────────────

def explain_risk(risk_report: dict) -> dict:
    """
    Accepts a risk report JSON from crypto_risk_agent and returns a human-readable explanation.

    Args:
        risk_report (dict): Output of analyze_token() from crypto_risk_agent.py

    Returns:
        dict: {
            token, risk_score, risk_category,
            explanation, recommendations, disclaimer, metadata
        }
    """
    app = build_explanation_graph()

    initial_state: ExplanationState = {
        "risk_report":      risk_report,
        "prompt":           "",
        "raw_llm_response": "",
        "explanation":      "",
        "recommendations":  [],
        "disclaimer":       "",
        "use_llm":          False,
        "final_output":     {},
    }

    result = app.invoke(initial_state)
    return result["final_output"]


def analyze_and_explain(token: str) -> dict:
    """
    End-to-end pipeline: token symbol → risk analysis → plain-language explanation.

    Args:
        token (str): Cryptocurrency symbol, e.g. "BTC", "PEPE"

    Returns:
        dict: Full explanation output including score, category, explanation,
              recommendations, and disclaimer.
    """
    logger.info(f"Starting end-to-end pipeline for: {token}")
    risk_report = analyze_token(token)
    return explain_risk(risk_report)


# ─── CLI Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_tokens = ["BTC", "ETH", "PEPE", "SOL"]

    for token in test_tokens:
        print(f"\n{'═' * 62}")
        print(f"  RISK EXPLANATION — {token}")
        print(f"{'═' * 62}")
        result = analyze_and_explain(token)

        print(f"\n📊 Risk Score : {result['risk_score']} / 100")
        print(f"🔖 Category   : {result['risk_category']}")
        print(f"\n📝 Explanation:\n{result['explanation']}")
        print(f"\n✅ Recommendations:")
        for rec in result["recommendations"]:
            print(f"  {rec}")
        print(f"\n⚠️  Disclaimer:\n{result['disclaimer']}")
        print(f"\n🔧 Source: {result['metadata']['explanation_source']}")
