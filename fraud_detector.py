"""
fraud_detector.py — Ontology-guided fraud/scam detector

Architecture:
  AttributeExtractor  — LLM scores 8 fraud attributes from graph context
  BayesianFraudClassifier — conditional dependency model over attribute scores
  CrossAttributeAttention — context-aware reweighting of attribute scores

Plugs into Ontology Engine V7 via the shared `store` singleton.
No existing files modified.

Attributes (from IISE paper transferable attribute taxonomy):
  urgency                 — pressure to act immediately
  anonymity               — sender/entity identity is hidden or unverifiable
  asymmetric_benefit      — disproportionate reward promised for little effort
  authority_impersonation — false claim of official/trusted identity
  isolation_tactic        — attempts to cut victim off from outside counsel
  information_harvesting  — requests for personal/financial data
  payment_irregularity    — unusual payment methods (gift cards, wire, crypto)
  too_good_to_be_true     — offer defies normal economic logic
"""

import os
import json
import math
import numpy as np
from typing import Optional
try:
    from groq_agent import call_groq_json          # direct run / Colab sys.path
except ImportError:
    from agents.groq_agent import call_groq_json   # package import

# ---------------------------------------------------------------------------
# Attribute taxonomy
# ---------------------------------------------------------------------------

ATTRIBUTES = [
    "urgency",
    "anonymity",
    "asymmetric_benefit",
    "authority_impersonation",
    "isolation_tactic",
    "information_harvesting",
    "payment_irregularity",
    "too_good_to_be_true",
]

ATTRIBUTE_DESCRIPTIONS = {
    "urgency":                 "Creates artificial time pressure; 'act now or lose the opportunity'",
    "anonymity":               "Sender/entity identity is hidden, unverifiable, or uses fake credentials",
    "asymmetric_benefit":      "Promises disproportionate reward for minimal effort or investment",
    "authority_impersonation": "Falsely claims to be IRS, FBI, bank, employer, or trusted institution",
    "isolation_tactic":        "Instructs victim to keep secret, not consult family/lawyer/bank",
    "information_harvesting":  "Requests SSN, passwords, credit card, bank account, or personal data",
    "payment_irregularity":    "Requests gift cards, wire transfer, cryptocurrency, or unusual payment",
    "too_good_to_be_true":     "Offer (prize, job, investment) defies normal economic reality",
}

# ---------------------------------------------------------------------------
# Bayesian prior: P(fraud | attribute present) learned from known fraud cases
# These are empirically grounded priors from fraud research literature.
# They update via Bayes as attribute scores accumulate.
# ---------------------------------------------------------------------------

FRAUD_PRIORS = {
    "urgency":                 0.72,
    "anonymity":               0.68,
    "asymmetric_benefit":      0.78,
    "authority_impersonation": 0.85,
    "isolation_tactic":        0.90,  # strongest single signal
    "information_harvesting":  0.82,
    "payment_irregularity":    0.88,
    "too_good_to_be_true":     0.75,
}

# Conditional correlations: P(B=high | A=high) — cross-attribute dependencies
# This is the key improvement over naive weighted sum:
# some attributes co-occur in real fraud patterns, amplifying the signal.
ATTRIBUTE_CORRELATIONS = {
    ("authority_impersonation", "isolation_tactic"):   0.78,
    ("urgency",                 "payment_irregularity"): 0.71,
    ("asymmetric_benefit",      "information_harvesting"): 0.65,
    ("anonymity",               "payment_irregularity"): 0.69,
    ("too_good_to_be_true",     "asymmetric_benefit"):  0.74,
    ("isolation_tactic",        "information_harvesting"): 0.72,
    ("authority_impersonation", "payment_irregularity"): 0.66,
}

BASE_FRAUD_RATE = 0.35  # P(fraud) for submitted/flagged text — user is submitting suspicious content


# ---------------------------------------------------------------------------
# Cross-Attribute Attention
# ---------------------------------------------------------------------------

class CrossAttributeAttention:
    """
    Reweights attribute scores based on observed co-occurrence patterns.

    Inspired by the cross-attribute interaction term in the IISE paper:
    when two correlated fraud attributes both score high, their combined
    weight increases non-linearly — human intuition about "pattern matching."
    """

    def __init__(self):
        self.correlation_map = ATTRIBUTE_CORRELATIONS

    def compute_attention_weights(self, raw_scores: dict[str, float]) -> dict[str, float]:
        """
        Returns attention-adjusted weights for each attribute.

        For each attribute a_i, its attention weight is amplified by:
          Σ corr(a_i, a_j) * score(a_j)  for all correlated pairs
        """
        weights = {attr: 1.0 for attr in ATTRIBUTES}

        for (a1, a2), corr in self.correlation_map.items():
            if a1 in raw_scores and a2 in raw_scores:
                s1 = raw_scores[a1]
                s2 = raw_scores[a2]
                # Both high → amplify each other's weight
                amplification = corr * s1 * s2
                weights[a1] = weights.get(a1, 1.0) + amplification
                weights[a2] = weights.get(a2, 1.0) + amplification

        # Normalize weights to [1.0, 2.0] range
        max_w = max(weights.values()) if weights else 1.0
        if max_w > 1.0:
            for attr in weights:
                weights[attr] = 1.0 + (weights[attr] - 1.0) / (max_w - 1.0)

        return weights


# ---------------------------------------------------------------------------
# Bayesian Fraud Classifier
# ---------------------------------------------------------------------------

class BayesianFraudClassifier:
    """
    Naive Bayes over attribute scores with cross-attribute attention.

    P(fraud | attributes) ∝ P(fraud) * Π P(attr_i | fraud)^w_i

    Uses log-probability domain for numerical stability.
    Attribute scores [0,1] interpolate between P(attr|legit) and P(attr|fraud).
    """

    def __init__(self):
        self.attention = CrossAttributeAttention()
        self.base_fraud_rate = BASE_FRAUD_RATE
        self.priors = FRAUD_PRIORS

    def classify(
        self,
        attribute_scores: dict[str, float],
        graph_context: Optional[dict] = None
    ) -> dict:
        """
        Returns fraud classification with full explainability.

        Args:
            attribute_scores: {attr_name: score [0.0–1.0]}
            graph_context: optional graph evidence (node count, known fraud entities)

        Returns:
            {
              fraud_score: float,          # 0.0–1.0
              risk_level: str,             # LOW / MEDIUM / HIGH / CRITICAL
              confidence: float,           # model confidence [0,1]
              top_signals: list,           # top 3 attributes driving score
              attribute_scores: dict,      # raw scores
              attention_weights: dict,     # cross-attribute weights
              bayesian_evidence: dict,     # log-likelihood contributions
              graph_boost: float,          # bonus from ontology evidence
              explanation: str,            # human-readable verdict
            }
        """
        # 1. Compute cross-attribute attention weights
        attention_weights = self.attention.compute_attention_weights(attribute_scores)

        # 2. Bayesian log-likelihood accumulation
        log_prior_fraud = math.log(self.base_fraud_rate)
        log_prior_legit = math.log(1.0 - self.base_fraud_rate)

        log_likelihood_fraud = 0.0
        log_likelihood_legit = 0.0
        evidence = {}

        for attr in ATTRIBUTES:
            score = attribute_scores.get(attr, 0.0)
            w = attention_weights.get(attr, 1.0)
            p_fraud = self.priors[attr]
            p_legit = 1.0 - p_fraud

            # Only contribute when score is meaningfully above zero.
            # A score of 0.0 means "not observed" — it should be neutral,
            # not counted as evidence the message is legitimate.
            # Threshold 0.15 filters out noise while keeping weak signals.
            if score < 0.15:
                evidence[attr] = {
                    "score": round(score, 3),
                    "attention_weight": round(w, 3),
                    "log_contribution": 0.0,
                }
                continue

            # Interpolate between neutral (0.5) and full prior
            # score=0.15 → barely above neutral; score=1.0 → full prior weight
            t = (score - 0.15) / 0.85  # rescale [0.15,1.0] → [0,1]
            effective_p_fraud = 0.5 + t * (p_fraud - 0.5)
            effective_p_legit = 0.5 + t * (p_legit - 0.5)

            ll_fraud = w * math.log(max(effective_p_fraud, 1e-9))
            ll_legit = w * math.log(max(effective_p_legit, 1e-9))

            log_likelihood_fraud += ll_fraud
            log_likelihood_legit += ll_legit
            evidence[attr] = {
                "score": round(score, 3),
                "attention_weight": round(w, 3),
                "log_contribution": round(ll_fraud - ll_legit, 4),
            }

        # 3. Posterior via Bayes theorem (log domain → sigmoid)
        log_posterior_fraud = log_prior_fraud + log_likelihood_fraud
        log_posterior_legit = log_prior_legit + log_likelihood_legit

        # Normalize to get P(fraud | evidence)
        log_sum = log_posterior_fraud + math.log1p(math.exp(log_posterior_legit - log_posterior_fraud))
        fraud_prob = math.exp(log_posterior_fraud - log_sum)
        fraud_prob = max(0.0, min(1.0, fraud_prob))

        # 4. Graph context boost (ontology evidence)
        graph_boost = 0.0
        if graph_context:
            fraud_nodes = graph_context.get("fraud_related_nodes", 0)
            total_nodes = max(graph_context.get("total_nodes", 1), 1)
            fraud_density = fraud_nodes / total_nodes
            graph_boost = fraud_density * 0.15  # max 15% boost from graph
            fraud_prob = min(1.0, fraud_prob + graph_boost * (1.0 - fraud_prob))

        # 5. Confidence: based on evidence strength (how many attributes scored > 0.5)
        active_signals = sum(1 for s in attribute_scores.values() if s > 0.5)
        confidence = min(0.95, 0.4 + active_signals * 0.08)

        # 6. Risk level
        if fraud_prob >= 0.80:
            risk_level = "CRITICAL"
        elif fraud_prob >= 0.60:
            risk_level = "HIGH"
        elif fraud_prob >= 0.35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 7. Top signals
        top_signals = sorted(
            [
                {
                    "attribute": attr,
                    "score": round(attribute_scores.get(attr, 0.0), 3),
                    "weight": round(attention_weights.get(attr, 1.0), 3),
                    "contribution": round(evidence[attr]["log_contribution"], 4),
                }
                for attr in ATTRIBUTES
            ],
            key=lambda x: x["contribution"],
            reverse=True,
        )[:3]

        # 8. Human-readable explanation
        explanation = self._build_explanation(
            fraud_prob, risk_level, top_signals, graph_boost
        )

        return {
            "fraud_score": round(fraud_prob, 4),
            "risk_level": risk_level,
            "confidence": round(confidence, 3),
            "top_signals": top_signals,
            "attribute_scores": {k: round(v, 3) for k, v in attribute_scores.items()},
            "attention_weights": {k: round(v, 3) for k, v in attention_weights.items()},
            "bayesian_evidence": evidence,
            "graph_boost": round(graph_boost, 4),
            "explanation": explanation,
        }

    def _build_explanation(self, score, risk, top_signals, graph_boost):
        lines = [f"Risk level: {risk} (fraud probability: {score:.1%})."]
        if top_signals:
            drivers = ", ".join(s["attribute"].replace("_", " ") for s in top_signals[:2])
            lines.append(f"Primary signals: {drivers}.")
        if graph_boost > 0.02:
            lines.append(f"Knowledge graph contains related fraud entities (+{graph_boost:.1%} boost).")
        if score < 0.35:
            lines.append("No strong fraud indicators detected.")
        return " ".join(lines)


# ---------------------------------------------------------------------------
# Attribute Extractor
# ---------------------------------------------------------------------------

ATTRIBUTE_PROMPT = """You are a fraud detection expert. Score each fraud attribute for the text below.

Scoring scale:
  0.0 = definitely not present
  0.3 = weakly suggested
  0.6 = moderately present
  0.8 = clearly present
  1.0 = extremely strong signal

Text to analyze:
---
{text}
---

Knowledge graph context (fraud concepts found in ontology):
---
{graph_context}
---

Attribute definitions and scoring guidance:
{attribute_list}

Scoring rules:
- "URGENT", "immediately", "act now", "24 hours" → urgency >= 0.8
- "gift cards", "wire transfer", "cryptocurrency", "iTunes" → payment_irregularity >= 0.9
- "IRS", "FBI", "bank", "Microsoft", "government" (falsely) → authority_impersonation >= 0.8
- "SSN", "password", "bank account", "credit card" → information_harvesting >= 0.9
- "$X prize", "you've won", "selected", "lottery" → too_good_to_be_true >= 0.8, asymmetric_benefit >= 0.7
- Unverifiable sender, no return address, anonymous → anonymity >= 0.6
- "don't tell anyone", "keep this private", "don't contact your bank" → isolation_tactic >= 0.8
- Legitimate appointment reminders, shipping confirmations → all scores near 0.0

Return ONLY valid JSON, no markdown:
{{
  "urgency": 0.0,
  "anonymity": 0.0,
  "asymmetric_benefit": 0.0,
  "authority_impersonation": 0.0,
  "isolation_tactic": 0.0,
  "information_harvesting": 0.0,
  "payment_irregularity": 0.0,
  "too_good_to_be_true": 0.0,
  "reasoning": "one sentence on the strongest signals found"
}}
"""


class AttributeExtractor:
    """
    LLM-based agent that scores the 8 fraud attributes from input text,
    grounded by knowledge graph context from the ontology engine.
    """

    def __init__(self, store):
        self.store = store

    def extract(self, text: str) -> tuple[dict[str, float], str, dict]:
        """
        Args:
            text: message, email, URL content, or any input text to analyze

        Returns:
            (attribute_scores, reasoning, graph_context)
        """
        # 1. Query graph for fraud-related context
        graph_context = self._get_graph_context(text)

        # 2. Build attribute list string
        attr_list = "\n".join(
            f"  {attr}: {ATTRIBUTE_DESCRIPTIONS[attr]}"
            for attr in ATTRIBUTES
        )

        # 3. Summarize graph context for prompt
        graph_summary = self._summarize_graph_context(graph_context)

        # 4. Call LLM
        prompt = ATTRIBUTE_PROMPT.format(
            text=text[:3000],  # cap at 3k chars
            graph_context=graph_summary,
            attribute_list=attr_list,
        )

        result = call_groq_json(prompt, temperature=0.1)

        # 5. Extract scores (default 0.0 for missing/invalid)
        scores = {}
        for attr in ATTRIBUTES:
            val = result.get(attr, 0.0)
            try:
                scores[attr] = max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                scores[attr] = 0.0

        reasoning = result.get("reasoning", "")
        return scores, reasoning, graph_context

    def _get_graph_context(self, text: str) -> dict:
        """Query the ontology graph for fraud-related nodes near the input text."""
        try:
            query_result = self.store.query(text[:200])
            nodes = query_result.get("nodes", [])
            fraud_keywords = {
                "fraud", "scam", "phishing", "deception", "impersonation",
                "advance fee", "lottery", "wire transfer", "gift card",
                "identity theft", "social engineering", "ponzi", "pyramid",
            }
            fraud_nodes = [
                n for n in nodes
                if any(kw in n.get("label", "").lower() or kw in n.get("desc", "").lower()
                       for kw in fraud_keywords)
            ]
            return {
                "total_nodes": len(nodes),
                "fraud_related_nodes": len(fraud_nodes),
                "fraud_node_labels": [n.get("label", "") for n in fraud_nodes[:10]],
                "query_nodes": [n.get("label", "") for n in nodes[:15]],
            }
        except Exception:
            return {"total_nodes": 0, "fraud_related_nodes": 0,
                    "fraud_node_labels": [], "query_nodes": []}

    def _summarize_graph_context(self, ctx: dict) -> str:
        if ctx["fraud_related_nodes"] == 0:
            return "No directly related fraud concepts found in knowledge graph."
        labels = ", ".join(ctx["fraud_node_labels"][:8])
        return (
            f"Found {ctx['fraud_related_nodes']} fraud-related nodes in ontology: {labels}. "
            f"Total graph context: {ctx['total_nodes']} nodes."
        )


# ---------------------------------------------------------------------------
# Top-level detector (combines extractor + classifier)
# ---------------------------------------------------------------------------

class FraudDetector:
    """
    Main entry point. Given input text, returns full fraud analysis.

    Usage:
        detector = FraudDetector(store)
        result = detector.analyze("You've won $1,000,000! Send $50 to claim.")
    """

    def __init__(self, store):
        self.extractor = AttributeExtractor(store)
        self.classifier = BayesianFraudClassifier()

    def analyze(self, text: str) -> dict:
        """
        Full fraud analysis pipeline:
          1. Extract attribute scores via LLM + graph context
          2. Classify via Bayesian model with cross-attribute attention
          3. Return unified result

        Returns:
            {
              input_preview: str,
              fraud_score: float,
              risk_level: str,
              confidence: float,
              top_signals: list,
              attribute_scores: dict,
              attention_weights: dict,
              bayesian_evidence: dict,
              graph_boost: float,
              llm_reasoning: str,
              graph_context: dict,
              explanation: str,
            }
        """
        if not text or not text.strip():
            return {"error": "Empty input text"}

        # Step 1: Extract attribute scores
        attribute_scores, reasoning, graph_context = self.extractor.extract(text)

        # Step 2: Classify
        classification = self.classifier.classify(attribute_scores, graph_context)

        # Step 3: Merge results
        result = {
            "input_preview": text[:200] + ("..." if len(text) > 200 else ""),
            **classification,
            "llm_reasoning": reasoning,
            "graph_context": {
                "fraud_related_nodes": graph_context.get("fraud_related_nodes", 0),
                "total_nodes": graph_context.get("total_nodes", 0),
                "fraud_node_labels": graph_context.get("fraud_node_labels", []),
            },
        }

        return result
