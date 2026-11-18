"""
zero_shot_classifier.py — Ontology-grounded Zero-Shot Fraud Classifier

Three-source Dempster-Shafer fusion — replicating human fraud intuition:

  Source 1: PROTOTYPE COSINE MATCH
    Attribute vector of input → cosine similarity to each fraud TYPE prototype
    "This looks like a phishing email" = pattern recognition
    → converts to DS belief mass per fraud type

  Source 2: BAYESIAN ATTRIBUTE PRIORS
    Each attribute score × its empirical prior P(fraud|attribute)
    "Payment irregularity is 88% predictive of fraud"
    → Bayesian update from prior knowledge

  Source 3: GRAPH PATH EVIDENCE
    BFS from input's matched entities to known fraud hub nodes
    "Concepts in this text connect to fraud topology"
    → structural evidence from ontology graph

  DS FUSION:
    Combine all three evidence sources via Dempster's rule
    Zero sources = full uncertainty (not false certainty)
    Contradicting sources = reduced certainty (not averaged away)
    Agreeing sources = amplified certainty (multiplicative, not additive)

This is the "real deal" — no source relies on the LLM's pre-trained bias.
LLM only extracts attribute scores; all classification logic is ontology-grounded.
"""

import os
import json
import math
from pathlib import Path
from collections import deque

try:
    from groq_agent import call_groq_json
    from graph_store import GraphStore
    from prototype_builder import PrototypeBuilder, FRAUD_TYPE_SEEDS
    from fraud_detector import (
        ATTRIBUTES, ATTRIBUTE_DESCRIPTIONS, FRAUD_PRIORS,
        ATTRIBUTE_CORRELATIONS, CrossAttributeAttention
    )
except ImportError:
    from agents.groq_agent import call_groq_json
    from agents.graph_store import GraphStore
    from agents.prototype_builder import PrototypeBuilder, FRAUD_TYPE_SEEDS
    from agents.fraud_detector import (
        ATTRIBUTES, ATTRIBUTE_DESCRIPTIONS, FRAUD_PRIORS,
        ATTRIBUTE_CORRELATIONS, CrossAttributeAttention
    )

# ---------------------------------------------------------------------------
# DS evidence-to-BBA converters
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two attribute dicts."""
    dot   = sum(vec_a.get(a, 0) * vec_b.get(a, 0) for a in ATTRIBUTES)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values())) + 1e-9
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values())) + 1e-9
    return dot / (mag_a * mag_b)


def _prototype_to_bba(cosine_sim: float, confidence: float) -> tuple:
    """
    Convert prototype cosine similarity → BBA (belief_fraud, belief_legit, uncertainty).

    High cosine → strong fraud belief
    Low cosine  → mild legit evidence (absence of pattern ≠ proof of legit)
    confidence  → how well the prototype is defined in the ontology
    """
    # Scale by prototype confidence — a poorly-defined prototype contributes less
    effective_sim = cosine_sim * confidence

    if effective_sim < 0.15:
        # No meaningful match to any fraud pattern
        return (0.0, 0.05, 0.95)

    if effective_sim > 0.80:
        # Very strong match to known fraud pattern
        belief_fraud = 0.60 + 0.30 * (effective_sim - 0.80) / 0.20
        return (min(0.90, belief_fraud), 0.02, max(0.08, 1.0 - belief_fraud - 0.02))

    # Smooth ramp in between
    belief_fraud  = 0.50 * effective_sim
    belief_legit  = 0.03 * (1.0 - effective_sim)
    uncertainty   = max(0.0, 1.0 - belief_fraud - belief_legit)
    return (belief_fraud, belief_legit, uncertainty)


def _bayesian_to_bba(score: float, prior: float) -> tuple:
    """
    Convert Bayesian attribute evidence → BBA.
    Uses the same logic as DempsterShaferFraudClassifier._bba
    but separated here for clarity.
    """
    if score < 0.10:
        return (0.0, 0.0, 1.0)  # Absent attribute = full uncertainty

    discriminability = (prior - 0.5) * 2.0  # [0, 1]
    commitment  = score * discriminability
    b_fraud     = commitment * prior
    b_legit     = commitment * (1.0 - prior)
    uncertainty = max(0.0, 1.0 - b_fraud - b_legit)
    return (b_fraud, b_legit, uncertainty)


def _graph_path_to_bba(path_score: float) -> tuple:
    """
    Convert graph path evidence → BBA.
    path_score = 0.0–1.0 based on BFS proximity to fraud hub nodes
    """
    if path_score < 0.05:
        return (0.0, 0.0, 1.0)  # No graph connection

    # Graph evidence is weaker than direct attribute evidence
    # (topological proximity ≠ same as explicit scam pattern)
    b_fraud     = 0.35 * path_score
    b_legit     = 0.02 * (1.0 - path_score)
    uncertainty = max(0.0, 1.0 - b_fraud - b_legit)
    return (b_fraud, b_legit, uncertainty)


# ---------------------------------------------------------------------------
# Dempster's combination rule
# ---------------------------------------------------------------------------

def _combine_two_bbas(bba1: tuple, bba2: tuple) -> tuple:
    """
    Dempster's combination rule for two BBAs.
    bba = (belief_fraud, belief_legit, uncertainty)
    """
    f1, l1, u1 = bba1
    f2, l2, u2 = bba2

    # Conflict: one says fraud, other says legit
    conflict = f1 * l2 + l1 * f2
    K = 1.0 - conflict

    if K < 1e-6:
        # Total conflict — fall back to equal uncertainty
        return (0.0, 0.0, 1.0)

    # Dempster's rule: normalize conflict away
    b_fraud     = (f1 * f2 + f1 * u2 + u1 * f2) / K
    b_legit     = (l1 * l2 + l1 * u2 + u1 * l2) / K
    uncertainty = (u1 * u2) / K

    # Normalize to [0,1] (floating point drift)
    total = b_fraud + b_legit + uncertainty
    if total > 1e-6:
        b_fraud     /= total
        b_legit     /= total
        uncertainty /= total

    return (b_fraud, b_legit, uncertainty)


def _combine_all_bbas(bbas: list[tuple]) -> tuple:
    """Fold Dempster's rule over a list of BBAs."""
    if not bbas:
        return (0.0, 0.0, 1.0)
    result = bbas[0]
    for bba in bbas[1:]:
        result = _combine_two_bbas(result, bba)
    return result


# ---------------------------------------------------------------------------
# Graph path evidence extractor
# ---------------------------------------------------------------------------

FRAUD_HUB_KEYWORDS = [
    "fraud", "scam", "phishing", "ponzi", "pyramid scheme",
    "advance fee", "identity theft", "money laundering", "ransomware",
    "pig butchering", "romance scam", "lottery scam", "impersonation scam",
]


def _extract_graph_evidence(store, matched_node_ids: list[str]) -> dict:
    """
    BFS from matched entities to fraud hub nodes in the ontology graph.
    Returns a dict with path_score and evidence details.
    """
    if not matched_node_ids or not store.nodes:
        return {"path_score": 0.0, "hub_nodes_found": [], "min_depth": -1}

    # Find fraud hub nodes in the graph
    fraud_hubs = set()
    for node_id, node in store.nodes.items():
        label = node.get("label", "").lower()
        desc  = node.get("desc",  "").lower()
        if any(kw in label or kw in desc for kw in FRAUD_HUB_KEYWORDS):
            fraud_hubs.add(node_id)

    if not fraud_hubs:
        return {"path_score": 0.0, "hub_nodes_found": [], "max_depth": -1}

    # BFS from each matched node — find shortest path to any fraud hub
    min_depth = float("inf")
    hub_nodes_found = []
    max_bfs_depth = 4  # max 4 hops

    for start_id in matched_node_ids[:5]:  # limit BFS starts
        if start_id not in store.nodes:
            continue

        visited = {start_id}
        queue   = deque([(start_id, 0)])

        while queue:
            node_id, depth = queue.popleft()
            if depth > max_bfs_depth:
                break

            if node_id in fraud_hubs:
                min_depth = min(min_depth, depth)
                hub_label = store.nodes[node_id].get("label", node_id)
                if hub_label not in hub_nodes_found:
                    hub_nodes_found.append(hub_label)

            # Traverse neighbors via edges
            for edge in store.edges:
                if edge.get("src") == node_id and edge.get("dst") not in visited:
                    visited.add(edge["dst"])
                    queue.append((edge["dst"], depth + 1))
                elif edge.get("dst") == node_id and edge.get("src") not in visited:
                    visited.add(edge["src"])
                    queue.append((edge["src"], depth + 1))

    # Convert depth to score: depth 0=1.0, depth 1=0.8, depth 2=0.5, depth 3=0.3, depth 4=0.15
    depth_scores = {0: 0.20, 1: 0.70, 2: 0.45, 3: 0.25, 4: 0.10}
    path_score = depth_scores.get(min_depth, 0.0) if min_depth != float("inf") else 0.0

    return {
        "path_score":       path_score,
        "hub_nodes_found":  hub_nodes_found[:5],
        "min_depth":        min_depth if min_depth != float("inf") else -1,
    }


# ---------------------------------------------------------------------------
# Zero-Shot Fraud Classifier
# ---------------------------------------------------------------------------

class ZeroShotFraudClassifier:
    """
    Three-source Dempster-Shafer fraud classifier grounded in the ontology.

    Evidence sources:
      1. Prototype cosine match  — zero-shot pattern recognition
      2. Bayesian attribute priors — empirical discriminability
      3. Graph path evidence     — ontology topology
    """

    def __init__(self, store):
        self.store      = store
        self.attention  = CrossAttributeAttention()
        self.proto_builder = PrototypeBuilder(store)
        self._prototypes = None

    def _load_prototypes(self):
        """Load or build prototypes as needed."""
        if self._prototypes is not None:
            return self._prototypes

        self._prototypes = self.proto_builder.load()

        if not self._prototypes or self.proto_builder.needs_rebuild():
            print("  [ZeroShot] Building prototypes from ontology graph...")
            self._prototypes = self.proto_builder.build(verbose=True)

        return self._prototypes

    def _extract_attribute_scores(self, text: str) -> tuple:
        """
        Use LLM to extract attribute scores + return matched node IDs from graph.
        LLM reads user text; graph provides semantic context.
        """
        # Query graph for entities mentioned in the text (top 5)
        graph_result = self.store.query(text[:300])
        matched_node_ids = graph_result.get("matches", [])

        # Build graph context for the LLM prompt
        graph_context_lines = []
        for node_id in matched_node_ids[:5]:
            if node_id in self.store.nodes:
                n = self.store.nodes[node_id]
                graph_context_lines.append(
                    f"- {n.get('label','?')}: {n.get('desc','')[:100]}"
                )

        graph_context = (
            "\n".join(graph_context_lines)
            if graph_context_lines
            else "No relevant fraud concepts found in graph."
        )

        # Build attribute scoring prompt — no .format() on ATTRIBUTE_PROMPT
        # to avoid brace-escaping issues; build it directly here
        attr_lines = "\n".join(
            f'  "{a}": 0.0,  // {ATTRIBUTE_DESCRIPTIONS[a]}'
            for a in ATTRIBUTES
        )
        json_keys = ", ".join(f'"{a}": 0.0' for a in ATTRIBUTES)

        prompt = (
            "You are a fraud detection system. Score each attribute 0.0-1.0 based ONLY on the text.\n\n"
            "Scoring rules:\n"
            '- "URGENT", "immediately", "act now", "24 hours" -> urgency >= 0.8\n'
            '- "gift cards", "wire transfer", "cryptocurrency", "iTunes" -> payment_irregularity >= 0.9\n'
            '- "IRS", "FBI", "bank", "Microsoft", "government" (falsely) -> authority_impersonation >= 0.8\n'
            '- "SSN", "password", "bank account", "credit card" -> information_harvesting >= 0.9\n'
            '- "you\'ve won", "selected", "lottery", "prize", "inheritance" -> too_good_to_be_true >= 0.8\n'
            "- Unverifiable or anonymous sender -> anonymity >= 0.6\n"
            '- "don\'t tell anyone", "keep this private" -> isolation_tactic >= 0.8\n'
            "- Suspicious URL with real brand name -> anonymity >= 0.7, authority_impersonation >= 0.6\n"
            "- Legitimate appointment / shipping confirmation -> all scores near 0.0\n\n"
            f"Ontology graph context (fraud concepts related to this text):\n{graph_context}\n\n"
            f"Text to analyze:\n{text[:2000]}\n\n"
            "Return ONLY valid JSON, no markdown:\n"
            "{\n"
            f"{attr_lines}\n"
            '  "reasoning": "one sentence on the strongest signals found"\n'
            "}"
        )

        result = call_groq_json(prompt, temperature=0.0)

        scores = {}
        for attr in ATTRIBUTES:
            try:
                scores[attr] = max(0.0, min(1.0, float(result.get(attr, 0.0))))
            except (TypeError, ValueError):
                scores[attr] = 0.0

        reasoning = result.get("reasoning", "No reasoning provided")
        return scores, reasoning, matched_node_ids

    def classify(self, text: str) -> dict:
        """
        Full zero-shot ontology-grounded classification.

        Returns detailed result with all three evidence sources explained.
        """
        # --- Step 1: Extract attribute scores (LLM + graph context) ---
        raw_scores, reasoning, matched_node_ids = self._extract_attribute_scores(text)

        # --- Step 2: Cross-attribute attention (co-occurrence amplification) ---
        attn_weights   = self.attention.compute_attention_weights(raw_scores)
        boosted_scores = {
            attr: min(1.0, raw_scores[attr] * attn_weights[attr])
            for attr in ATTRIBUTES
        }

        # --- Step 3: Load ontology prototypes ---
        prototypes = self._load_prototypes()

        # --- Step 4: Prototype cosine match (Source 1) ---
        proto_evidence = {}
        best_fraud_type = None
        best_cosine     = 0.0

        for fraud_type, proto in prototypes.items():
            if fraud_type == "_meta":
                continue
            if "attributes" not in proto:
                continue

            cosine = _cosine_similarity(boosted_scores, proto["attributes"])
            conf   = proto.get("confidence", 0.5)
            bba    = _prototype_to_bba(cosine, conf)

            proto_evidence[fraud_type] = {
                "cosine": round(cosine, 4),
                "bba":    bba,
                "is_fallback": proto.get("is_fallback", False),
            }

            if cosine > best_cosine:
                best_cosine     = cosine
                best_fraud_type = fraud_type

        # Aggregate prototype BBAs — take the strongest fraud type's BBA
        # plus a weighted blend of all types
        if proto_evidence:
            all_proto_bbas = [v["bba"] for v in proto_evidence.values()]
            proto_bba      = _combine_all_bbas(all_proto_bbas)
        else:
            proto_bba = (0.0, 0.0, 1.0)  # No prototypes = full uncertainty

        # --- Step 5: Bayesian attribute priors (Source 2) ---
        bayesian_bbas = []
        for attr in ATTRIBUTES:
            if boosted_scores[attr] > 0.05:
                bba = _bayesian_to_bba(boosted_scores[attr], FRAUD_PRIORS[attr])
                bayesian_bbas.append(bba)

        bayesian_bba = _combine_all_bbas(bayesian_bbas) if bayesian_bbas else (0.0, 0.0, 1.0)

        # --- Step 6: Graph path evidence (Source 3) ---
        graph_evidence = _extract_graph_evidence(self.store, matched_node_ids)
        graph_bba      = _graph_path_to_bba(graph_evidence["path_score"])

        # --- Step 7: Dempster-Shafer three-source fusion ---
        final_bba = _combine_all_bbas([proto_bba, bayesian_bba, graph_bba])
        belief_fraud, belief_legit, uncertainty = final_bba

        # --- Step 8: Pignistic transform → sigmoid calibration ---
        # Sigmoid over belief_fraud: maps 0→~8%, 0.3→50%, 0.6→99%
        fraud_prob = 1.0 / (1.0 + math.exp(-9.0 * (belief_fraud - 0.45)))

        # --- Step 9: Risk level ---
        if fraud_prob >= 0.75:
            risk_level = "CRITICAL"
        elif fraud_prob >= 0.55:
            risk_level = "HIGH"
        elif fraud_prob >= 0.42:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # --- Step 10: Top triggering attributes ---
        top_attrs = sorted(
            [(a, boosted_scores[a]) for a in ATTRIBUTES if boosted_scores[a] > 0.3],
            key=lambda x: -x[1]
        )[:4]

        # --- Assemble result ---
        return {
            "fraud_probability": round(fraud_prob, 4),
            "risk_level":        risk_level,
            "belief_fraud":      round(belief_fraud, 4),
            "belief_legit":      round(belief_legit, 4),
            "uncertainty":       round(uncertainty, 4),

            # Three evidence sources (explainability)
            "evidence_sources": {
                "prototype_cosine": {
                    "bba":            [round(x, 4) for x in proto_bba],
                    "best_match":     best_fraud_type,
                    "best_cosine":    round(best_cosine, 4),
                    "fraud_types":    {
                        k: {"cosine": v["cosine"], "is_fallback": v["is_fallback"]}
                        for k, v in proto_evidence.items()
                    },
                },
                "bayesian_priors": {
                    "bba":           [round(x, 4) for x in bayesian_bba],
                    "active_attrs":  len(bayesian_bbas),
                },
                "graph_path": {
                    "bba":           [round(x, 4) for x in graph_bba],
                    "path_score":    graph_evidence["path_score"],
                    "hub_nodes":     graph_evidence["hub_nodes_found"],
                    "min_depth":     graph_evidence["min_depth"],
                    "matched_nodes": matched_node_ids[:5],
                },
            },

            # Attribute breakdown
            "attributes": {
                attr: {
                    "raw":     round(raw_scores[attr], 3),
                    "boosted": round(boosted_scores[attr], 3),
                    "attn":    round(attn_weights.get(attr, 1.0), 3),
                }
                for attr in ATTRIBUTES
            },

            "top_signals":    top_attrs,
            "reasoning":      reasoning,
            "graph_nodes_in_ontology": len(self.store.nodes),
        }


# ---------------------------------------------------------------------------
# Main detector — drop-in replacement for FraudDetector
# ---------------------------------------------------------------------------

class ZeroShotFraudDetector:
    """
    Drop-in replacement for FraudDetector.
    Uses ZeroShotFraudClassifier instead of DempsterShaferFraudClassifier.
    """

    def __init__(self, store):
        self.classifier = ZeroShotFraudClassifier(store)

    def analyze(self, text: str) -> dict:
        result = self.classifier.classify(text)

        # Format for API response
        return {
            "fraud_probability":  result["fraud_probability"],
            "risk_level":         result["risk_level"],
            "belief_fraud":       result["belief_fraud"],
            "belief_legit":       result["belief_legit"],
            "uncertainty":        result["uncertainty"],
            "top_signals":        result["top_signals"],
            "reasoning":          result["reasoning"],
            "evidence_sources":   result["evidence_sources"],
            "attributes":         result["attributes"],
            "graph_nodes":        result["graph_nodes_in_ontology"],
            "classifier":         "zero_shot_ds_fusion_v1",
        }

    def build_prototypes(self) -> dict:
        """Trigger prototype rebuild from current graph state."""
        return self.classifier.proto_builder.build(verbose=True)
