"""
prototype_builder.py — Ontology-driven fraud prototype construction

Builds fraud TYPE prototypes entirely from the knowledge graph.
No labeled fraud examples needed — this IS the zero-shot part.

How it works:
  1. Walk graph nodes — find nodes that belong to known fraud concept clusters
  2. For each fraud TYPE (advance_fee, phishing, impersonation, etc.),
     collect all related nodes and their descriptions
  3. Score each cluster's attribute profile using the LLM
     (the LLM reads graph node descriptions, not raw user input)
  4. Average scores → prototype vector for that fraud type
  5. Save to data/prototypes.json

At inference time, ZeroShotFraudClassifier loads these prototypes
and computes cosine similarity between input attribute vector
and each fraud type prototype — purely ontology-grounded.
"""

import os
import json
import math
from pathlib import Path

try:
    from groq_agent import call_groq_json
    from graph_store import GraphStore
except ImportError:
    from agents.groq_agent import call_groq_json
    from agents.graph_store import GraphStore

from agents.fraud_detector import ATTRIBUTES, ATTRIBUTE_DESCRIPTIONS

# ---------------------------------------------------------------------------
# Fraud type taxonomy — maps graph node labels to fraud types
# These are the "zero-shot classes" — defined by ontology, not by labels
# ---------------------------------------------------------------------------

FRAUD_TYPE_SEEDS = {
    "advance_fee_fraud": [
        "advance fee", "advance-fee", "nigerian", "419", "lottery scam",
        "prize", "inheritance", "overpayment", "windfall",
    ],
    "impersonation_scam": [
        "irs", "irs impersonation", "government impersonation", "tech support",
        "microsoft", "apple", "bank impersonation", "social security",
        "authority impersonation",
    ],
    "phishing": [
        "phishing", "spear phishing", "credential harvesting", "fake login",
        "email fraud", "smishing", "vishing", "account takeover",
    ],
    "investment_fraud": [
        "ponzi", "pyramid scheme", "investment fraud", "pump and dump",
        "cryptocurrency scam", "affinity fraud", "securities fraud",
    ],
    "romance_scam": [
        "romance scam", "catfishing", "sweetheart scam", "pig butchering",
        "online dating fraud", "confidence trick",
    ],
    "identity_theft": [
        "identity theft", "identity fraud", "synthetic identity",
        "account takeover", "social engineering",
    ],
}

# Attribute profile prompt — reads graph node descriptions, not user input
PROTOTYPE_PROMPT = """You are a fraud researcher building a knowledge base.
Below are descriptions of fraud-related concepts from a knowledge graph,
all belonging to the fraud category: {fraud_type}

Knowledge graph nodes:
---
{node_descriptions}
---

Based ONLY on these knowledge graph descriptions, score how strongly
this fraud TYPE typically exhibits each attribute.
This is a TYPE-LEVEL profile, not a specific instance.

Scoring: 0.0 = not characteristic of this fraud type, 1.0 = defining characteristic

{attribute_list}

Return ONLY valid JSON:
{{
  "urgency": 0.0,
  "anonymity": 0.0,
  "asymmetric_benefit": 0.0,
  "authority_impersonation": 0.0,
  "isolation_tactic": 0.0,
  "information_harvesting": 0.0,
  "payment_irregularity": 0.0,
  "too_good_to_be_true": 0.0,
  "confidence": 0.0,
  "reasoning": "one sentence explaining this fraud type's signature pattern"
}}

confidence = how well the graph nodes characterize this fraud type (0=poor coverage, 1=excellent)
"""


class PrototypeBuilder:
    """
    Builds fraud type prototypes from the knowledge graph.
    Call build() after seeding the ontology.
    """

    def __init__(self, store):
        self.store = store
        self.proto_path = Path("data/prototypes.json")
        self.proto_path.parent.mkdir(parents=True, exist_ok=True)

    def _find_nodes_for_type(self, fraud_type: str) -> list[dict]:
        """
        Find graph nodes related to a fraud type using seed keywords.
        Uses BFS from matched nodes to gather related context.
        """
        seeds = FRAUD_TYPE_SEEDS[fraud_type]
        matched = []
        seen_ids = set()

        for seed in seeds:
            result = self.store.query(seed)
            for node_id in result.get("matches", []):
                if node_id not in seen_ids and node_id in self.store.nodes:
                    matched.append(self.store.nodes[node_id])
                    seen_ids.add(node_id)

            # Also check direct label/desc matches
            for node_id, node in self.store.nodes.items():
                if node_id in seen_ids:
                    continue
                label = node.get("label", "").lower()
                desc  = node.get("desc",  "").lower()
                if any(s in label or s in desc for s in seeds):
                    matched.append(node)
                    seen_ids.add(node_id)

        return matched[:30]  # cap at 30 nodes per type

    def _score_prototype(self, fraud_type: str, nodes: list[dict]) -> dict:
        """
        Score attribute profile for a fraud type from its graph nodes.
        This is what makes it ontology-grounded — the LLM reads
        graph descriptions, not user-submitted text.
        """
        if not nodes:
            # No nodes found — use fallback defaults from research literature
            return self._fallback_prototype(fraud_type)

        node_descriptions = "\n".join(
            f"- {n.get('label','?')} ({n.get('type','concept')}): {n.get('desc','')}"
            for n in nodes[:20]
        )

        attr_list = "\n".join(
            f"  {attr}: {ATTRIBUTE_DESCRIPTIONS[attr]}"
            for attr in ATTRIBUTES
        )

        prompt = PROTOTYPE_PROMPT.format(
            fraud_type=fraud_type.replace("_", " "),
            node_descriptions=node_descriptions,
            attribute_list=attr_list,
        )

        result = call_groq_json(prompt, temperature=0.1)

        scores = {}
        for attr in ATTRIBUTES:
            try:
                scores[attr] = max(0.0, min(1.0, float(result.get(attr, 0.5))))
            except (TypeError, ValueError):
                scores[attr] = 0.5

        return {
            "attributes": scores,
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", ""),
            "node_count": len(nodes),
            "node_labels": [n.get("label", "") for n in nodes[:10]],
        }

    def _fallback_prototype(self, fraud_type: str) -> dict:
        """
        Research-literature fallback when graph has no nodes for this type.
        These are grounded in published fraud taxonomy papers.
        """
        fallbacks = {
            "advance_fee_fraud": {
                "urgency": 0.70, "anonymity": 0.80, "asymmetric_benefit": 0.95,
                "authority_impersonation": 0.50, "isolation_tactic": 0.60,
                "information_harvesting": 0.70, "payment_irregularity": 0.85,
                "too_good_to_be_true": 0.95,
            },
            "impersonation_scam": {
                "urgency": 0.90, "anonymity": 0.70, "asymmetric_benefit": 0.30,
                "authority_impersonation": 0.95, "isolation_tactic": 0.75,
                "information_harvesting": 0.70, "payment_irregularity": 0.85,
                "too_good_to_be_true": 0.20,
            },
            "phishing": {
                "urgency": 0.75, "anonymity": 0.85, "asymmetric_benefit": 0.20,
                "authority_impersonation": 0.80, "isolation_tactic": 0.40,
                "information_harvesting": 0.95, "payment_irregularity": 0.30,
                "too_good_to_be_true": 0.20,
            },
            "investment_fraud": {
                "urgency": 0.60, "anonymity": 0.60, "asymmetric_benefit": 0.90,
                "authority_impersonation": 0.50, "isolation_tactic": 0.55,
                "information_harvesting": 0.65, "payment_irregularity": 0.70,
                "too_good_to_be_true": 0.90,
            },
            "romance_scam": {
                "urgency": 0.55, "anonymity": 0.80, "asymmetric_benefit": 0.75,
                "authority_impersonation": 0.30, "isolation_tactic": 0.90,
                "information_harvesting": 0.70, "payment_irregularity": 0.80,
                "too_good_to_be_true": 0.60,
            },
            "identity_theft": {
                "urgency": 0.65, "anonymity": 0.75, "asymmetric_benefit": 0.20,
                "authority_impersonation": 0.70, "isolation_tactic": 0.50,
                "information_harvesting": 0.95, "payment_irregularity": 0.40,
                "too_good_to_be_true": 0.20,
            },
        }
        attrs = fallbacks.get(fraud_type, {a: 0.5 for a in ATTRIBUTES})
        return {
            "attributes": attrs,
            "confidence": 0.4,  # lower confidence — not from graph
            "reasoning": f"Fallback prototype from research literature (no graph nodes found for {fraud_type})",
            "node_count": 0,
            "node_labels": [],
            "is_fallback": True,
        }

    def build(self, verbose=True) -> dict:
        """
        Build all fraud type prototypes from the current graph state.
        Returns prototype dict and saves to data/prototypes.json.
        """
        if len(self.store.nodes) == 0:
            print("⚠ Graph is empty — seed the ontology first (run seed_fraud_ontology.py)")
            return {}

        prototypes = {
            "_meta": {
                "graph_nodes": len(self.store.nodes),
                "graph_edges": len(self.store.edges),
                "graph_docs":  len(self.store.docs),
                "attributes":  ATTRIBUTES,
            }
        }

        for fraud_type in FRAUD_TYPE_SEEDS:
            if verbose:
                print(f"  Building prototype: {fraud_type}...", end=" ", flush=True)

            nodes = self._find_nodes_for_type(fraud_type)
            proto = self._score_prototype(fraud_type, nodes)
            prototypes[fraud_type] = proto

            if verbose:
                src = "graph" if proto["node_count"] > 0 else "fallback"
                conf = proto["confidence"]
                print(f"✓ ({proto['node_count']} nodes, conf={conf:.2f}, src={src})")

        # Save
        self.proto_path.write_text(json.dumps(prototypes, indent=2))
        if verbose:
            print(f"\n✓ Prototypes saved to {self.proto_path}")

        return prototypes

    def load(self) -> dict:
        """Load existing prototypes from disk."""
        if self.proto_path.exists():
            return json.loads(self.proto_path.read_text())
        return {}

    def needs_rebuild(self) -> bool:
        """True if no prototypes file exists or graph has grown significantly."""
        if not self.proto_path.exists():
            return True
        protos = self.load()
        meta = protos.get("_meta", {})
        saved_nodes = meta.get("graph_nodes", 0)
        current_nodes = len(self.store.nodes)
        # Rebuild if graph grew by more than 20%
        return current_nodes > saved_nodes * 1.2
