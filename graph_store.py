
import json, hashlib, threading
from pathlib import Path
from collections import deque

DATA_FILE = Path("data/graph.json")

RING_RULES  = {0: {"concept","organization"}, 1: {"method","technology"}, 2: {"application","challenge","person"}}
TYPE_TO_RING = {t: ring for ring, types in RING_RULES.items() for t in types}
DOC_COLORS   = ["#00d4ff","#00e5a0","#8866ff","#f0b840","#ff7744","#ff4488","#ff4466","#44aaff"]

class GraphStore:
    def __init__(self):
        self.nodes: dict = {}   # id → node dict
        self.edges: list = []   # [{source,target,relation,doc}]
        self.docs:  list = []   # [{name,nodes,edges,color}]
        self._lock = threading.Lock()   # ← thread-safe for parallel PDF ingest
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if DATA_FILE.exists():
            try:
                d = json.loads(DATA_FILE.read_text())
                raw_nodes = d.get("nodes", {})
                if isinstance(raw_nodes, list):
                    self.nodes = {n["id"]: n for n in raw_nodes}
                else:
                    self.nodes = raw_nodes
                raw_edges = d.get("edges", [])
                # Normalize s/t → source/target (UI export format)
                fixed = []
                for e in raw_edges:
                    if "s" in e and "source" not in e:
                        fixed.append({"source": e["s"], "target": e["t"],
                                      "relation": e.get("r", "related_to"),
                                      "doc": e.get("doc", "seeded")})
                    else:
                        fixed.append(e)
                self.edges = fixed
                self.docs  = d.get("docs",  [])
            except Exception as e:
                print(f"[GraphStore] load error: {e}")

    def save(self):
        DATA_FILE.write_text(json.dumps({
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "docs":  self.docs,
        }, indent=2))

    def clear(self):
        with self._lock:
            self.nodes.clear(); self.edges.clear(); self.docs.clear()
            self.save()

    def _nid(self, label: str) -> str:
        return label.lower().strip().replace(" ", "_")[:40]

    def _angle(self, nid: str) -> float:
        h = int(hashlib.md5(nid.encode()).hexdigest()[:8], 16)
        return (h % 360) * 3.14159 / 180

    def add_document(self, extraction: dict):
        with self._lock:   # ← serialise concurrent PDF chunk saves
            doc_name = extraction.get("doc_name", "unknown")
            color = DOC_COLORS[len(self.docs) % len(DOC_COLORS)]
            nb, eb = len(self.nodes), len(self.edges)

            for n in extraction.get("nodes", []):
                label = (n.get("label") or "").strip()
                if not label:
                    continue
                nid   = self._nid(label)
                ntype = (n.get("type") or "concept").lower()
                if nid in self.nodes:
                    existing = self.nodes[nid]
                    if not existing.get("desc") and n.get("desc"):
                        existing["desc"] = n["desc"]
                    srcs = existing.setdefault("sources", [existing.get("doc", doc_name)])
                    if doc_name not in srcs:
                        srcs.append(doc_name)
                    existing["crossDoc"] = len(srcs) > 1
                else:
                    ring = TYPE_TO_RING.get(ntype, 2)
                    self.nodes[nid] = {
                        "id": nid, "label": label, "type": ntype,
                        "desc": n.get("desc", ""),
                        "ring": ring, "angle": self._angle(nid),
                        "sources": [doc_name], "doc": doc_name, "crossDoc": False,
                    }

            seen = {(e["source"], e["target"]) for e in self.edges}
            for e in extraction.get("edges", []):
                src = self._nid(e.get("source", ""))
                tgt = self._nid(e.get("target", ""))
                if src in self.nodes and tgt in self.nodes and (src, tgt) not in seen:
                    self.edges.append({"source": src, "target": tgt,
                                       "relation": e.get("relation", "related_to"),
                                       "doc": doc_name})
                    seen.add((src, tgt))

            existing_doc = next((d for d in self.docs if d["name"] == doc_name), None)
            if existing_doc:
                existing_doc["nodes"] += len(self.nodes) - nb
                existing_doc["edges"] += len(self.edges) - eb
            else:
                self.docs.append({"name": doc_name,
                                  "nodes": len(self.nodes) - nb,
                                  "edges": len(self.edges) - eb,
                                  "color": color})
            self.save()

    def to_ui_json(self) -> dict:
        edges_ui = [{"s": e["source"], "t": e["target"],
                     "r": e.get("relation", "related_to")} for e in self.edges]
        return {"nodes": list(self.nodes.values()),
                "edges": edges_ui,
                "sources": self.docs}

    def query(self, q: str) -> dict:
        q_lower = q.lower()
        keywords = [w for w in q_lower.replace("?","").replace(",","").split() if len(w) > 3]
        matches = [nid for nid, n in self.nodes.items()
                   if any(kw in n["label"].lower() or kw in n.get("desc","").lower()
                          for kw in keywords)]
        if not matches:
            return {"answer": f"No nodes found matching '{q}'.",
                    "path": [], "path_rels": [], "matches": []}
        adj: dict = {}
        for e in self.edges:
            adj.setdefault(e["source"], []).append((e["target"], e["relation"]))
            adj.setdefault(e["target"], []).append((e["source"], e["relation"]))
        visited, path, rels = {matches[0]}, [matches[0]], []
        q2 = deque([(matches[0], 0)])
        while q2:
            cur, depth = q2.popleft()
            if depth >= 2: continue
            for nb, rel in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb); path.append(nb); rels.append(rel)
                    q2.append((nb, depth + 1))
        summary = [self.nodes[nid]["label"] for nid in path[:5] if nid in self.nodes]
        return {"answer": f"Found {len(matches)} match(es). Key entities: {', '.join(summary)}.",
                "path": path, "path_rels": rels, "matches": matches}

store = GraphStore()
