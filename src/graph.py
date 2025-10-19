import networkx as nx
import pandas as pd
from typing import Tuple, Dict, Any, List

def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        src = f"SRC::{r['Nguoi_nop_norm']}"
        dst = f"ACC::{r['Tai_khoan_norm']}"
        if not G.has_node(src):
            G.add_node(src, label=r["Nguoi_nop_norm"], kind="Nguon")
        if not G.has_node(dst):
            G.add_node(dst, label=r["Tai_khoan_norm"], kind="TaiKhoan")
        if G.has_edge(src, dst):
            G[src][dst]["Tong_tien"] += r.get("Tong_tien", 0)
            G[src][dst]["So_lenh"] += r.get("So_lenh", 0)
        else:
            G.add_edge(src, dst, Tong_tien=r.get("Tong_tien", 0), So_lenh=r.get("So_lenh", 0), NoP_Rut=r.get("NoP_Rut", ""))
    return G

def annotate_nodes_with_stats(G: nx.DiGraph):
    in_amount = {}
    out_amount = {}
    for u, v, d in G.edges(data=True):
        out_amount[u] = out_amount.get(u, 0) + d.get("Tong_tien", 0)
        in_amount[v] = in_amount.get(v, 0) + d.get("Tong_tien", 0)
    bet = nx.betweenness_centrality(G, normalized=True) if len(G) > 0 else {}
    for n in G.nodes():
        G.nodes[n]["amount_in"] = in_amount.get(n, 0)
        G.nodes[n]["amount_out"] = out_amount.get(n, 0)
        G.nodes[n]["betweenness"] = bet.get(n, 0.0)
    return G

def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """Return mapping node -> community id using greedy modularity (always)
    and Louvain when available. Prefer Louvain if installed.
    """
    community_map: Dict[str, int] = {}
    try:
        import community as community_louvain  # python-louvain
        parts = community_louvain.best_partition(G.to_undirected())
        # normalize to 0..k
        label_map = {}
        next_id = 0
        for node, lab in parts.items():
            if lab not in label_map:
                label_map[lab] = next_id
                next_id += 1
            community_map[node] = label_map[lab]
        return community_map
    except Exception:
        pass

    # Fallback: Greedy Modularity
    comms = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
    for cid, comm in enumerate(comms):
        for node in comm:
            community_map[node] = cid
    return community_map

def edges_dataframe(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "Nguon_node": u,
            "Dich_node": v,
            "Nguon_label": G.nodes[u].get("label", ""),
            "Dich_label": G.nodes[v].get("label", ""),
            "Tong_tien": d.get("Tong_tien", 0),
            "So_lenh": d.get("So_lenh", 0),
            "NoP_Rut": d.get("NoP_Rut", ""),
        })
    return pd.DataFrame(rows)

def nodes_dataframe(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for n, d in G.nodes(data=True):
        rows.append({
            "NodeID": n,
            "Label": d.get("label", ""),
            "Loai": d.get("kind", ""),
            "Amount_in": d.get("amount_in", 0),
            "Amount_out": d.get("amount_out", 0),
            "Betweenness": d.get("betweenness", 0.0),
            "Community": d.get("community", None),
        })
    return pd.DataFrame(rows)
