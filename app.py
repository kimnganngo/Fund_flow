import streamlit as st
import pandas as pd
import networkx as nx
from unidecode import unidecode
from pyvis.network import Network
from community import community_louvain
import plotly.graph_objects as go
import io

# ====================== CẤU HÌNH ỨNG DỤNG ======================
st.set_page_config(page_title="🔍 Fund Flow Link Analysis", layout="wide")
st.title("🔍 Fund Flow Link Analysis – Phân tích mối liên hệ tài khoản từ dòng tiền")

# ====================== HÀM HỖ TRỢ ==============================
def normalize_name(name):
    if pd.isna(name):
        return ""
    return unidecode(str(name)).strip().lower().replace("  ", " ")

def to_billion(x):
    try:
        return float(x) / 1_000_000_000.0
    except Exception:
        return 0.0

def fmt_ty(x, digits=2):
    try:
        return f"{to_billion(x):,.{digits}f} tỷ"
    except Exception:
        return "0 tỷ"

# ====================== UPLOAD DỮ LIỆU ==========================
st.sidebar.header("📂 Upload file dữ liệu")

file_flow = st.sidebar.file_uploader("Upload file dòng tiền (.csv / .xlsx)", type=["csv", "xlsx"])
file_group = st.sidebar.file_uploader("Upload file nhóm tài khoản (.csv / .xlsx)", type=["csv", "xlsx"])

min_amount = st.sidebar.number_input("Ngưỡng tối thiểu (VND)", min_value=0, value=10_000_000)
graph_mode = st.sidebar.radio(
    "Chế độ đồ thị",
    ["Sankey (trái→phải)", "Network (community)", "PyVis (hierarchical)"],
    index=0
)

# ====================== XỬ LÝ DỮ LIỆU ==========================
if file_flow:
    ext = file_flow.name.split(".")[-1]
    df = pd.read_excel(file_flow) if ext == "xlsx" else pd.read_csv(file_flow)
    df.columns = [c.strip() for c in df.columns]
    df["Nguon_norm"] = df["Người nộp tiền/chuyển khoản"].apply(normalize_name)
    df["Nhan_norm"] = df["Tên nhà đầu tư"].apply(normalize_name)
    df = df[df["Số tiền (VNĐ)"] >= min_amount]

    st.subheader("📋 Dữ liệu dòng tiền đã xử lý")
    st.dataframe(df.head(20), width='stretch')

    # --- Xây đồ thị ---
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src = row["Nguon_norm"]
        dst = row["Nhan_norm"]
        amt = row["Số tiền (VNĐ)"]
        so_lenh = row.get("Số lệnh", 0)
        nop_rut = row.get("Nộp/rút", "")
        ctck = row.get("CTCK", "")

        if not src or not dst:
            continue
        if not G.has_node(src):
            G.add_node(src, label=row["Người nộp tiền/chuyển khoản"], kind="Nguon", amount_out=0, amount_in=0)
        if not G.has_node(dst):
            G.add_node(dst, label=row["Tên nhà đầu tư"], kind="TaiKhoan", amount_out=0, amount_in=0)

        G.nodes[src]["amount_out"] += amt
        G.nodes[dst]["amount_in"] += amt
        if G.has_edge(src, dst):
            G[src][dst]["Tong_tien"] += amt
            G[src][dst]["So_lenh"] += so_lenh
        else:
            G.add_edge(src, dst, Tong_tien=amt, So_lenh=so_lenh, NoP_Rut=nop_rut, CTCK=ctck)

    # --- Community Detection ---
    try:
        partition = community_louvain.best_partition(G.to_undirected())
    except Exception:
        partition = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        partition = {n: i for i, comm in enumerate(partition) for n in comm}
    nx.set_node_attributes(G, partition, "community")

    # --- Centrality ---
    bc = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, bc, "betweenness")

    # --- Xuất bảng nodes / edges ---
    edges_df = pd.DataFrame([(u, v, d["Tong_tien"], d["So_lenh"], d["NoP_Rut"], d["CTCK"])
                             for u, v, d in G.edges(data=True)],
                             columns=["Nguon", "Nhan", "Tong_tien", "So_lenh", "NoP_Rut", "CTCK"])
    nodes_df = pd.DataFrame([(n, d["label"], d["kind"], d["community"],
                              d["amount_in"], d["amount_out"], d["betweenness"])
                             for n, d in G.nodes(data=True)],
                             columns=["Node", "Label", "Loai", "Community",
                                      "Amount_in", "Amount_out", "Betweenness"])

    # --- Thêm cột tỷ đồng ---
    edges_df["Tong_tien_ty"] = edges_df["Tong_tien"].apply(to_billion)
    nodes_df["Amount_in_ty"] = nodes_df["Amount_in"].apply(to_billion)
    nodes_df["Amount_out_ty"] = nodes_df["Amount_out"].apply(to_billion)

    # --- Xuất Excel ---
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        edges_df.to_excel(writer, index=False, sheet_name="edges")
        nodes_df.to_excel(writer, index=False, sheet_name="nodes")
        df.to_excel(writer, index=False, sheet_name="raw_flow")
    st.download_button("📥 Tải toàn bộ dữ liệu (Excel)", data=buf.getvalue(),
                       file_name="fund_flow_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    st.subheader("📊 Kết quả phân tích (theo tỷ đồng)")
    st.dataframe(nodes_df[["Label", "Loai", "Community", "Amount_in_ty", "Amount_out_ty", "Betweenness"]],
                 width='stretch')

    # ================== ĐỒ THỊ ==================
    st.markdown("---")
    st.subheader("📈 Đồ thị luồng tiền")

    palette = ["#ff6b6b", "#845ef7", "#339af0", "#40c057", "#fab005", "#e64980", "#20c997", "#495057"]

    if graph_mode == "Sankey (trái→phải)":
        sources = sorted({G.nodes[n]["label"] for n, d in G.nodes(data=True) if d.get("kind") == "Nguon"})
        accounts = sorted({G.nodes[n]["label"] for n, d in G.nodes(data=True) if d.get("kind") == "TaiKhoan"})
        labels = sources + accounts
        idx = {lab: i for i, lab in enumerate(labels)}
        srcs, tgts, vals, hovers = [], [], [], []
        for u, v, d in G.edges(data=True):
            s, t = G.nodes[u]["label"], G.nodes[v]["label"]
            if s in idx and t in idx:
                val_ty = to_billion(d["Tong_tien"])
                srcs.append(idx[s]); tgts.append(idx[t]); vals.append(val_ty)
                hovers.append(f"{s} → {t}<br>{fmt_ty(d['Tong_tien'])} | Lệnh: {d['So_lenh']} | {d['NoP_Rut']}")
        colors = ["#2b8a3e"]*len(sources) + ["#1971c2"]*len(accounts)
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(label=labels, pad=25, thickness=18, color=colors),
            link=dict(source=srcs, target=tgts, value=vals,
                      customdata=hovers, hovertemplate="%{customdata}<extra></extra>")
        )])
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=650)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Giá trị hiển thị theo **tỷ đồng**")

    elif graph_mode == "Network (community)":
        pos = nx.spring_layout(G.to_undirected(), seed=42, k=0.6)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none")
        node_x, node_y, text, size, color = [], [], [], [], []
        for n, d in G.nodes(data=True):
            x, y = pos[n]
            node_x.append(x); node_y.append(y)
            text.append(f"{d['label']}<br>IN: {fmt_ty(d['amount_in'])} | OUT: {fmt_ty(d['amount_out'])}")
            size.append(max(12, (d['amount_in']+d['amount_out'])**0.25)*3)
            c = d["community"]; color.append(palette[c % len(palette)])
        node_trace = go.Scatter(x=node_x, y=node_y, mode="markers", text=text, hoverinfo="text",
                                marker=dict(size=size, line=dict(width=1)))
        node_trace.marker.color = color
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(showlegend=False, hovermode="closest",
                                         margin=dict(b=20,l=20,r=20,t=20), height=650))
        st.plotly_chart(fig, use_container_width=True)
        png = fig.to_image(format="png", scale=2)
        st.download_button("⬇️ Tải ảnh PNG của đồ thị", data=png, file_name="graph.png", mime="image/png")

    else:
        net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#222222", directed=True)
        net.set_options("""
        const options = {
          layout: { hierarchical: { enabled: true, direction: 'LR', nodeSpacing: 200, levelSeparation: 250 } },
          physics: { enabled: false },
          edges: { arrows: { to: { enabled: true } }, smooth: { type: 'cubicBezier' } }
        }
        """)
        for n, d in G.nodes(data=True):
            color = palette[d["community"] % len(palette)]
            size = max(10, (d["amount_in"] + d["amount_out"]) ** 0.25)
            title = f"{d['label']}<br>IN: {fmt_ty(d['amount_in'])} | OUT: {fmt_ty(d['amount_out'])}"
            level = 0 if d["kind"] == "Nguon" else 1
            net.add_node(n, label=d["label"], title=title, color=color, size=size, level=level)
        for u, v, d in G.edges(data=True):
            title = f"{G.nodes[u]['label']} → {G.nodes[v]['label']} | {fmt_ty(d['Tong_tien'])}"
            net.add_edge(u, v, value=to_billion(d["Tong_tien"]), title=title)
        try:
            html_path = "network.html"
            net.write_html(html_path, notebook=False)
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=680, scrolling=True)
        except Exception as e:
            st.warning(f"PyVis render gặp lỗi: {str(e)}. Tự động chuyển sang Sankey.")
            st.session_state["force_plotly"] = True
