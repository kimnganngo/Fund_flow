import io
import os
import base64
import pandas as pd
import streamlit as st
from pyvis.network import Network
import plotly.graph_objects as go

from src.processing import (
    load_flow, load_group, filter_flow, join_group, risk_score
)
from src.graph import build_graph, annotate_nodes_with_stats, detect_communities, edges_dataframe, nodes_dataframe

# --- Helpers đơn vị VND -> tỷ đồng ---
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

st.set_page_config(page_title="FlowLink • Phân tích nguồn tiền", layout="wide")

st.title("💸 FlowLink – Phân tích mối quan hệ nguồn tiền giữa các tài khoản chứng khoán")
st.caption("Upload file luồng tiền + (tuỳ chọn) file nhóm danh tính. App chuẩn hoá dữ liệu, tạo network, cảnh báo cross-group, cộng đồng (community detection), và cho phép tải CSV/Excel/PNG/HTML.")

# --------------- Sidebar controls ---------------
with st.sidebar:
    st.header("⚙️ Cấu hình")
    st.markdown("**Upload dữ liệu**")
    up_flow = st.file_uploader("File A – Luồng tiền (.xlsx/.csv)", type=["xlsx", "csv"], key="flow")
    up_group = st.file_uploader("File B – Danh sách nhóm (.xlsx/.csv)", type=["xlsx", "csv"], key="group")
    st.markdown("---")
    include_self = st.toggle("Bao gồm 'Tự chuyển khoản = TRUE'", value=False)
    filter_type = st.multiselect("Loại giao dịch", ["Nộp", "Rút"], default=["Nộp", "Rút"])
    min_amount = st.number_input("Ngưỡng tiền tối thiểu hiển thị trên cạnh (VND)", min_value=0, value=1_000_000_000, step=100_000_000)
    st.markdown("---")
    st.subheader("🎨 Tùy chọn hiển thị")
    graph_engine = st.radio("Engine đồ thị", ["PyVis (HTML)", "Plotly (PNG export)"], index=0, horizontal=False)
    st.markdown("---")
    st.subheader("📦 Dữ liệu mẫu")
    if st.button("Tải dataset mẫu (.zip)"):
        from pathlib import Path
        import zipfile, tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write("sample_data/flow_sample.csv", arcname="sample_data/flow_sample.csv")
            zf.write("sample_data/group_sample.csv", arcname="sample_data/group_sample.csv")
        with open(tmp.name, "rb") as f:
            st.download_button("Download sample_data.zip", data=f.read(), file_name="sample_data.zip", mime="application/zip")

st.markdown("---")

# --------------- Helpers ---------------
def read_any(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded)
    else:
        try:
            return pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="cp1258")

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Sheet1")
    buf.seek(0)
    return buf.read()

# --------------- Data ingestion ---------------
flow_df_raw = read_any(up_flow)
group_df_raw = read_any(up_group)

if flow_df_raw is None:
    st.info("⬅️ Vui lòng upload **File A – Luồng tiền** ở sidebar để bắt đầu. Hoặc bấm 'Tải dataset mẫu' để xem cấu trúc.")
    st.stop()

# load & normalize
try:
    flow_df = load_flow(flow_df_raw)
except Exception as e:
    st.error(f"Lỗi đọc/chuẩn hoá file luồng tiền: {e}")
    st.stop()

if group_df_raw is not None:
    try:
        group_df = load_group(group_df_raw)
    except Exception as e:
        st.error(f"Lỗi đọc/chuẩn hoá file nhóm: {e}")
        group_df = None
else:
    group_df = None

# filter & edges
flow_filtered, edges = filter_flow(
    flow_df, include_self_transfer=include_self, nop_rut_filter=set(filter_type), min_edge_amount=int(min_amount)
)

# join group info for later alert tables (on transaction-level rows)
flow_joined = join_group(flow_filtered, group_df)

st.subheader("📄 Dữ liệu sau xử lý")
st.write("**Luồng tiền (đã chuẩn hoá & áp bộ lọc):**")
st.dataframe(flow_joined.head(200), use_container_width=True)

# --------------- Build Graph ---------------
st.markdown("---")
st.subheader("🕸️ Biểu đồ mạng mối quan hệ")

G = build_graph(edges)
G = annotate_nodes_with_stats(G)

# community detection
comm_map = detect_communities(G)
for n, cid in comm_map.items():
    G.nodes[n]["community"] = cid

# -------- Visualization --------
import networkx as nx
palette = ["#ff6b6b", "#845ef7", "#339af0", "#40c057", "#fab005", "#e64980", "#20c997", "#495057"]

if graph_mode == "Sankey (trái→phải)":
    # Chuẩn bị node list (nguồn + tài khoản), link theo tỷ đồng
    sources = sorted({G.nodes[n]["label"] for n, d in G.nodes(data=True) if d.get("kind") == "Nguon"})
    accounts = sorted({G.nodes[n]["label"] for n, d in G.nodes(data=True) if d.get("kind") == "TaiKhoan"})
    labels = sources + accounts
    idx = {lab: i for i, lab in enumerate(labels)}

    link_src = []
    link_tgt = []
    link_val = []
    link_hover = []

    for u, v, d in G.edges(data=True):
        src_label = G.nodes[u].get("label", "")
        dst_label = G.nodes[v].get("label", "")
        if src_label in idx and dst_label in idx:
            val_ty = to_billion(d.get("Tong_tien", 0))
            if val_ty <= 0:
                continue
            link_src.append(idx[src_label])
            link_tgt.append(idx[dst_label])
            link_val.append(val_ty)
            link_hover.append(
                f"{src_label} → {dst_label}<br>{fmt_ty(d.get('Tong_tien', 0))} | Lệnh: {d.get('So_lenh',0)} | {d.get('NoP_Rut','')}"
            )

    node_colors = (["#2b8a3e"] * len(sources)) + (["#1971c2"] * len(accounts))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        orientation="h",
        node=dict(
            pad=25, thickness=18,
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=link_src, target=link_tgt, value=link_val,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=link_hover
        )
    )])
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=650)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Giá trị cạnh hiển thị theo **tỷ đồng** (value trong Sankey = tỷ).")

elif graph_mode == "Network (community)":
    # Force layout + color theo community, hiển thị số tiền theo tỷ trong hover
    pos = nx.spring_layout(G.to_undirected(), seed=42, k=0.5)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none")

    node_x, node_y, text, size, color = [], [], [], [], []
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        info = (
            f"{d.get('label')}<br>"
            f"IN: {fmt_ty(d.get('amount_in',0))} | OUT: {fmt_ty(d.get('amount_out',0))} | "
            f"BC: {d.get('betweenness',0):.4f} | C: {d.get('community')}"
        )
        text.append(info)
        size.append(max(12, (d.get('amount_in',0)+d.get('amount_out',0)) ** 0.25) * 3)
        cid = d.get("community")
        color.append(palette[(cid if cid is not None else 0) % len(palette)])

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", text=text, hoverinfo="text",
        marker=dict(size=size, line=dict(width=1))
    )
    node_trace.marker.color = color

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode="closest",
                                     margin=dict(b=20,l=20,r=20,t=20), height=650))
    st.plotly_chart(fig, use_container_width=True)
    # PNG export (giữ đơn vị tỷ đồng ở hover)
    png_bytes = fig.to_image(format="png", scale=2)
    st.download_button("⬇️ Tải ảnh PNG của đồ thị", data=png_bytes, file_name="graph.png", mime="image/png")

else:  # PyVis (hierarchical)
    from pyvis.network import Network
    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#222222", directed=True)

    # Bật layout hierarchical trái→phải + tắt physics để node không chồng chéo
    net.set_options("""
    const options = {
      layout: { hierarchical: { enabled: true, direction: 'LR', sortMethod: 'hubsize', nodeSpacing: 220, levelSeparation: 240 } },
      physics: { enabled: false },
      edges: { arrows: { to: { enabled: true } }, smooth: { type: 'cubicBezier' } }
    }
    """)

    for n, d in G.nodes(data=True):
        label = d.get("label", n)
        size = max(10, (d.get("amount_in", 0) + d.get("amount_out", 0)) ** 0.25)
        cid = d.get("community")
        color = palette[cid % len(palette)] if cid is not None else ("#2b8a3e" if d.get("kind") == "Nguon" else "#1971c2")
        title = f"{d.get('kind')} | IN: {fmt_ty(d.get('amount_in',0))} | OUT: {fmt_ty(d.get('amount_out',0))} | BC: {d.get('betweenness',0):.4f} | C: {cid}"

        # level: nguồn = 0, tài khoản = 1 để đảm bảo trái→phải
        level = 0 if d.get("kind") == "Nguon" else 1
        net.add_node(n, label=label, title=title, size=size, color=color, level=level)

    for u, v, ed in G.edges(data=True):
        w = ed.get("Tong_tien", 0)
        width = max(1, min(10, w / 1_000_000_000))
        title = f"{G.nodes[u].get('label')} → {G.nodes[v].get('label')} | {fmt_ty(w)} | Lệnh: {ed.get('So_lenh', 0)} | {ed.get('NoP_Rut', '')}"
        net.add_edge(u, v, value=to_billion(w), title=title)

    try:
        html_path = "network.html"
        net.write_html(html_path, notebook=False)
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=680, scrolling=True)
    except Exception as e:
        st.warning(f"PyVis render gặp lỗi: {str(e)}. Tự động chuyển sang chế độ Sankey.")
        st.session_state["force_plotly"] = True
        st.experimental_rerun()


# --------------- Tables & Alerts ---------------
st.markdown("---")
st.subheader("📊 Thống kê nhanh")

edges_df = edges_dataframe(G)
nodes_df = nodes_dataframe(G)

# Thêm cột tỷ đồng cho bảng
if not edges_df.empty:
    edges_df["Tong_tien_ty"] = edges_df["Tong_tien"].apply(to_billion)
if not nodes_df.empty:
    nodes_df["Amount_in_ty"]  = nodes_df["Amount_in"].apply(to_billion)
    nodes_df["Amount_out_ty"] = nodes_df["Amount_out"].apply(to_billion)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Số nguồn (unique)", len([n for n, d in G.nodes(data=True) if d.get("kind") == "Nguon"]))
with col2:
    st.metric("Số tài khoản (unique)", len([n for n, d in G.nodes(data=True) if d.get("kind") == "TaiKhoan"]))
with col3:
    st.metric("Số cạnh (luồng tiền)", G.number_of_edges())

st.write("**Top nguồn cấp nhiều nhất (theo tổng tiền OUT):**")
top_src = nodes_df[nodes_df["Loai"]=="Nguon"].sort_values("Amount_out", ascending=False).head(20)
st.dataframe(top_src, use_container_width=True)

st.write("**Top tài khoản nhận nhiều nhất (theo tổng tiền IN):**")
top_dst = nodes_df[nodes_df["Loai"]=="TaiKhoan"].sort_values("Amount_in", ascending=False).head(20)
st.dataframe(top_dst, use_container_width=True)

# Alerts
st.markdown("---")
st.subheader("🚨 Bảng cảnh báo (kèm Risk score)")

if group_df is not None and not group_df.empty:
    acc_group = flow_joined[["Tai_khoan_norm", "STT_nhom", "Moi_quan_he_voi_cong_ty"]].drop_duplicates()
    e = edges.merge(acc_group, on="Tai_khoan_norm", how="left")

    src_group_counts = (
        e.groupby(["Nguoi_nop_norm"])
         .agg(
            so_tk=("Tai_khoan_norm", "nunique"),
            so_nhom=("STT_nhom", "nunique"),
            tong_tien=("Tong_tien", "sum")
         )
         .reset_index()
    )

    # Risk score for alerts
    def _risk_from_row(row):
        rel = ""  # aggregate relation hint
        subset = e[e["Nguoi_nop_norm"] == row["Nguoi_nop_norm"]]
        rel = " | ".join(sorted(set(str(x) for x in subset["Moi_quan_he_voi_cong_ty"].dropna())))
        fake_row = {
            "Tong_tien": row["tong_tien"],
            "Moi_quan_he_voi_cong_ty": rel,
            "Tu_chuyen_khoan": "FALSE"
        }
        return risk_score(fake_row, {"amount_high": 5_000_000_000})

    src_group_counts["risk_score"] = src_group_counts.apply(_risk_from_row, axis=1)

    cross_group_alert = src_group_counts[src_group_counts["so_nhom"] > 1].sort_values(["risk_score","so_nhom","tong_tien"], ascending=[False, False, False])
    st.write("**Nguồn cấp tiền vào nhiều nhóm khác nhau (cross-group):**")
    st.dataframe(cross_group_alert, use_container_width=True)

    same_group_alert = src_group_counts[(src_group_counts["so_nhom"] == 1) & (src_group_counts["so_tk"] >= 2)].sort_values(["risk_score","so_tk"], ascending=[False, False])
    st.write("**Nguồn cấp tiền cho ≥2 tài khoản trong cùng 1 nhóm danh tính (very-strong):**")
    st.dataframe(same_group_alert, use_container_width=True)
else:
    st.info("Chưa có file nhóm → bỏ qua cảnh báo theo nhóm.")

# --------------- Downloads ---------------
st.markdown("---")
st.subheader("📥 Tải dữ liệu xuất")

# Excel workbook export
sheets = {
    "nodes": nodes_df,
    "edges": edges_df,
    "flow_processed": flow_joined
}
excel_bytes = to_excel_bytes(sheets)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.download_button("⬇️ edges.csv", data=to_csv_bytes(edges_df), file_name="edges.csv", mime="text/csv")
with c2:
    st.download_button("⬇️ nodes.csv", data=to_csv_bytes(nodes_df), file_name="nodes.csv", mime="text/csv")
with c3:
    st.download_button("⬇️ flow_processed.csv", data=to_csv_bytes(flow_joined), file_name="flow_processed.csv", mime="text/csv")
with c4:
    st.download_button("⬇️ export.xlsx", data=excel_bytes, file_name="export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("© FlowLink – Network + Communities + Alerts + Export.")
