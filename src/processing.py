import re
import pandas as pd
from unidecode import unidecode

# --------- Normalization helpers ---------
def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = unidecode(name)                 # strip Vietnamese diacritics
    name = re.sub(r"\s+", " ", name)      # squeeze whitespace
    name = name.upper()                     # unify casing
    return name

def normalize_account(acc: str) -> str:
    if pd.isna(acc):
        return ""
    acc = str(acc).strip().upper()
    acc = re.sub(r"\s+", "", acc)         # remove inner spaces
    return acc

def parse_amount(x):
    if pd.isna(x):
        return 0
    s = str(x).replace(".", "").replace(",", "")
    s = re.sub(r"[^\d\-]", "", s)
    if s in ("", "-", "--"):
        return 0
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return 0

# --------- Data loaders ---------
FLOW_COLMAP_GUESS = {
    "Nguoi_nop": ["Nguoi nop tien/chuyen khoan", "Nguoi nop", "Nguoi chuyen", "Nguoi_nop", "Nguoi nộp"],
    "Tai_khoan": ["Tai khoan", "Ma TK", "Tài khoản", "TK", "Tai_khoan"],
    "Ten_nha_dau_tu": ["Ten nha dau tu", "Ten NDT", "Ten", "Chu TK", "Ten_chu_tk", "Ten_nha_dau_tu"],
    "Tu_chuyen_khoan": ["Tu chuyen khoan", "Tu_chuyen_khoan", "Tu CK", "Tu_ck", "SelfTransfer", "Tự chuyển khoản"],
    "So_lenh": ["So lenh", "So_lenh", "Lenh", "Count", "Số lệnh"],
    "So_tien": ["So tien (VND)", "So_tien", "So tien", "Tien", "Amount", "Giá trị", "Số tiền (VNĐ)"],
    "NoP_Rut": ["Nop/rut", "Nop_Rut", "Loai", "Type", "Nộp/rút"],
    "CTCK": ["CTCK", "Broker", "Firm"]
}

GROUP_COLMAP_GUESS = {
    "STT_nhom": ["STT nhom", "STT_nhom", "NhomID", "GroupID"],
    "Ma_TK": ["Ma TK", "Tai khoan", "TK", "Tai_khoan"],
    "Ten": ["Ten", "Ten_chu_tk", "Chu TK", "OwnerName"],
    "Moi_quan_he_voi_cong_ty": ["Moi quan he voi cong ty", "MQH", "Relation", "Moi_quan_he_voi_cong_ty"]
}

def _auto_map_columns(df: pd.DataFrame, mapping: dict) -> dict:
    lower = {c.lower(): c for c in df.columns}
    chosen = {}
    for std, candidates in mapping.items():
        for cand in candidates:
            if cand.lower() in lower:
                chosen[std] = lower[cand.lower()]
                break
    return chosen

def load_flow(df: pd.DataFrame) -> pd.DataFrame:
    colmap = _auto_map_columns(df, FLOW_COLMAP_GUESS)
    # rename to standard
    df = df.rename(columns={v: k for k, v in colmap.items()})
    # enforce required columns
    required = ["Nguoi_nop", "Tai_khoan", "Ten_nha_dau_tu", "Tu_chuyen_khoan", "So_lenh", "So_tien", "NoP_Rut"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc '{r}' trong file luồng tiền.")
    # normalize
    df["Nguoi_nop_norm"] = df["Nguoi_nop"].apply(normalize_name)
    df["Tai_khoan_norm"] = df["Tai_khoan"].apply(normalize_account)
    df["Ten_nha_dau_tu_norm"] = df["Ten_nha_dau_tu"].apply(normalize_name)
    df["So_tien_num"] = df["So_tien"].apply(parse_amount)
    # clean NoP_Rut
    df["NoP_Rut"] = df["NoP_Rut"].astype(str).str.strip().str.title()
    return df

def load_group(df: pd.DataFrame) -> pd.DataFrame:
    colmap = _auto_map_columns(df, GROUP_COLMAP_GUESS)
    df = df.rename(columns={v: k for k, v in colmap.items()})
    required = ["STT_nhom", "Ma_TK", "Ten"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc '{r}' trong file nhóm.")
    df["Ma_TK_norm"] = df["Ma_TK"].apply(normalize_account)
    df["Ten_norm"] = df["Ten"].apply(normalize_name)
    if "Moi_quan_he_voi_cong_ty" not in df.columns:
        df["Moi_quan_he_voi_cong_ty"] = ""
    return df

# --------- Business logic ---------
def filter_flow(df: pd.DataFrame, include_self_transfer: bool, nop_rut_filter: set, min_edge_amount: int):
    tmp = df.copy()
    if not include_self_transfer and "Tu_chuyen_khoan" in tmp.columns:
        # True-like values
        tmp = tmp[~tmp["Tu_chuyen_khoan"].astype(str).str.upper().isin(["TRUE", "1", "Y", "YES"])]
    if nop_rut_filter:
        tmp = tmp[tmp["NoP_Rut"].isin(nop_rut_filter)]
    # group edges
    edges = (
        tmp.groupby(["Nguoi_nop_norm", "Tai_khoan_norm", "NoP_Rut"], as_index=False)
           .agg(Tong_tien=("So_tien_num", "sum"), So_lenh=("So_lenh", "sum"))
    )
    edges = edges[edges["Tong_tien"] >= min_edge_amount]
    return tmp, edges

def join_group(flow_df: pd.DataFrame, group_df: pd.DataFrame) -> pd.DataFrame:
    if group_df is None or group_df.empty:
        out = flow_df.copy()
        out["STT_nhom"] = pd.NA
        out["Ten_nhom"] = pd.NA
        out["Moi_quan_he_voi_cong_ty"] = pd.NA
        return out

    # many-to-one join — keep all memberships
    merged = flow_df.merge(
        group_df[["STT_nhom", "Ma_TK_norm", "Ten", "Ten_norm", "Moi_quan_he_voi_cong_ty"]],
        left_on="Tai_khoan_norm",
        right_on="Ma_TK_norm",
        how="left"
    )
    merged = merged.drop(columns=["Ma_TK_norm"])
    merged = merged.rename(columns={"Ten": "Ten_nhom"})
    return merged

def risk_score(row, thresholds):
    score = 0
    # +1: high amount
    if row.get("Tong_tien", 0) >= thresholds.get("amount_high", 5_000_000_000):
        score += 1
    # +2: group has relation flags
    rel = str(row.get("Moi_quan_he_voi_cong_ty", "")).upper()
    if any(k in rel for k in ["NOI BO", "NBTT", "CO DONG LON", "CDL", "EMPLOYEE", "STAFF", "NHAN VIEN"]):
        score += 2
    # -2: self-transfer
    stk = str(row.get("Tu_chuyen_khoan", "")).upper()
    if stk in ["TRUE", "1", "Y", "YES"]:
        score -= 2
    return max(score, 0)
