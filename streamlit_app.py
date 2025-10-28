import io
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# --- Paths ---
BASE_DIR = Path("/home/ubuntu/datathon")
LINAGE_DIR = BASE_DIR / "pca-clinicalage" / "linAge"
RISK_DIR = BASE_DIR / "risk-assessment"
RECOMMENDATIONS_DIR = BASE_DIR / "recommendations_output"
DEFAULT_QDATA = LINAGE_DIR / "qDataMat.csv"
DEFAULT_DATA = LINAGE_DIR / "dataMat_test.csv"
DEFAULT_PARS = LINAGE_DIR / "linAge_Paras.csv"
PRECOMP_OUTPUT = LINAGE_DIR / "dataMatrix_Normalized_With_Derived_Features_LinAge_PhenoAge.csv"

# Ensure recommendations output directory exists
RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)


# --- Import linAge module ---
sys.path.append(str(LINAGE_DIR))
import linAge as la  # noqa: E402

sys.path.append(str(RISK_DIR))
from infer import load_model as load_diabetes_model 
from infer import load_hypertension_model

# --- Recommendations system imports ---
from recommendations_system.recommender import (
    UserContext,
    ConditionRisk,
    generate_recommendations,
)


# --- Streamlit page setup ---
st.set_page_config(
    page_title="linAge Datathon Demo",
    page_icon="🧬",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* Modern metric cards */
.metric-card {border-radius: 14px; padding: 18px 18px; background: linear-gradient(180deg,#0f172a 0%, #111827 100%);
             border: 1px solid rgba(255,255,255,0.08); color: #e5e7eb;}
.metric-title {font-size: 0.95rem; color: #9ca3af; margin-bottom: 6px;}
.metric-value {font-size: 2rem; font-weight: 700; color: #f8fafc;}
.metric-sub {font-size: 0.95rem; color: #cbd5e1; margin-top: 8px;}
.pill {display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.85rem; margin-left: 8px;}
.pill.good {background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.25)}
.pill.warn {background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.25)}
.pill.bad {background: rgba(244,63,94,0.15); color: #f43f5e; border: 1px solid rgba(244,63,94,0.25)}
.section {margin: 10px 0 22px 0;}
/* Center big heading */
.center {text-align:center;}
/* Range bar styles */
.range-container { margin-top: 8px; }
.range-track { position: relative; height: 10px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.08); overflow: visible; }
.range-marker { position: absolute; top: -8px; width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 10px solid #e5e7eb; }
.range-legend { display:flex; justify-content: space-between; font-size: 0.75rem; color: #cbd5e1; margin-top: 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Helpers ---
def assign_age_bin_label(age_value: float, bin_width: int = 10) -> str:
    if pd.isna(age_value):
        return np.nan
    start = int(np.floor(age_value / bin_width) * bin_width)
    end = start + bin_width - 1
    return f"{start}-{end}"


@st.cache_data(show_spinner=False)
def load_reference_stats() -> pd.DataFrame:
    df = pd.read_csv(PRECOMP_OUTPUT)
    # Build reference stats for BAI from the precomputed cohort
    stats = (
        df.groupby(["sex", "ageBin"])['delAge']
          .agg(bai_mean='mean', bai_std='std')
          .reset_index()
    )
    stats['bai_std'] = stats['bai_std'].replace(0, np.nan)
    return stats


@st.cache_data(show_spinner=False)
def load_default_frames():
    q = pd.read_csv(DEFAULT_QDATA)
    d = pd.read_csv(DEFAULT_DATA)
    keep = (q['RIDAGEYR'] >= 40) & (q['RIDAGEYR'] < 85)
    return d[keep].reset_index(drop=True), q[keep].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_codebook_labels() -> dict:
    """Load Var->Human label mapping from codeBook.csv."""
    codebook_path = BASE_DIR / "pca-clinicalage" / "codeBook.csv"
    if not codebook_path.exists():
        return {}
    cb = pd.read_csv(codebook_path, dtype=str)
    if 'Var' in cb.columns and 'Human' in cb.columns:
        mapping = (
            cb[['Var', 'Human']]
              .dropna(subset=['Var'])
              .set_index('Var')['Human']
              .to_dict()
        )
        return mapping
    return {}


st.markdown("""
<style>
    .metric-card {
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        color: white;
        border: 1px solid #333;
        min-height: 160px;  /* Increased to fit longer sub text */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;  /* Ensures full height within column */
    }
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: #ccc;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: white;
        margin-bottom: 8px;
        flex-shrink: 0;  /* Prevents value from shrinking */
    }
    .metric-sub {
        font-size: 12px;
        color: #aaa;
        line-height: 1.4;
        flex-grow: 1;  /* Allows sub to expand and fill space */
        word-wrap: break-word;  /* Ensures long words wrap */
    }
    .center {
        text-align: center;
    }
    .section {
        margin: 20px 0;
    }
    /* Ensure columns stretch to equal height */
    .stColumns > div {
        display: flex;
    }
    .stColumns > div > div {
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)

# Recommendations cards styling
st.markdown(
    """
<style>
  .rec-card {background:#0b1220; border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:16px; margin:8px 0; color:#e5e7eb}
  .rec-title {font-size:16px; font-weight:700; margin-bottom:6px;}
  .rec-desc {font-size:13px; color:#cbd5e1; margin-bottom:10px;}
  .rec-actions {margin:0; padding-left:18px;}
  .rec-actions li {margin-bottom:6px;}
  .rec-cites {font-size:12px; color:#94a3b8; margin-top:8px;}
</style>
""",
    unsafe_allow_html=True,
)

def make_metric_card(title: str, value: str, sub: str = "", pill: tuple | None = None, extra_html: str | None = None):
    pill_html = ""
    if pill is not None:
        pill_text, pill_kind = pill
        pill_html = f'<span class="pill {pill_kind}">{pill_text}</span>'
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-title">{title}{pill_html}</div>
          <div class="metric-value">{value}</div>
          {f'<div class="metric-sub">{sub}</div>' if sub else ''}
          {extra_html or ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

def compute_bai_from_reference(del_age: float, sex_code: int, chron_age: float, ref_stats: pd.DataFrame):
    sex_label = {1: 'Male', 2: 'Female'}.get(int(sex_code), 'Unknown')
    age_bin = assign_age_bin_label(float(chron_age))
    row = ref_stats[(ref_stats['sex'] == sex_label) & (ref_stats['ageBin'] == age_bin)]
    if row.empty or pd.isna(del_age):
        return np.nan, 'Unknown'
    mean = float(row['bai_mean'].iloc[0])
    std = float(row['bai_std'].iloc[0])
    if std == 0 or np.isnan(std):
        return np.nan, 'Unknown'
    z = (float(del_age) - mean) / std
    # Category mapping consistent with linAge.compute_bai
    if np.isnan(z):
        cat = 'Unknown'
    elif z > 2.0:
        cat = 'High Risk'
    elif 1.0 < z <= 2.0:
        cat = 'Accelerated Aging'
    elif -1.0 <= z <= 1.0:
        cat = 'Normal'
    else:
        cat = 'Healthy'
    return z, cat


def read_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return pd.read_csv(io.BytesIO(uploaded_file.read()))


def classify_bio_pill(delta_years: float, chron_years: float) -> tuple:
    """Map bio age delta to a pill label and style.
    - Younger: always green
    - Slightly higher (<=1y or <=2%): amber
    - Higher (<=3y or <=5%): amber
    - Higher (>5% and >3y): red
    """
    pct = (abs(delta_years) / chron_years) * 100 if chron_years and chron_years > 0 else 0.0
    if delta_years < 0:
        return ("Younger", "good")
    if delta_years <= 1.0 or pct <= 2.0:
        return ("Slightly higher", "warn")
    if delta_years <= 3.0 or pct <= 5.0:
        return ("Higher", "warn")
    return ("Higher", "bad")


@st.cache_resource(show_spinner=False)
def get_diabetes_model():
    if load_diabetes_model is None:
        return None
    try:
        return load_diabetes_model()
    except Exception as e:
        st.warning(f"Diabetes model unavailable: {e}")
        return None


def classify_risk_pill(prob: float) -> tuple:
    # Simple thresholds for display
    if prob < 0.25:
        return ("Low risk", "good")
    if prob < 0.45:
        return ("Borderline", "warn")
    return ("High Risk", "bad")


def _pos_percent(value: float, min_value: float, max_value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if np.isnan(v):
        return 0.0
    v = float(np.clip(v, min_value, max_value))
    return ((v - min_value) / (max_value - min_value)) * 100.0


def bai_range_html(bai_z: float | None) -> str:
    if bai_z is None or (isinstance(bai_z, float) and np.isnan(bai_z)):
        return ""
    min_v, max_v = -3.0, 3.0
    pos = _pos_percent(bai_z, min_v, max_v)
    bg = "linear-gradient(90deg,#10b981 0%,#10b981 33.33%,#3b82f6 33.33%,#3b82f6 50%,#f59e0b 50%,#f59e0b 66.66%,#ef4444 66.66%,#ef4444 100%)"
    return (
        f'<div class="range-container">'
        f'<div class="range-track" style="background:{bg}">'
        f'<div class="range-marker" style="left: calc({pos:.2f}% - 6px)"></div>'
        f'</div>'
        f'<div class="range-legend"><span>-3</span><span>-1</span><span>0</span><span>+1</span><span>+3</span></div>'
        f'</div>'
    )


def prob_range_html(prob: float | None) -> str:
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return ""
    p = float(np.clip(prob, 0.0, 1.0))
    pos = p * 100.0
    bg = "linear-gradient(90deg,#10b981 0%,#10b981 25%,#f59e0b 25%,#f59e0b 45%,#ef4444 45%,#ef4444 100%)"
    return (
        f'<div class="range-container">'
        f'<div class="range-track" style="background:{bg}">'
        f'<div class="range-marker" style="left: calc({pos:.2f}% - 6px)"></div>'
        f'</div>'
        f'<div class="range-legend"><span>0%</span><span>25%</span><span>45%</span><span>100%</span></div>'
        f'</div>'
    )


def _rescale_top_values_to_original(top_df: pd.DataFrame, model_wrapper) -> pd.DataFrame:
    """Given a top-k details DataFrame with 'feature' and scaled 'value',
    convert 'value' back to original (pre-standardization) using the model's scaler.
    Falls back to input if scaler or mapping is unavailable.
    """
    try:
        pipe = getattr(model_wrapper, "pipeline", None)
        if pipe is None or "prep" not in pipe.named_steps:
            return top_df
        prep = pipe.named_steps["prep"]
        # We expect a ColumnTransformer with a 'num' pipeline containing an optional StandardScaler
        num_pipe = getattr(prep, "named_transformers_", {}).get("num")
        if num_pipe is None:
            return top_df
        scaler = getattr(getattr(num_pipe, "named_steps", {}), "get", lambda _: None)("scaler")
        if scaler is None or not hasattr(scaler, "mean_"):
            return top_df
        # Determine feature order used by scaler
        try:
            names = list(prep.get_feature_names_out())
        except Exception:
            names = list(getattr(model_wrapper, "features", []))
        def _strip_prefix(n: str) -> str:
            return n.split("__", 1)[1] if "__" in n else n
        names = [_strip_prefix(n) for n in names]
        name_to_idx = {n: i for i, n in enumerate(names)}
        means = np.asarray(scaler.mean_)
        scales = np.asarray(getattr(scaler, "scale_", np.ones_like(means)))

        out = top_df.copy()
        if "feature" in out.columns and "value" in out.columns:
            def _inv(row):
                fname = row["feature"]
                sval = row["value"]
                if fname in name_to_idx and pd.notna(sval):
                    j = name_to_idx[fname]
                    return float(sval) * float(scales[j]) + float(means[j])
                return row["value"]
            out["value"] = out.apply(_inv, axis=1)
        return out
    except Exception:
        return top_df


@st.cache_resource(show_spinner=False)
def get_htn_model():
    if load_hypertension_model is None:
        return None
    try:
        return load_hypertension_model()
    except Exception as e:
        st.warning(f"Hypertension model unavailable: {e}")
        return None


# --- Sidebar ---
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This demo estimates biological age with linAge and shows Biological Age Index (BAI)."
    )
    st.markdown("Models and parameters are for demonstration; not medical advice.")


# --- Main layout ---
st.markdown("## 🧬 Amad Biological Age Index")
st.caption("Amad uses CLinAge and PhenoAge to estimate biological age and mortality risk.")

ref_stats = load_reference_stats()
data_df, qdata_df = load_default_frames()

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("### Upload data")

    up_merged = st.file_uploader(
        "Upload lab tests and individual data file",
        type=["csv"],
        key="merged_csv",
    )
    # st.caption("Upload lab tests and individual data file")

with col_right:
    st.markdown("### Or select a predefined sample")
    # Choose first 10 valid rows as stable samples
    sample_rows = data_df.head(10).copy()
    sample_q = qdata_df.head(10).copy()
    sample_labels = []
    for i in range(len(sample_rows)):
        age = int(sample_q.loc[i, 'RIDAGEYR'])
        sex = {1: 'M', 2: 'F'}.get(int(sample_q.loc[i, 'RIAGENDR']), '?')
        seqn = sample_rows.loc[i, 'SEQN'] if 'SEQN' in sample_rows.columns else i + 1
        sample_labels.append(f"SEQN {seqn} • {sex} • {age}y")

    # Initialize selection state
    if 'predef_selection' not in st.session_state:
        st.session_state['predef_selection'] = None
    for i in range(len(sample_labels)):
        st.session_state.setdefault(f"predef_{i}", False)

    def _on_predef_select(idx: int):
        st.session_state['predef_selection'] = idx
        for j in range(len(sample_labels)):
            st.session_state[f"predef_{j}"] = (j == idx)

    # Render checkboxes with callbacks to enforce single selection
    for i, label in enumerate(sample_labels):
        st.checkbox(
            label,
            key=f"predef_{i}",
            on_change=_on_predef_select,
            args=(i,)
        )

    selected_idx = st.session_state.get('predef_selection', None)

run_btn = st.button("Run analysis", type="primary")


def run_pipeline(return_single_row: bool = True):
    # Single merged CSV path
    if up_merged is not None:
        merged = read_uploaded_csv(up_merged)

        # Ensure presence of required questionnaire/demographic columns
        required_q_cols = [
            'BPQ020','DIQ010','HUQ010','HUQ020','HUQ050','HUQ070','KIQ020','MCQ010','MCQ053',
            'MCQ160A','MCQ160B','MCQ160C','MCQ160D','MCQ160E','MCQ160F','MCQ160G','MCQ160I',
            'MCQ160J','MCQ160K','MCQ160L','MCQ220','OSQ010A','OSQ010B','OSQ010C','OSQ060',
            'PFQ056','RIAGENDR','RIDAGEYR'
        ]
        for col in required_q_cols:
            if col not in merged.columns:
                merged[col] = np.nan

        # Required biomarker/data columns from model parameters
        lin_pars = pd.read_csv(DEFAULT_PARS)
        data_pars = lin_pars.loc[lin_pars['parType'] == 'DATA', 'parName'].dropna().unique().tolist()
        minimal_data_cols = [
            'LBXCRPN','SSBNP','LBXCOT','LBDTCSI','LBDHDLSI','LBDSTRSI','URXUCRSI','URXUMASI',
            'LBDSALSI','LBDSCRSI','LBDSGLSI','LBXLYPCT','LBXMCVSI','LBXRDW','LBXSAPSI','LBXWBCSI'
        ]
        # Include BMI for derived feature used by HTN model ('agi')
        minimal_data_cols.append('BMXBMI')
        # Curated diabetes features
        curated_diab = [
            'RIDAGEYR','RIAGENDR','BMXWAIST','LBXGLU','LBXGLUSI','LBDSGLSI','LBXIN','LBXINSI','LBXGH',
            'LBDLDLSI','LBDHDLSI','LBDTRSI','LBDSCRSI','URXUMASI','URXUCRSI','LBXCOT','LBDCOTSI'
        ]
        # Curated hypertension features
        curated_htn = [
            'RIDAGEYR','RIAGENDR','BMXWAIST','BPXSY1','BPXSY2','BPXSY3','BPXSY4','BPXDI1','BPXDI2','BPXDI3','BPXDI4',
            'LBDLDLSI','LBDHDLSI','LBDTRSI','LBDSCRSI','URXUMASI','URXUCRSI','LBXCRPN','LBXGLU','LBXGLUSI','LBDSGLSI','LBXCOT','LBDCOTSI'
        ]
        minimal_data_cols = list(sorted(set(minimal_data_cols + curated_diab + curated_htn)))
        needed_cols = set(data_pars).union(minimal_data_cols)
        for col in needed_cols:
            if col not in merged.columns:
                merged[col] = np.nan

        # Split merged into data and q-data frames
        d_user = merged[[c for c in merged.columns if c in needed_cols]].copy()
        q_user = merged[required_q_cols].copy()

        out = la.run_linage_on_frames(d_user, q_user, str(DEFAULT_PARS), compute_bai_within_sample=False)
        # For single subject compute BAI using reference cohort
        out['ageBin'] = out['chronAge'].apply(assign_age_bin_label)
        out['sexLabel'] = q_user['RIAGENDR'].map({1: 'Male', 2: 'Female'})
        bai_z, bai_cat = compute_bai_from_reference(out['delAge'].iloc[0], int(q_user['RIAGENDR'].iloc[0]), float(out['chronAge'].iloc[0]), ref_stats)
        out['BAI'] = [bai_z]
        out['BAICategory'] = [bai_cat]

        # Build a combined feature frame for risk models in one concat to avoid fragmentation
        add_parts = []
        # Keep only needed cols from user inputs to reduce size
        add_parts.append(d_user)
        add_parts.append(q_user)
        base = out.copy()
        features = pd.concat([base.reset_index(drop=True)] + [p.reset_index(drop=True) for p in add_parts], axis=1)
        # Drop duplicate columns keeping the first occurrence to avoid alignment issues
        features = features.loc[:, ~features.columns.duplicated(keep='first')]
        # Compute derived features expected by models
        if 'chronAge' in features.columns and 'BMXBMI' in features.columns:
            with np.errstate(all='ignore'):
                features['agi'] = (features['chronAge'].astype(float) / features['BMXBMI'].astype(float))

        return out, features

    if selected_idx is not None and selected_idx >= 0:
        
        d_one = data_df.iloc[[selected_idx]].copy()
        q_one = qdata_df.iloc[[selected_idx]].copy()
        out = la.run_linage_on_frames(d_one, q_one, str(DEFAULT_PARS), compute_bai_within_sample=False)
        out['ageBin'] = out['chronAge'].apply(assign_age_bin_label)
        out['sexLabel'] = q_one['RIAGENDR'].map({1: 'Male', 2: 'Female'})
        bai_z, bai_cat = compute_bai_from_reference(out['delAge'].iloc[0], int(q_one['RIAGENDR'].iloc[0]), float(out['chronAge'].iloc[0]), ref_stats)
        out['BAI'] = [bai_z]
        out['BAICategory'] = [bai_cat]
        # Build combined feature frame using concat to avoid fragmentation
        features = pd.concat([out.reset_index(drop=True), d_one.reset_index(drop=True), q_one.reset_index(drop=True)], axis=1)
        features = features.loc[:, ~features.columns.duplicated(keep='first')]
        if 'chronAge' in features.columns and 'BMXBMI' in features.columns:
            with np.errstate(all='ignore'):
                features['agi'] = (features['chronAge'].astype(float) / features['BMXBMI'].astype(float))
        
        return out, features

    st.info("Upload a merged CSV or select one predefined sample.")
    return None


if run_btn:
    result_pair = run_pipeline()
    if result_pair is not None:
        result, features_for_model = result_pair
        if result is None or result.empty:
            st.info("No results to display.")
        else:
            row = result.iloc[0]
        chron = float(row['chronAge'])
        bio = float(row['linAge'])
        delta = bio - chron
        pct = (abs(delta) / chron) * 100 if chron > 0 else 0.0
        pct_txt = f"{pct:.1f}%"
        age_dir = "younger" if delta < 0 else "older"
        emoji = " 🎉" if delta < 0 else ""

        st.markdown('<div class="center section">', unsafe_allow_html=True)
        st.markdown("### The estimated biological age")
        st.markdown(f"<div class='center' style='font-size:40px;font-weight:800'>{bio:.1f} years{emoji}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            make_metric_card(
                "Chronological age",
                f"{chron:.1f} years",
                sub="Real age in years",
            )
        with c2:
            pill = classify_bio_pill(delta, chron)
            make_metric_card(
                "Biological age",
                f"{bio:.1f} years",
                sub=f"You are {abs(delta):.1f} years {age_dir} than the chronological age. ({pct_txt} age {'reduction' if delta < 0 else 'increase'})",
                pill=pill,
            )
        with c3:
            bai_val = row.get('BAI', np.nan)
            bai_cat = row.get('BAICategory', 'Unknown')
            pill_kind = 'good' if (isinstance(bai_val, (int, float)) and not np.isnan(bai_val) and bai_val < -1.0) else ('bad' if (isinstance(bai_val, (int, float)) and not np.isnan(bai_val) and bai_val > 1.0) else 'warn')
            make_metric_card(
                "Age Index (BAI)",
                f"{bai_val:.2f}" if pd.notna(bai_val) else "—",
                sub=f"{bai_cat}" if isinstance(bai_cat, str) else "",
                pill=(bai_cat, pill_kind),
                extra_html=bai_range_html(float(bai_val)) if pd.notna(bai_val) else "",
            )

        st.markdown("")
        st.markdown(
            f"You have the mortality risk of an average {bio:.1f}-year-old person.")

        st.markdown("")
        # Diabetes risk card (uses trained model over linAge/pheno features)
        diab_prob = None
        explain = None
        try:
            model = get_diabetes_model()
            if model is not None:
                explain = model.explain_from_frame(features_for_model.iloc[[0]], top_k=15)
                diab_prob = float(explain["probability"])
                dr_pill = classify_risk_pill(diab_prob)
                c4, _ = st.columns([1, 2])
                with c4:
                    make_metric_card(
                        "Type 2 Diabetes risk",
                        f"{diab_prob*100:.1f}%",
                        sub="Estimated probability based on provided data",
                        pill=dr_pill,
                        extra_html=prob_range_html(diab_prob),
                    )
                # Explanations (only if risk > 15%)
                if diab_prob > 0.15:
                    with st.expander("Why this diabetes risk? Top factors"):
                        codebook_labels = load_codebook_labels()
                        codebook_labels.update({"chronAge": "Chronological age", "linAge": "Biological age"})
                        def _label(n):
                            return codebook_labels.get(n, n)
                        pos = explain["top_positive"][ ["feature","value","coef","contribution"] ].copy()
                        # Rescale displayed 'value' back to original units
                        pos = _rescale_top_values_to_original(pos, model)
                        # Keep only positive coefficients (risk-increasing)
                        pos = pos[pos["coef"] > 0]
                        # Sort by coefficient descending
                        pos = pos.sort_values("coef", ascending=False).reset_index(drop=True)
                        pos["feature"] = pos["feature"].map(_label)

                        st.markdown("#### Factors increasing risk")
                        st.dataframe(pos.style.format({"value": "{:.3f}", "coef": "{:.3f}", "contribution": "{:.3f}"}))
        except Exception as e:
            st.info(f"Diabetes risk unavailable: {e}")

        # Hypertension risk card (no bioage features used in training)
        htn_prob = None
        explain_htn = None
        try:
            htn_model = get_htn_model()
            if htn_model is not None:
                explain_htn = htn_model.explain_from_frame(features_for_model.iloc[[0]], top_k=15)
                htn_prob = float(explain_htn["probability"])
                htn_pill = classify_risk_pill(htn_prob)
                c5, _ = st.columns([1, 2])
                with c5:
                    make_metric_card(
                        "Hypertension risk",
                        f"{htn_prob*100:.1f}%",
                        sub="Estimated probability based on clinical and lab features (no bioage)",
                        pill=htn_pill,
                        extra_html=prob_range_html(htn_prob),
                    )
                if htn_prob > 0.15:
                    with st.expander("Why this hypertension risk? Top factors"):
                        codebook_labels = load_codebook_labels()
                        codebook_labels.update({"chronAge": "Chronological age", "linAge": "Biological age"})
                        def _label(n):
                            return codebook_labels.get(n, n)
                        pos = explain_htn["top_positive"][ ["feature","value","coef","contribution"] ].copy()
                        # Rescale displayed 'value' back to original units
                        pos = _rescale_top_values_to_original(pos, htn_model)
                        # Keep only positive coefficients (risk-increasing)
                        pos = pos[pos["coef"] > 0]
                        # Sort by coefficient descending
                        pos = pos.sort_values("coef", ascending=False).reset_index(drop=True)
                        pos["feature"] = pos["feature"].map(_label)
                        st.markdown("#### Factors increasing risk")
                        print(pos)
                        st.dataframe(pos.style.format({"value": "{:.3f}", "coef": "{:.3f}", "contribution": "{:.3f}"}))
        except Exception as e:
            st.info(f"Hypertension risk unavailable: {e}")

        # --- Recommendations section ---
        try:
            st.markdown("### Personalized recommendations from clinical guidelines")

            # Build user context
            sex_label = row.get("sexLabel", "")
            bmi_val = None
            w_kg = None
            h_cm = None
            if "BMXBMI" in features_for_model.columns:
                try:
                    bmi_val = float(features_for_model.iloc[0]["BMXBMI"])
                except Exception:
                    bmi_val = None
            if "BMXWT" in features_for_model.columns:
                try:
                    w_kg = float(features_for_model.iloc[0]["BMXWT"])
                except Exception:
                    w_kg = None
            if "BMXHT" in features_for_model.columns:
                try:
                    h_cm = float(features_for_model.iloc[0]["BMXHT"])
                except Exception:
                    h_cm = None

            user_ctx = UserContext(
                age_years=float(row.get("chronAge", np.nan)),
                sex=str(sex_label) if isinstance(sex_label, str) else str(sex_label),
                bmi=bmi_val,
                weight_kg=w_kg,
                height_cm=h_cm,
                bio_age_years=float(row.get("linAge", np.nan)),
                bai_z=(float(row.get("BAI")) if pd.notna(row.get("BAI", np.nan)) else None),
            )

            # Collect risk conditions and top risk factors (human labels when available)
            codebook_labels = load_codebook_labels()
            codebook_labels.update({"chronAge": "Chronological age", "linAge": "Biological age"})
            def _label(n: str) -> str:
                return codebook_labels.get(n, n)

            risks = []
            if diab_prob is not None and explain is not None:
                pos_df = explain["top_positive"][ ["feature","coef","contribution"] ].copy()
                pos_df = pos_df[pos_df["coef"] > 0].sort_values("contribution", ascending=False).head(5)
                diab_factors = [_label(str(f)) for f in pos_df["feature"].tolist()]
                risks.append(ConditionRisk(name="diabetes", probability=float(diab_prob), top_risk_factors=diab_factors))
            if htn_prob is not None and explain_htn is not None:
                pos_df = explain_htn["top_positive"][ ["feature","coef","contribution"] ].copy()
                pos_df = pos_df[pos_df["coef"] > 0].sort_values("contribution", ascending=False).head(5)
                htn_factors = [_label(str(f)) for f in pos_df["feature"].tolist()]
                risks.append(ConditionRisk(name="hypertension", probability=float(htn_prob), top_risk_factors=htn_factors))

            if not risks:
                st.info("Risk results unavailable; recommendations need risk estimates to tailor guidance.")
            else:
                # Get patient SEQN first
                seqn = "unknown"
                if "SEQN" in features_for_model.columns:
                    try:
                        seqn = str(int(features_for_model.iloc[0]["SEQN"]))
                    except Exception:
                        pass
                
                # Check if recommendations already exist for this patient
                existing_files = list(RECOMMENDATIONS_DIR.glob(f"recommendations_SEQN_{seqn}_*.json"))
                existing_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
                
                rec_data = None
                recs = []
                
                if existing_files:
                    # Load existing recommendations
                    latest_file = existing_files[0]
                    try:
                        with open(latest_file, "r", encoding="utf-8") as f:
                            rec_data = json.load(f)
                        
                        # Convert back to Recommendation objects for rendering
                        from recommendations_system.recommender import Recommendation
                        recs = [
                            Recommendation(
                                title=r.get("title", ""),
                                description=r.get("description", ""),
                                actions=r.get("actions", []),
                                citations=r.get("citations", []),
                            )
                            for r in rec_data.get("recommendations", [])
                        ]
                        
                        # st.info(f"📂 Loaded existing recommendations from: `{latest_file.name}`")
                    except Exception as e:
                        st.warning(f"Could not load existing file, generating new recommendations. Error: {e}")
                        existing_files = []  # Force regeneration
                
                if not existing_files:
                    # Generate new recommendations
                    with st.spinner("🔍 Generating personalized recommendations..."):
                        time.sleep(3)  # 3 second delay
                        recs = generate_recommendations(user=user_ctx, risks=risks, top_k=6)
                    
                    if recs:
                        # Prepare recommendation data for saving
                        rec_data = {
                            "patient_id": seqn,
                            "timestamp": datetime.now().isoformat(),
                            "patient_info": {
                                "age": float(row.get("chronAge", 0)),
                                "sex": str(sex_label),
                                "bmi": float(bmi_val) if bmi_val is not None else None,
                                "bio_age": float(row.get("linAge", 0)),
                                "bai_z": float(row.get("BAI")) if pd.notna(row.get("BAI", np.nan)) else None,
                            },
                            "risks": [
                                {
                                    "condition": r.name,
                                    "probability": r.probability,
                                    "top_factors": r.top_risk_factors,
                                }
                                for r in risks
                            ],
                            "recommendations": [
                                {
                                    "title": rec.title,
                                    "description": rec.description,
                                    "actions": rec.actions,
                                    "citations": rec.citations,
                                }
                                for rec in recs
                            ],
                        }
                        
                        # Save to file
                        rec_file = RECOMMENDATIONS_DIR / f"recommendations_SEQN_{seqn}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(rec_file, "w", encoding="utf-8") as f:
                            json.dump(rec_data, f, indent=2, ensure_ascii=False)
                        
                        # st.success(f"✅ New recommendations saved to: `{rec_file.name}`")
                
                if not recs:
                    st.info("No recommendations could be generated from the loaded guidelines.")
                else:
                    # Render as blocks
                    for rec in recs:
                        actions_html = "".join([f"<li>{a}</li>" for a in rec.actions])
                        # Create clickable citation links
                        cite_links = []
                        for c in rec.citations:
                            source = c.get('source', '')
                            page_start = c.get('page_start', '')
                            page_end = c.get('page_end', '')
                            if source:
                                # Create absolute path to the PDF file
                                pdf_path = f"/home/ubuntu/datathon/recommendations_system/guidelines/{source}"
                                cite_links.append(f'<a href="file://{pdf_path}" target="_blank" style="color: #4A90E2; text-decoration: underline;">{source} p.{page_start}-{page_end}</a>')
                        cites_html = "; ".join(cite_links)
                        st.markdown(
                            f"""
                            <div class='rec-card'>
                              <div class='rec-title'>{rec.title}</div>
                              <div class='rec-desc'>{rec.description}</div>
                              <ul class='rec-actions'>
                                {actions_html}
                              </ul>
                              {f"<div class='rec-cites'>Citations: {cites_html}</div>" if cites_html else ''}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    
                    # Add regenerate button if showing existing recommendations
                    st.markdown("")
                    # if existing_files:
                    #     if st.button("🔄 Regenerate Recommendations", key="regen_recs"):
                    #         # Delete existing files for this patient to force regeneration
                    #         for old_file in existing_files:
                    #             try:
                    #                 old_file.unlink()
                    #             except Exception:
                    #                 pass
                    #         st.rerun()
        except Exception as e:
            st.info(f"Recommendations unavailable: {e}")

        with st.expander("Details (inputs and derived metrics)"):
            # Load human-readable labels from codeBook.csv, and add custom labels for derived fields
            codebook_labels = load_codebook_labels()
            custom_labels = {
                'chronAge': 'Chronological age (years)',
                'linAge': 'Biological age (linAge, years)',
                'delAge': 'LinAge delta (bio - chrono, years)',
                'phenoAge': 'PhenoAge (years)',
                'phenoDelAge': 'PhenoAge delta (years)',
                'BAI': 'Biological Age Index (z-score)',
                'BAICategory': 'BAI Category',
                'sexLabel': 'Sex',
                'ageBin': 'Age bin (years)',
                'LDLV': 'LDL Cholesterol (Friedewald)',
                'crAlbRat': 'Urine Albumin-to-Creatinine ratio',
                'fs1Score': 'Frailty Score (FS1)',
                'fs2Score': 'Frailty Score (FS2)',
                'fs3Score': 'Frailty Score (FS3)'
            }
            labels_map = {**codebook_labels, **custom_labels}

            # Use only the first row for details; map keys to friendly labels
            details = result.iloc[[0]].T.rename(columns={0: 'value'})
            # Drop artifact index if present
            if 'Unnamed: 0' in details.index:
                details = details.drop(index=['Unnamed: 0'])
            # Apply human-friendly labels
            details.index = [labels_map.get(k, k) for k in details.index]
            # Round numeric values to 3 decimals, keep others as-is, and display as strings
            numeric_series = pd.to_numeric(details['value'], errors='coerce')
            def _fmt_num(x: float) -> str:
                if pd.isna(x):
                    return ""
                s = f"{x:.3f}"
                s = s.rstrip('0').rstrip('.')
                return s
            formatted_numeric = numeric_series.round(3).map(_fmt_num)
            display_df = details.copy()
            display_df['value'] = details['value'].astype(str)
            mask = numeric_series.notna()
            display_df.loc[mask, 'value'] = formatted_numeric[mask]

            st.dataframe(display_df)
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button("Download results CSV", data=csv, file_name="linage_results.csv", mime="text/csv")


