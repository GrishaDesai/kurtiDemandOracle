"""
app.py — Kurti Demand Oracle
Fixes in this version:
  1. KeyError 'product_code' — aggregate_sales returns empty DataFrame
     with correct columns so merge never fails on missing key
  2. HTML rendering in product cards — all card HTML in one
     st.markdown block, no st.* calls inside card loops
  3. Sidebar garbage output — render_sidebar returns values cleanly,
     no st.* calls that leak debug objects
  4. Top-N limit (TOP_N_FINAL) — only top 10 products kept after
     re-ranking, prevents noise from weak matches
  5. Outlier-robust demand estimation — winsorize qty before
     weighted average so one 50-sale product doesn't dominate 4-sale ones
  6. Threshold and pool size hardcoded in CONFIG, not sidebar sliders
"""

import os, io, json, base64, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import torch
import chromadb
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Kurti Demand Oracle",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CONFIG  — all tuning knobs here, not in sidebar
# ══════════════════════════════════════════════════════════════════
DB_DIR         = "fashion_vectors_new"
COLLECTION     = "fashion_products"
MODEL_ID       = "patrickjohncyh/fashion-clip"
GEMINI_FLASH   = "gemini-2.0-flash-001"
GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta/models"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MDL = "google/gemini-2.0-flash-001"
SALES_EXCEL    = "AI ML Task Sheet.xlsx"

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY",  "")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "")

CANDIDATE_POOL  = 50    # ChromaDB retrieves this many per category
FINAL_THRESHOLD = 0.30   # combined score cutoff
TOP_N_FINAL     = 20     # keep only top-N after re-ranking (noise control)
VISUAL_WEIGHT   = 0.55
TAG_WEIGHT      = 0.45

# Winsorize outlier sales at this percentile before averaging
# e.g. 90 → clip top 10% of qty values so one bestseller doesn't dominate
WINSORIZE_PCT   = 90

TAG_WEIGHTS = {
    "fabric"       : 0.25,
    "pattern"      : 0.20,
    "work"         : 0.20,
    "style"        : 0.15,
    "print_density": 0.12,
    "motif"        : 0.08,
}

CONFIDENCE_TIERS = [
    (8,  "High",     "sage"),
    (5,  "Medium",   "gold"),
    (2,  "Low",      "rose"),
    (1,  "Very Low", "rose"),
]

TAG_COLS = [
    "category", "fabric", "pattern", "motif", "work",
    "occasion", "style", "primary_color", "secondary_color",
    "print_density", "ethnic_score",
]

LLM_PROMPT = """
Analyze this kurti / Indian ethnic wear product image.
Return ONLY valid JSON — no markdown, no backticks, no preamble.

{
  "category"       : "",
  "fabric"         : "",
  "pattern"        : "",
  "motif"          : "",
  "work"           : "",
  "occasion"       : "",
  "style"          : "",
  "primary_color"  : "",
  "secondary_color": "",
  "print_density"  : "",
  "ethnic_score"   : ""
}

Allowed values (lowercase only):
  category      : kurta_set, tshirt, top, dress, saree, pants, shirt
  fabric        : cotton, rayon, silk, georgette, linen, chiffon, crepe
  pattern       : solid, floral, ethnic, geometric, abstract, stripes, checks
  motif         : paisley, lotus, leaf, mandala, tribal, elephant, bandhani,
                  zari_vines, chevron, ikat, none
  work          : printed, embroidery, mirror_work, zari, sequins, thread_work,
                  lace, plain
  occasion      : casual, festive, party, office, wedding
  style         : straight, aline, anarkali, flared, oversized, co_ord
  print_density : minimal, medium, heavy
  ethnic_score  : ethnic, western, fusion

Return ONLY the JSON object.
"""

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg        : #08080d;
    --surface   : #10101a;
    --surface2  : #16161f;
    --border    : rgba(255,255,255,0.07);
    --gold      : #d4a853;
    --gold2     : #f0cc80;
    --rose      : #c97b7b;
    --sage      : #7bb89a;
    --text      : #ede9e2;
    --muted     : #6b6878;
    --radius    : 16px;
    --radius-sm : 8px;
}
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 5rem !important; max-width: 1440px; }
header, [data-testid="stHeader"] { background: transparent !important; }

[data-testid="collapsedControl"] {
    visibility: visible !important; color: var(--gold) !important;
    background: var(--surface) !important;
    border: 1px solid rgba(212,168,83,0.35) !important;
    border-radius: 8px !important; padding: 0.3rem 0.45rem !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(212,168,83,0.1) !important;
}
[data-testid="stSidebarCollapseButton"] button {
    color: var(--gold) !important;
    background: rgba(212,168,83,0.08) !important;
    border: 1px solid rgba(212,168,83,0.25) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

body::after {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:9999;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    opacity:.7;
}

.masthead { padding:1.5rem 0 1rem; border-bottom:1px solid var(--border); margin-bottom:2rem; }
.masthead-eyebrow { font-size:.65rem; letter-spacing:.3em; text-transform:uppercase; color:var(--muted); margin-bottom:.5rem; }
.masthead-title { font-family:'Playfair Display',serif; font-size:clamp(2.2rem,4vw,3.4rem); font-weight:400; line-height:1.05; color:var(--text); margin:0; }
.masthead-title em { color:var(--gold); font-style:italic; }
.masthead-rule { height:1px; margin:.8rem 0 0; background:linear-gradient(90deg,var(--gold) 0%,transparent 60%); }

[data-testid="stFileUploader"] {
    background:var(--surface) !important;
    border:1.5px dashed rgba(212,168,83,.4) !important;
    border-radius:var(--radius) !important; padding:1.5rem !important;
}
[data-testid="stFileUploader"]:hover { border-color:var(--gold) !important; }
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p { color:var(--muted) !important; }

.preview-card { border-radius:var(--radius); overflow:hidden; border:1px solid var(--border); box-shadow:0 24px 64px rgba(0,0,0,.6); background:var(--surface); }
.preview-card img { width:100%; display:block; aspect-ratio:3/4; object-fit:cover; }
.preview-footer { padding:.6rem 1rem; border-top:1px solid var(--border); font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); display:flex; justify-content:space-between; }

.stButton > button {
    width:100%;
    background:linear-gradient(135deg,#b8873a,var(--gold),var(--gold2)) !important;
    color:#1a0e00 !important; font-family:'Outfit',sans-serif !important;
    font-weight:600 !important; font-size:.82rem !important;
    letter-spacing:.14em !important; text-transform:uppercase !important;
    border:none !important; border-radius:var(--radius-sm) !important;
    padding:.9rem 2rem !important; box-shadow:0 6px 28px rgba(212,168,83,.28) !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; opacity:.92 !important; }

.sec-head { font-size:.62rem; letter-spacing:.28em; text-transform:uppercase; color:var(--muted); margin:2rem 0 1rem; display:flex; align-items:center; gap:1rem; }
.sec-head::after { content:''; flex:1; height:1px; background:var(--border); }
.sec-head .badge { font-size:.58rem; padding:.2rem .7rem; border-radius:100px; background:rgba(212,168,83,.1); border:1px solid rgba(212,168,83,.25); color:var(--gold); }

.score-row { display:flex; gap:.6rem; margin:.6rem 0 1.4rem; flex-wrap:wrap; align-items:center; }
.score-pill { display:inline-flex; flex-direction:column; align-items:center; padding:.5rem 1rem; border-radius:var(--radius-sm); font-size:.6rem; letter-spacing:.1em; text-transform:uppercase; gap:.15rem; }
.score-pill.visual { background:rgba(123,184,154,.08); border:1px solid rgba(123,184,154,.25); color:var(--sage); }
.score-pill.tag    { background:rgba(212,168,83,.08);  border:1px solid rgba(212,168,83,.25);  color:var(--gold); }
.score-pill.final  { background:rgba(212,168,83,.15);  border:1px solid rgba(212,168,83,.4);   color:var(--gold2); }
.score-pill b { font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:400; }
.score-op { color:var(--muted); font-size:1rem; align-self:center; }

.chips { display:flex; flex-wrap:wrap; gap:.35rem; margin:.4rem 0 1rem; }
.chip { display:inline-flex; align-items:center; gap:.3rem; padding:.28rem .75rem; border-radius:100px; font-size:.68rem; background:var(--surface2); border:1px solid var(--border); color:var(--muted); white-space:nowrap; }
.chip.gold { background:rgba(212,168,83,.07); border-color:rgba(212,168,83,.3); color:var(--gold); }
.chip.sage { background:rgba(123,184,154,.07); border-color:rgba(123,184,154,.3); color:var(--sage); }
.chip.rose { background:rgba(201,123,123,.07); border-color:rgba(201,123,123,.3); color:var(--rose); }

.metrics-row { display:flex; gap:.9rem; margin:1rem 0 1.5rem; }
.m-card { flex:1; background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1.4rem 1.6rem; position:relative; overflow:hidden; transition:transform .2s; }
.m-card:hover { transform:translateY(-3px); }
.m-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--border); }
.m-card.primary::before   { background:linear-gradient(90deg,#b8873a,var(--gold2)); }
.m-card.secondary::before { background:linear-gradient(90deg,var(--sage),rgba(123,184,154,.3)); }
.m-label { font-size:.6rem; letter-spacing:.2em; text-transform:uppercase; color:var(--muted); margin-bottom:.5rem; }
.m-value { font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:400; line-height:1; color:var(--text); }
.m-card.primary   .m-value { color:var(--gold); }
.m-card.secondary .m-value { color:var(--sage); }
.m-unit { font-size:.68rem; color:var(--muted); margin-top:.35rem; }

.conf-badge { display:inline-flex; align-items:center; gap:.5rem; padding:.4rem 1rem; border-radius:100px; font-size:.72rem; font-weight:500; margin:.5rem 0 1rem; }
.conf-badge.sage { background:rgba(123,184,154,.1); border:1px solid rgba(123,184,154,.3); color:var(--sage); }
.conf-badge.gold { background:rgba(212,168,83,.1);  border:1px solid rgba(212,168,83,.3);  color:var(--gold); }
.conf-badge.rose { background:rgba(201,123,123,.1); border:1px solid rgba(201,123,123,.3); color:var(--rose); }

.price-band { display:flex; align-items:baseline; gap:.8rem; margin:.6rem 0 1.5rem; flex-wrap:wrap; }
.price-val     { font-family:'Playfair Display',serif; font-size:1.3rem; color:var(--text); }
.price-val.mid { color:var(--gold); font-size:1.9rem; }
.price-sep     { color:var(--muted); font-size:.65rem; letter-spacing:.1em; text-transform:uppercase; }

/* product grid */
.prod-grid-row { display:flex; gap:.8rem; margin-bottom:.8rem; }
.prod-card {
    flex:1; background:var(--surface); border:1px solid var(--border);
    border-radius:var(--radius); overflow:hidden;
    transition:transform .2s, border-color .2s, box-shadow .2s;
}
.prod-card:hover { transform:translateY(-4px); border-color:rgba(212,168,83,.3); box-shadow:0 16px 40px rgba(0,0,0,.5); }
.prod-thumb-wrap { width:100%; aspect-ratio:2/3; overflow:hidden; background:var(--surface2); }
.prod-thumb-wrap img { width:100%; height:100%; object-fit:cover; display:block; }
.prod-thumb-empty { width:100%; aspect-ratio:2/3; display:flex; align-items:center; justify-content:center; background:var(--surface2); font-size:2rem; opacity:.25; }
.prod-info { padding:.7rem .8rem .85rem; }
.prod-code { font-size:.58rem; letter-spacing:.1em; color:var(--muted); margin-bottom:.35rem; }
.prod-scores { display:flex; gap:.3rem; margin-bottom:.4rem; flex-wrap:wrap; }
.ps-final { font-size:.6rem; padding:.16rem .55rem; border-radius:100px; background:rgba(212,168,83,.12); border:1px solid rgba(212,168,83,.3); color:var(--gold); font-weight:500; }
.ps-vis   { font-size:.58rem; padding:.13rem .48rem; border-radius:100px; background:rgba(123,184,154,.08); border:1px solid rgba(123,184,154,.2); color:var(--sage); }
.ps-tag   { font-size:.58rem; padding:.13rem .48rem; border-radius:100px; background:rgba(212,168,83,.06); border:1px solid rgba(212,168,83,.15); color:var(--muted); }
.prod-price { font-family:'Playfair Display',serif; font-size:1.1rem; color:var(--gold); margin:.3rem 0 .2rem; }
.prod-sold  { font-size:.62rem; color:var(--sage); margin-bottom:.35rem; }
.prod-tags  { display:flex; flex-wrap:wrap; gap:.2rem; }
.pt { font-size:.56rem; padding:.1rem .38rem; border-radius:4px; background:var(--surface2); color:var(--muted); }
.pt.match { background:rgba(212,168,83,.1); color:var(--gold); border:1px solid rgba(212,168,83,.2); }

.info-box { background:rgba(212,168,83,.05); border:1px solid rgba(212,168,83,.18); border-radius:var(--radius-sm); padding:1rem 1.2rem; margin:1rem 0; }
.info-box-title { font-size:.6rem; letter-spacing:.2em; text-transform:uppercase; color:var(--gold); margin-bottom:.5rem; }
.info-row { font-size:.73rem; color:var(--muted); padding:.2rem 0; }
.info-row b { color:var(--text); font-weight:500; }
.info-box.danger { border-color:rgba(201,123,123,.3); }
.info-box.danger .info-box-title { color:var(--rose); }

.empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:3rem 1rem; gap:.8rem; color:var(--muted); text-align:center; }
.empty-icon { font-size:3rem; opacity:.2; }
.empty-text { font-size:.72rem; letter-spacing:.15em; text-transform:uppercase; }

.rule { height:1px; background:var(--border); margin:2rem 0; }
[data-testid="stSpinner"] > div { border-top-color:var(--gold) !important; }
[data-testid="stTabs"] [role="tab"] { font-size:.72rem !important; letter-spacing:.14em !important; text-transform:uppercase !important; color:var(--muted) !important; border:none !important; padding:.5rem 1.2rem !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--gold) !important; border-bottom:2px solid var(--gold) !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:10px; }
</style>
"""

# ══════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_clip():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor, device


@st.cache_resource(show_spinner=False)
def load_chroma():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(name=COLLECTION)


@st.cache_data(show_spinner=False)
def load_sales(path: str) -> pd.DataFrame:
    """
    Always returns a DataFrame with columns:
      code, qty, rate, date
    Even if the file is missing, returns empty DF with those columns
    so downstream merges never raise KeyError.
    """
    empty = pd.DataFrame(columns=["code", "qty", "rate", "date"])
    if not os.path.exists(path):
        return empty
    try:
        df = pd.read_excel(path, sheet_name="sales data")
    except Exception:
        return empty
    df.columns = df.columns.str.strip().str.lower()
    col_map = {}
    for c in df.columns:
        if   "date"  in c:                  col_map[c] = "date"
        elif "code"  in c:                  col_map[c] = "code"
        elif "qty"   in c or "quant" in c:  col_map[c] = "qty"
        elif "rate"  in c or "price" in c:  col_map[c] = "rate"
    df.rename(columns=col_map, inplace=True)
    for col in ["code", "qty", "rate", "date"]:
        if col not in df.columns:
            df[col] = None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.strip()
    df["qty"]  = pd.to_numeric(df["qty"],  errors="coerce").fillna(0)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0)
    return df


@st.cache_resource(show_spinner=False)
def load_drive_map() -> dict:
    if not GOOGLE_API_KEY or not DRIVE_FOLDER_ID:
        return {}
    import urllib.request, urllib.parse
    image_map, page_token = {}, None
    while True:
        params = {
            "q": f"'{DRIVE_FOLDER_ID}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name)",
            "pageSize": "1000", "key": GOOGLE_API_KEY,
        }
        if page_token:
            params["pageToken"] = page_token
        url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            break
        for f in data.get("files", []):
            image_map[os.path.splitext(f["name"])[0]] = f["id"]
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return image_map


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_drive_b64(file_id: str):
    import urllib.request
    url = (f"https://www.googleapis.com/drive/v3/files/"
           f"{file_id}?alt=media&key={GOOGLE_API_KEY}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return "data:image/jpeg;base64," + base64.b64encode(r.read()).decode()
    except Exception:
        return None


def product_img_b64(code: str, drive_map: dict):
    fid = drive_map.get(str(code).strip())
    return fetch_drive_b64(fid) if fid else None


# ══════════════════════════════════════════════════════════════════
# GEMINI KEYS
# ══════════════════════════════════════════════════════════════════
def load_gemini_keys():
    keys = []
    for i in range(1, 11):
        k = os.getenv(f"GEMINI_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    k = os.getenv("GEMINI_KEY", "").strip()
    if k and k not in keys:
        keys.append(k)
    return keys

GEMINI_KEYS    = load_gemini_keys()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "").strip()

# ══════════════════════════════════════════════════════════════════
# BACKGROUND REMOVAL
# ══════════════════════════════════════════════════════════════════
def remove_background(img: Image.Image) -> Image.Image:
    """
    Remove background using rembg (U2Net).
    Returns an RGB image on a white canvas — identical to what
    build_feature_store.py does for indexed product images,
    so query and index embeddings are in the same space.
    """
    try:
        from rembg import remove as rembg_remove
        rgba  = rembg_remove(img)                        # RGBA
        white = Image.new("RGB", rgba.size, (255, 255, 255))
        white.paste(rgba, mask=rgba.split()[3])          # alpha as mask
        return white
    except ImportError:
        # rembg not installed — fall back to original image
        st.warning("⚠️ rembg not installed. Run: pip install rembg")
        return img
    except Exception as e:
        st.warning(f"⚠️ Background removal failed: {e}. Using original image.")
        return img


# ══════════════════════════════════════════════════════════════════
# STEP 1 — LLM TAG EXTRACTION
# ══════════════════════════════════════════════════════════════════
def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _call_gemini(api_key: str, b64: str):
    url = f"{GEMINI_BASE}/{GEMINI_FLASH}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [
            {"text": LLM_PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}],
        "generationConfig": {"temperature": 0},
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        result = r.json()
        if "candidates" in result:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip(), False
        code  = result.get("error", {}).get("code", 0)
        msg   = result.get("error", {}).get("message", "").lower()
        quota = code == 429 or any(
            w in msg for w in ("quota", "rate limit", "resource_exhausted")
        )
        return None, quota
    except Exception:
        return None, False


def _call_openrouter(b64: str):
    if not OPENROUTER_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type" : "application/json",
    }
    payload = {
        "model": OPENROUTER_MDL,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": LLM_PROMPT},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }},
        ]}],
        "temperature": 0, "max_tokens": 500,
    }
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        result = r.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None


def _parse_json(raw: str):
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except Exception:
        return None


def extract_llm_tags(img: Image.Image):
    b64 = _img_to_b64(img)
    for idx, key in enumerate(GEMINI_KEYS):
        text, quota = _call_gemini(key, b64)
        if text:
            tags = _parse_json(text)
            if tags:
                for col in TAG_COLS:
                    tags.setdefault(col, "")
                return tags, f"gemini_key_{idx+1}"
        if quota and idx < len(GEMINI_KEYS) - 1:
            continue
    if OPENROUTER_KEY:
        raw_or = _call_openrouter(b64)
        if raw_or:
            tags = _parse_json(raw_or)
            if tags:
                for col in TAG_COLS:
                    tags.setdefault(col, "")
                return tags, "openrouter"
    return {col: "" for col in TAG_COLS}, "fallback"


# ══════════════════════════════════════════════════════════════════
# STEP 2 — FASHIONCLIP EMBEDDING
# ══════════════════════════════════════════════════════════════════
def embed_image(img: Image.Image, model, processor, device) -> list:
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
    vec = feat[0].cpu().numpy().astype("float32")
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


# ══════════════════════════════════════════════════════════════════
# STEP 3 — CHROMADB CANDIDATE RETRIEVAL
# HARD filter: category only. Large pool. No threshold here.
# ══════════════════════════════════════════════════════════════════
def retrieve_candidates(vector: list, category: str, collection) -> list:
    kw = dict(
        query_embeddings=[vector],
        n_results=CANDIDATE_POOL,
        include=["metadatas", "distances"],
    )
    if category:
        kw["where"] = {"category": {"$eq": category}}
    results = collection.query(**kw)

    candidates = []
    for pid, dist, meta in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        visual_sim = round(float(1 - dist), 4)
        candidates.append({"product_code": pid, "visual_sim": visual_sim, **meta})
    return candidates


# ══════════════════════════════════════════════════════════════════
# STEP 4 — PYTHON RE-RANKING
# combined_score = VISUAL_WEIGHT×visual_sim + TAG_WEIGHT×tag_score
# Filter: combined_score >= FINAL_THRESHOLD
# Limit : keep only TOP_N_FINAL products
# ══════════════════════════════════════════════════════════════════
def compute_tag_score(new_tags: dict, candidate_meta: dict):
    """
    Returns (score 0→1, matched_tags dict).
    Skips any tag where either side is empty — no penalty for missing.
    """
    score, matched = 0.0, {}
    for tag, weight in TAG_WEIGHTS.items():
        nv = str(new_tags.get(tag, "")).strip().lower()
        cv = str(candidate_meta.get(tag, "")).strip().lower()
        if not nv or nv in ("none", "nan") or not cv or cv in ("none", "nan"):
            continue
        if nv == cv:
            score += weight
            matched[tag] = cv
    return round(score, 4), matched


def rerank_candidates(candidates: list, new_tags: dict) -> pd.DataFrame:
    """
    Re-rank pool, apply threshold, keep top-N.
    Returns DataFrame sorted by final_score descending.
    """
    rows = []
    for c in candidates:
        tag_score, matched = compute_tag_score(new_tags, c)
        final_score = round(
            VISUAL_WEIGHT * c["visual_sim"] + TAG_WEIGHT * tag_score, 4
        )
        if final_score >= FINAL_THRESHOLD:
            rows.append({
                **c,
                "tag_score"   : tag_score,
                "final_score" : final_score,
                "matched_tags": matched,
            })

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("final_score", ascending=False)
        .head(TOP_N_FINAL)          # keep only top-N — noise control
        .reset_index(drop=True)
    )
    return df


# ══════════════════════════════════════════════════════════════════
# STEP 5 — SALES AGGREGATION
# FIX: always returns DataFrame with "product_code" column
#      so merge in estimate_demand never raises KeyError
# ══════════════════════════════════════════════════════════════════
def aggregate_sales(sales_df: pd.DataFrame, product_codes: list) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      product_code, total_qty, avg_price, num_transactions
    Always has these columns even when empty, so merge is safe.
    """
    empty = pd.DataFrame(columns=[
        "product_code", "total_qty", "avg_price", "num_transactions"
    ])

    if sales_df.empty or not product_codes:
        return empty

    # ensure "code" column exists after load_sales normalisation
    if "code" not in sales_df.columns:
        return empty

    df = sales_df[sales_df["code"].isin([str(c) for c in product_codes])].copy()
    if df.empty:
        return empty

    metrics = []
    for code, grp in df.groupby("code"):
        total_qty = grp["qty"].sum()
        if total_qty == 0:
            continue
        avg_price = (grp["qty"] * grp["rate"]).sum() / total_qty
        metrics.append({
            "product_code"    : str(code),
            "total_qty"       : int(total_qty),
            "avg_price"       : round(float(avg_price), 2),
            "num_transactions": len(grp),
        })

    if not metrics:
        return empty

    return pd.DataFrame(metrics)


# ══════════════════════════════════════════════════════════════════
# STEP 6 — DEMAND ESTIMATION
# Outlier-robust: winsorize qty before weighted average
# so one product with 50 sales doesn't dominate four with 4 sales
# ══════════════════════════════════════════════════════════════════
def estimate_demand(df_ranked: pd.DataFrame, df_sales: pd.DataFrame) -> dict:
    """
    Merge on product_code (both DFs guaranteed to have this column).
    Winsorize qty at WINSORIZE_PCT percentile before weighted average.
    """
    if df_ranked.empty or df_sales.empty:
        return {}

    # both DFs have "product_code" — safe merge
    df = df_ranked.merge(df_sales, on="product_code", how="inner")
    if df.empty:
        return {}

    w  = df["final_score"].values.astype(float)
    q  = df["total_qty"].values.astype(float)
    p  = df["avg_price"].values.astype(float)
    tw = w.sum()
    if tw == 0:
        return {}

    # winsorize qty — clip outliers at WINSORIZE_PCT percentile
    # e.g. if one product sold 50 and others sold ~4, clip 50 → ~cap
    cap = float(np.percentile(q, WINSORIZE_PCT))
    q_clipped = np.clip(q, 0, cap)

    pred_sales = round(float(np.dot(w, q_clipped) / tw), 1)
    price_min  = round(float(p.min()), 2)
    price_max  = round(float(p.max()), 2)
    price_mid  = round(float(np.dot(w, p) / tw), 2)

    n = len(df)
    confidence, conf_color = "Very Low", "rose"
    for min_count, label, color in CONFIDENCE_TIERS:
        if n >= min_count:
            confidence, conf_color = label, color
            break

    return {
        "predicted_sales": pred_sales,
        "price_min"      : price_min,
        "price_mid"      : price_mid,
        "price_max"      : price_max,
        "n_products"     : n,
        "confidence"     : confidence,
        "conf_color"     : conf_color,
        "winsorize_cap"  : round(cap, 1),
        "reference"      : df.sort_values("final_score", ascending=False).head(5)[[
            "product_code", "final_score", "visual_sim",
            "tag_score", "total_qty", "avg_price",
        ]].to_dict("records"),
    }


# ══════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════
def tag_source_badge(source: str) -> str:
    if source.startswith("gemini_key_"):
        n = source.split("_")[-1]
        return f'<span class="chip sage">✦ Gemini · Key {n}</span>'
    if source == "openrouter":
        return '<span class="chip gold">✦ OpenRouter</span>'
    return '<span class="chip rose">⚠ Fallback</span>'


def render_tag_chips(tags: dict, source: str):
    gold_keys = {"fabric", "work", "pattern", "print_density", "motif"}
    sage_keys = {"occasion", "style", "category"}
    chips = ""
    for k, v in tags.items():
        if not v or str(v).strip() in ("", "none", "nan"):
            continue
        cls = "gold" if k in gold_keys else ("sage" if k in sage_keys else "")
        chips += (
            f'<span class="chip {cls}">'
            f'<b>{k.replace("_"," ").title()}</b>&nbsp;{str(v).strip()}'
            f'</span>'
        )
    st.markdown(
        f'<div class="sec-head">Detected Attributes {tag_source_badge(source)}</div>'
        f'<div class="chips">{chips}</div>',
        unsafe_allow_html=True,
    )


def render_score_breakdown(df_ranked: pd.DataFrame):
    if df_ranked.empty:
        return
    av = df_ranked["visual_sim"].mean()
    at = df_ranked["tag_score"].mean()
    af = df_ranked["final_score"].mean()
    n  = len(df_ranked)
    st.markdown(f"""
    <div class="score-row">
        <div class="score-pill visual"><b>{av:.2f}</b>Avg Visual</div>
        <span class="score-op">× {VISUAL_WEIGHT}</span>
        <div class="score-pill tag"><b>{at:.2f}</b>Avg Tag</div>
        <span class="score-op">× {TAG_WEIGHT} =</span>
        <div class="score-pill final"><b>{af:.2f}</b>Avg Final</div>
        <span class="score-op" style="font-size:.75rem;">
            across {n} products
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_demand_results(result: dict):
    if not result:
        st.markdown("""
        <div class="info-box danger">
            <div class="info-box-title">⚠ No Sales Data</div>
            <div class="info-row">Similar products found but none have matching
            sales records.</div>
            <div class="info-row">Check that <b>product codes</b> in the sales file
            match ChromaDB product codes exactly.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    ps  = result["predicted_sales"]
    pm  = result["price_mid"]
    plo = result["price_min"]
    phi = result["price_max"]
    n   = result["n_products"]
    c   = result["confidence"]
    cc  = result["conf_color"]
    cap = result["winsorize_cap"]

    st.markdown(f"""
    <div class="metrics-row">
        <div class="m-card primary">
            <div class="m-label">Predicted Total Sales</div>
            <div class="m-value">{ps:.0f}</div>
            <div class="m-unit">units · similarity-weighted · outliers capped at {cap:.0f}</div>
        </div>
        <div class="m-card secondary">
            <div class="m-label">Recommended Price</div>
            <div class="m-value">₹{pm:,.0f}</div>
            <div class="m-unit">weighted avg of {n} similar products</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    icon = "●" if cc == "sage" else ("◐" if cc == "gold" else "○")
    st.markdown(
        f'<div class="conf-badge {cc}">{icon}&nbsp;{c} Confidence'
        f'<span style="opacity:.55;margin-left:.4rem;">· {n} similar products used</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;'
        'color:var(--muted);margin-bottom:.5rem;">Price Range of Similar Products</div>'
        f'<div class="price-band">'
        f'<span class="price-val">₹{plo:,.0f}</span>'
        f'<span class="price-sep">min ───</span>'
        f'<span class="price-val mid">₹{pm:,.0f}</span>'
        f'<span class="price-sep">─── recommended</span>'
        f'<span class="price-val">₹{phi:,.0f}</span>'
        f'<span class="price-sep">max</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if result.get("reference"):
        with st.expander("Reference products used in forecast"):
            ref = pd.DataFrame(result["reference"])
            ref.columns = ["Code","Final","Visual","Tag","Total Sold","Avg Price"]
            for col in ["Final","Visual","Tag"]:
                ref[col] = ref[col].round(3)
            st.dataframe(ref, use_container_width=True, hide_index=True)


def render_similar_products(
    df_ranked: pd.DataFrame,
    df_sales : pd.DataFrame,
    new_tags : dict,
    drive_map: dict,
):
    """
    Renders product cards as pure HTML in one st.markdown() call per row.
    Base64 images are embedded directly in the HTML so Streamlit never
    strips them (it only strips <img> tags that arrive as *bare* markdown,
    not ones inside a larger HTML block).
    All card content — image, scores, price, tags — is in one HTML string,
    so nothing can be misinterpreted as raw text.
    """
    if df_ranked.empty:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">👗</span>
            <span class="empty-text">No similar products found</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # merge sales safely — both have "product_code"
    if not df_sales.empty and "product_code" in df_sales.columns:
        df = df_ranked.merge(df_sales, on="product_code", how="left")
    else:
        df = df_ranked.copy()
        df["total_qty"] = None
        df["avg_price"] = None

    COLS_PER_ROW = 5

    # Inject extra CSS for uniform card image heights — done once
    st.markdown("""
    <style>
    .pc-wrap {
        display: flex;
        gap: .8rem;
        margin-bottom: .8rem;
        align-items: stretch;
    }
    .pc {
        flex: 1;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
        transition: transform .2s, border-color .2s, box-shadow .2s;
        display: flex;
        flex-direction: column;
    }
    .pc:hover {
        transform: translateY(-4px);
        border-color: rgba(212,168,83,.3);
        box-shadow: 0 16px 40px rgba(0,0,0,.5);
    }
    .pc-img-box {
        width: 100%;
        aspect-ratio: 2 / 3;
        overflow: hidden;
        background: var(--surface2);
        flex-shrink: 0;
    }
    .pc-img-box img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .pc-img-empty {
        width: 100%;
        aspect-ratio: 2 / 3;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--surface2);
        font-size: 2rem;
        opacity: .25;
    }
    .pc-info {
        padding: .6rem .75rem .8rem;
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: .25rem;
    }
    .pc-code {
        font-size: .58rem;
        letter-spacing: .1em;
        color: var(--muted);
    }
    .pc-scores {
        display: flex;
        gap: .28rem;
        flex-wrap: wrap;
    }
    .pc-s  { font-size: .58rem; padding: .13rem .48rem; border-radius: 100px; }
    .pc-sf { background: rgba(212,168,83,.12); border: 1px solid rgba(212,168,83,.3);  color: var(--gold);  font-weight: 500; }
    .pc-sv { background: rgba(123,184,154,.08); border: 1px solid rgba(123,184,154,.2); color: var(--sage); }
    .pc-st { background: rgba(212,168,83,.06);  border: 1px solid rgba(212,168,83,.15); color: var(--muted); }
    .pc-price { font-family: 'Playfair Display', serif; font-size: 1.05rem; color: var(--gold); margin-top: .1rem; }
    .pc-sold  { font-size: .62rem; color: var(--sage); }
    .pc-tags  { display: flex; flex-wrap: wrap; gap: .2rem; margin-top: .2rem; }
    .pc-t     { font-size: .56rem; padding: .1rem .38rem; border-radius: 4px; background: var(--surface2); color: var(--muted); }
    .pc-tm    { background: rgba(212,168,83,.1); color: var(--gold); border: 1px solid rgba(212,168,83,.2); }
    </style>
    """, unsafe_allow_html=True)

    for i in range(0, len(df), COLS_PER_ROW):
        chunk = df.iloc[i : i + COLS_PER_ROW].reset_index(drop=True)
        cards_html = '<div class="pc-wrap">'

        for _, row in chunk.iterrows():
            code         = str(row.get("product_code", "")).strip()
            final_score  = float(row.get("final_score", 0))
            visual_sim   = float(row.get("visual_sim",  0))
            tag_score    = float(row.get("tag_score",   0))
            matched_tags = row.get("matched_tags", {}) or {}
            price        = row.get("avg_price",  None)
            qty          = row.get("total_qty",  None)

            price_str = f"&#8377;{price:,.0f}" if price and not pd.isna(price) else "&#8212;"
            qty_html  = (
                f'<div class="pc-sold">&#8593; {int(qty)} sold</div>'
                if qty is not None and not pd.isna(qty) else ""
            )

            # ── image ────────────────────────────────────────────
            img_b64 = product_img_b64(code, drive_map)
            if img_b64:
                img_html = (
                    f'<div class="pc-img-box">'
                    f'<img src="{img_b64}" alt="{code}" loading="lazy">'
                    f'</div>'
                )
            else:
                img_html = '<div class="pc-img-empty">&#128249;</div>'

            # ── tag pills ────────────────────────────────────────
            tag_pills = ""
            for tag in ["fabric", "pattern", "work", "style", "motif", "print_density"]:
                val = str(row.get(tag, "")).strip()
                if not val or val.lower() in ("", "nan", "none"):
                    continue
                cls = "pc-t pc-tm" if tag in matched_tags else "pc-t"
                tag_pills += f'<span class="{cls}">{val}</span>'

            cards_html += f"""
<div class="pc">
  {img_html}
  <div class="pc-info">
    <div class="pc-code">{code}</div>
    <div class="pc-scores">
      <span class="pc-s pc-sf">&#10022; {final_score:.2f}</span>
      <span class="pc-s pc-sv">&#128065; {visual_sim:.2f}</span>
      <span class="pc-s pc-st">&#127991; {tag_score:.2f}</span>
    </div>
    <div class="pc-price">{price_str}</div>
    {qty_html}
    <div class="pc-tags">{tag_pills}</div>
  </div>
</div>"""

        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR  — only status info, no sliders (settings in CONFIG)
# FIX: function returns None implicitly — no leaking return values
# ══════════════════════════════════════════════════════════════════
def render_sidebar(drive_map: dict, chroma_count: int) -> None:
    """
    Renders system status only.
    All tuning settings live in CONFIG at top of file.
    Returns None explicitly — no leaked objects.
    """
    with st.sidebar:
        st.markdown("""
        <div style="padding:.6rem 0 1rem;
            border-bottom:1px solid rgba(255,255,255,0.07);
            margin-bottom:1rem;">
            <div style="font-size:.55rem;letter-spacing:.3em;
                text-transform:uppercase;color:#6b6878;margin-bottom:.3rem;">
                System Status
            </div>
            <div style="font-family:'Playfair Display',serif;
                font-size:1.1rem;color:#d4a853;">⚙ Oracle Panel</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**ChromaDB**")
        st.success(f"✅ {chroma_count:,} products indexed")

        st.markdown("**Gemini Keys**")
        if GEMINI_KEYS:
            st.success(f"✅ {len(GEMINI_KEYS)} key(s) loaded")
        else:
            st.warning("⚠️ No GEMINI_KEY_* found in .env")

        st.markdown("**OpenRouter**")
        if OPENROUTER_KEY:
            st.success("✅ Ready (fallback)")
        else:
            st.info("ℹ Not configured")

        st.markdown("**Drive Images**")
        if drive_map:
            st.success(f"✅ {len(drive_map):,} images linked")
        elif GOOGLE_API_KEY and DRIVE_FOLDER_ID:
            st.error("❌ Could not load from Drive")
        else:
            st.info("ℹ Not configured")

        st.markdown("**Sales File**")
        if os.path.exists(SALES_EXCEL):
            st.success(f"✅ {SALES_EXCEL}")
        else:
            st.warning(f"⚠️ {SALES_EXCEL} not found")

        st.divider()
        st.markdown("**Active Settings**")
        st.caption(
            f"Candidate pool : {CANDIDATE_POOL}\n\n"
            f"Final threshold: {FINAL_THRESHOLD}\n\n"
            f"Top-N kept     : {TOP_N_FINAL}\n\n"
            f"Visual weight  : {VISUAL_WEIGHT}\n\n"
            f"Tag weight     : {TAG_WEIGHT}\n\n"
            f"Outlier cap    : {WINSORIZE_PCT}th percentile"
        )
        st.divider()
        st.markdown("**Tag Weights**")
        st.caption(
            "fabric 0.25 · pattern 0.20 · work 0.20\n\n"
            "style 0.15 · print_density 0.12 · motif 0.08"
        )
        st.divider()
        st.caption(
            ".env keys:\n"
            "GEMINI_KEY_1…10\n"
            "OPENROUTER_KEY\n"
            "GOOGLE_API_KEY\n"
            "DRIVE_FOLDER_ID"
        )
    # explicit None return — nothing leaks out
    return None


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    with st.spinner("Loading FashionCLIP…"):
        clip_model, clip_proc, clip_device = load_clip()
    with st.spinner("Connecting to ChromaDB…"):
        collection = load_chroma()

    drive_map = load_drive_map()
    sales_df  = load_sales(SALES_EXCEL)

    # sidebar — returns None, no variables captured
    render_sidebar(drive_map, collection.count())

    st.markdown("""
    <div class="masthead">
        <div class="masthead-eyebrow">AI-Powered · Fashion Intelligence</div>
        <h1 class="masthead-title">Kurti <em>Demand Oracle</em></h1>
        <div class="masthead-rule"></div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.9], gap="large")

    # ── LEFT: upload ──────────────────────────────────────────────
    with left:
        st.markdown(
            '<div class="sec-head">Upload New Design</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64_prev = base64.b64encode(buf.getvalue()).decode()
            w, h = img.size
            st.markdown(f"""
            <div class="preview-card">
                <img src="data:image/png;base64,{b64_prev}" />
                <div class="preview-footer">
                    <span>📷 {uploaded.name}</span>
                    <span>{w} × {h} px</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("✦  Predict Demand")
        else:
            run_btn = False
            st.markdown("""
            <div class="empty-state" style="padding:4rem 1rem;">
                <span class="empty-icon">👗</span>
                <span class="empty-text">Drop an image to begin</span>
            </div>
            """, unsafe_allow_html=True)

    # ── RIGHT: results ────────────────────────────────────────────
    with right:
        if uploaded and run_btn:

            with st.spinner("Removing background…"):
                img_clean = remove_background(img)

            # Step 1
            with st.spinner("Extracting attributes with Gemini…"):
                llm_tags, tag_source = extract_llm_tags(img_clean)

            with st.expander("🔧 Debug: LLM response", expanded=(tag_source == "fallback")):
                st.write("Source:", tag_source)
                st.write("Tags:", llm_tags)
                st.write("Gemini keys loaded:", len(GEMINI_KEYS))
                st.write("OpenRouter key set:", bool(OPENROUTER_KEY))

            # Step 2
            with st.spinner("Generating FashionCLIP embedding…"):
                vector = embed_image(img_clean, clip_model, clip_proc, clip_device)

            # Step 3
            category = str(llm_tags.get("category", "")).strip().lower()
            with st.spinner(
                f"Retrieving top-{CANDIDATE_POOL} candidates "
                f"[category: {category or 'any'}]…"
            ):
                candidates = retrieve_candidates(vector, category, collection)

            # Step 4
            with st.spinner("Re-ranking by combined visual + tag score…"):
                df_ranked = rerank_candidates(candidates, llm_tags)

            # Step 5
            codes = df_ranked["product_code"].tolist() if not df_ranked.empty else []
            df_sales_agg = aggregate_sales(sales_df, codes)

            # Step 6
            result = estimate_demand(df_ranked, df_sales_agg)

            # render tags
            render_tag_chips(llm_tags, tag_source)

            # category warning
            if not category:
                st.markdown("""
                <div class="info-box danger">
                    <div class="info-box-title">⚠ Category Not Detected</div>
                    <div class="info-row">LLM could not identify the category.
                    Search ran without category filter.</div>
                </div>
                """, unsafe_allow_html=True)

            # no results
            if df_ranked.empty:
                st.markdown(f"""
                <div class="info-box danger" style="margin-top:1rem;">
                    <div class="info-box-title">No Similar Products Found</div>
                    <div class="info-row">No products in <b>category = {category or "any"}</b>
                    scored ≥ <b>{FINAL_THRESHOLD}</b> combined.</div>
                    <div class="info-row">Lower <b>FINAL_THRESHOLD</b> in CONFIG to widen the search.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="sec-head">Score Breakdown</div>',
                    unsafe_allow_html=True,
                )
                render_score_breakdown(df_ranked)

                st.markdown(
                    '<div class="sec-head">Demand Forecast</div>',
                    unsafe_allow_html=True,
                )
                render_demand_results(result)

            # persist to session
            st.session_state["df_ranked"]    = df_ranked
            st.session_state["df_sales_agg"] = df_sales_agg
            st.session_state["llm_tags"]     = llm_tags
            st.session_state["category"]     = category

        elif not uploaded:
            st.markdown("""
            <div class="empty-state" style="height:340px;">
                <span class="empty-text">Results will appear here</span>
            </div>
            """, unsafe_allow_html=True)

    # ── SIMILAR PRODUCTS full width ───────────────────────────────
    if st.session_state.get("df_ranked") is not None:
        df_r  = st.session_state["df_ranked"]
        df_s  = st.session_state.get("df_sales_agg", pd.DataFrame())
        tags  = st.session_state.get("llm_tags", {})
        cat   = st.session_state.get("category", "")

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        n = len(df_r)
        st.markdown(
            f'<div class="sec-head">Similar Designs &nbsp;'
            f'<span class="badge">'
            f'{n} of top-{TOP_N_FINAL} · category: {cat or "any"} · '
            f'threshold ≥ {FINAL_THRESHOLD} · '
            f'{VISUAL_WEIGHT}×visual + {TAG_WEIGHT}×tags'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        tab_cards, tab_table = st.tabs(["🎴  Cards", "📋  Table"])

        with tab_cards:
            render_similar_products(df_r, df_s, tags, drive_map)

        with tab_table:
            if not df_r.empty:
                merged = df_r.merge(df_s, on="product_code", how="left") \
                    if not df_s.empty and "product_code" in df_s.columns \
                    else df_r.copy()
                disp = [c for c in [
                    "product_code", "final_score", "visual_sim", "tag_score",
                    "fabric", "pattern", "motif", "work", "style",
                    "print_density", "primary_color", "total_qty", "avg_price",
                ] if c in merged.columns]
                st.dataframe(
                    merged[disp].sort_values("final_score", ascending=False),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No similar products found.")


if __name__ == "__main__":
    main()