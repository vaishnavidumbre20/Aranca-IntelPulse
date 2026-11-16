import os
from pathlib import Path
from datetime import datetime
import html
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_groq_api_key():
    """Read Groq key from Streamlit Cloud Secrets or local environment."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


def generate_with_retry(client, prompt, model="openai/gpt-oss-120b", attempts=3):
    """Retry transient server errors; quota (429) is surfaced immediately."""
    last_error = None
    for attempt in range(attempts):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise competitive intelligence and business research analyst. Use only supplied evidence; never fabricate facts."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1800,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            # Don't retry quota/rate-limit errors; they won't clear in a few seconds.
            if "429" in message or "rate_limit" in message or "quota" in message:
                raise
            retryable = any(term in message for term in ["503", "unavailable", "500", "internal", "timeout"])
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1)
)
    raise last_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Aranca IntelPulse",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "competitor_intelligence.csv"

# ---------------- VISUAL SYSTEM ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:Inter,sans-serif;}
.stApp {background:#f5f7fa;color:#172b45;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #e2e8ef;}
[data-testid="stSidebar"] * {color:#24364a;}
.block-container {max-width:1450px;padding-top:1.6rem;padding-bottom:2.5rem;}
h1,h2,h3,h4 {color:#173451!important;letter-spacing:-.025em;}
p, label {color:#526477;}
.brand {font-size:25px;font-weight:700;color:#174c7e;letter-spacing:-.04em;}
.brand-sub {font-size:10px;letter-spacing:.14em;color:#7b8998;font-weight:600;margin-top:3px;}
.page-title {font-size:32px;font-weight:700;color:#173451;letter-spacing:-.04em;}
.page-sub {font-size:14px;color:#6b7b8d;margin-top:5px;}
.eyebrow {font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#718397;}
.panel {background:#fff;border:1px solid #e2e8ef;border-radius:10px;padding:17px 19px;}
.metric {background:#fff;border:1px solid #e2e8ef;border-radius:9px;padding:15px 17px;}
.metric-label {font-size:12px;color:#718195;font-weight:600;}
.metric-value {font-size:27px;font-weight:700;color:#173e65;margin-top:5px;}
.signal-card {background:#fff;border:1px solid #e2e8ef;border-radius:9px;padding:14px 16px;margin:9px 0;}
.signal-title {font-size:14px;font-weight:600;line-height:1.5;color:#20384f;}
.meta {font-size:12px;color:#77889a;margin-top:5px;}
.tag {display:inline-block;padding:3px 8px;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:.04em;}
.high {background:#fff0ed;color:#a33e31;}
.medium {background:#fff6e5;color:#89631d;}
.low {background:#edf5f0;color:#386b52;}
.ai-panel {background:#fff;border:1px solid #dce5ed;border-left:4px solid #174c7e;border-radius:9px;padding:18px 20px;}
.stButton>button,.stDownloadButton>button {border-radius:6px;border:1px solid #174c7e;background:#174c7e;color:white;font-weight:600;padding:.48rem .85rem;}
.stButton>button:hover,.stDownloadButton>button:hover {background:#123e68;border-color:#123e68;color:white;}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div {background:#fff;border-color:#d8e1e9;}
[data-testid="stDataFrame"] {border:1px solid #e2e8ef;border-radius:8px;overflow:hidden;}
hr {border-color:#e3e9ef;}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="brand">Aranca IntelPulse</div><div class="brand-sub">GROWTH ADVISORY · COMPETITIVE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### Intelligence Filters")

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=60)
def load_data(path, modified):
    data = pd.read_csv(path).fillna("")
    required = ["Company", "Country", "Title", "Published", "Source", "Link", "Category", "Impact"]
    for col in required:
        if col not in data.columns:
            data[col] = ""
    if "Link" in data.columns:
        data = data.drop_duplicates(subset=["Link"], keep="last")
    return data

if not DATA_FILE.exists():
    st.error("competitor_intelligence.csv was not found. Keep it in the same folder as app.py.")
    st.stop()

try:
    df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load intelligence data: {exc}")
    st.stop()

# ---------------- FILTERS ----------------
countries = sorted([x for x in df["Country"].astype(str).unique() if x])
companies = sorted([x for x in df["Company"].astype(str).unique() if x])
categories = sorted([x for x in df["Category"].astype(str).unique() if x])
impacts = [x for x in ["High", "Medium", "Low"] if x in set(df["Impact"].astype(str))]

with st.sidebar:
    country = st.selectbox("Geography", ["All"] + countries)
    category = st.selectbox("Development category", ["All"] + categories)
    company = st.selectbox("Competitor", ["All"] + companies)
    impact = st.multiselect("Impact level", impacts, default=impacts)
    if st.button("Reset filters", use_container_width=True):
        st.rerun()
    st.divider()
    st.caption("Public-source intelligence")
    st.caption("Signals should be validated against their original sources.")

filtered = df.copy()
if country != "All":
    filtered = filtered[filtered["Country"] == country]
if category != "All":
    filtered = filtered[filtered["Category"] == category]
if company != "All":
    filtered = filtered[filtered["Company"] == company]
if impact:
    filtered = filtered[filtered["Impact"].isin(impact)]
else:
    filtered = filtered.iloc[0:0]


# ---------------- PREMIUM WELCOME / AGENT OVERVIEW ----------------
st.markdown("""
<div style="position:relative;overflow:hidden;background:linear-gradient(120deg,#0f2440 0%,#163b67 58%,#245a91 100%);border-radius:16px;padding:30px 32px;margin-bottom:16px;box-shadow:0 8px 24px rgba(15,35,65,.10);">
  <div style="position:absolute;right:-55px;top:-95px;width:300px;height:300px;border:1px solid rgba(255,255,255,.10);border-radius:50%;"></div>
  <div style="position:absolute;right:5px;top:-45px;width:220px;height:220px;border:1px solid rgba(255,255,255,.10);border-radius:50%;"></div>
  <div style="position:relative;max-width:900px;">
    <div style="font-size:10px;font-weight:700;letter-spacing:.17em;color:#9fc7ff;text-transform:uppercase;">ARANCA INTELPULSE AI · GROWTH ADVISORY</div>
    <div style="font-size:clamp(27px,3vw,38px);font-weight:750;letter-spacing:-.045em;color:#fff;line-height:1.15;margin-top:12px;">Competitive signals.<br/>Clarity for decisions.</div>
    <div style="font-size:14px;line-height:1.75;color:#d4e2f2;max-width:760px;margin-top:13px;">
      Your GenAI intelligence analyst monitors boutique and mid-sized Growth Advisory competitors across five markets—transforming public developments into source-linked insights for Practice Leaders.
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:7px;margin-top:18px;">
      <span style="background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.18);border-radius:20px;padding:5px 11px;font-size:11px;color:#f2f7ff;">US</span>
      <span style="background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.18);border-radius:20px;padding:5px 11px;font-size:11px;color:#f2f7ff;">KSA</span>
      <span style="background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.18);border-radius:20px;padding:5px 11px;font-size:11px;color:#f2f7ff;">UAE</span>
      <span style="background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.18);border-radius:20px;padding:5px 11px;font-size:11px;color:#f2f7ff;">India</span>
      <span style="background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.18);border-radius:20px;padding:5px 11px;font-size:11px;color:#f2f7ff;">Japan</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

welcome_a, welcome_b = st.columns([1.15, 1])
with welcome_a:
    if st.button("✦ Explore today's intelligence", type="primary", use_container_width=True, key="welcome_brief_button"):
        st.session_state["welcome_brief_trigger"] = True
        st.rerun()
with welcome_b:
    if st.button("↻ Monitoring setup & status", use_container_width=True, key="welcome_monitor_button"):
        st.info("Daily collection is handled by daily_monitor.py and your scheduled task. This dashboard shows collected records; this control does not initiate a live scan.")

# Compact, dataset-grounded pulse (not represented as daily counts unless date data supports it)
pulse_total = len(filtered)
pulse_high = int((filtered["Impact"].astype(str).str.lower() == "high").sum()) if "Impact" in filtered else 0
pulse_companies = filtered["Company"].nunique() if "Company" in filtered else 0
st.markdown("""
<div style="margin:20px 0 8px;font-size:10px;font-weight:700;letter-spacing:.15em;color:#718397;text-transform:uppercase;">INTELLIGENCE SNAPSHOT · CURRENT FILTERS</div>
""", unsafe_allow_html=True)
pulse1, pulse2, pulse3 = st.columns(3)
for col, label, value, note in [
    (pulse1, "Developments in view", f"{pulse_total:,}", "Records matching selected filters"),
    (pulse2, "High-impact signals", f"{pulse_high:,}", "Priority items for review"),
    (pulse3, "Competitors represented", f"{pulse_companies:,}", "Companies in current dataset view"),
]:
    col.markdown(f"""<div style="background:#fff;border:1px solid #e2e8ef;border-radius:10px;padding:15px 17px;">
    <div style="font-size:11px;font-weight:600;color:#718195;">{label}</div>
    <div style="font-size:27px;font-weight:750;letter-spacing:-.03em;color:#173e65;margin:5px 0;">{value}</div>
    <div style="font-size:11px;color:#94a3b8;">{note}</div></div>""",unsafe_allow_html=True)

st.markdown("#### Intelligence capabilities")
cap1, cap2, cap3, cap4 = st.columns(4)
capabilities = [
    (cap1, "↗ Business Updates", "Growth signals, new services, revenue announcements and geographic expansion."),
    (cap2, "⇄ Deals & Engagements", "Client wins, project announcements, strategic partnerships and engagements."),
    (cap3, "◇ M&A Activity", "Acquisitions, mergers, investments and funding announcements."),
    (cap4, "◎ Digital Presence", "Website updates, LinkedIn posts, blogs, articles and press releases."),
]
for col, title, desc in capabilities:
    with col:
        st.markdown(f"""<div style="height:145px;background:white;border:1px solid #e2e8ef;border-radius:10px;padding:16px;">
        <div style="font-size:13px;font-weight:700;color:#173451;">{title}</div>
        <div style="font-size:12px;line-height:1.65;color:#64748b;margin-top:9px;">{desc}</div></div>""",unsafe_allow_html=True)

with st.expander("How the agent works", expanded=False):
    flow1, flow2, flow3 = st.columns(3)
    flow1.markdown("**01 · Discover & collect**\\n\\nDaily collection from public sources across selected competitors.")
    flow2.markdown("**02 · Synthesize with AI**\\n\\nCategorize and summarize by company, geography and business segment.")
    flow3.markdown("**03 · Deliver PL insights**\\n\\nPriority signals with publication dates and source links for validation.")

# ---------------- HEADER / STATUS ----------------
st.markdown('<div class="page-title">Competitive Intelligence</div><div class="page-sub">A focused view of competitor developments, emerging signals and AI-generated implications for Aranca Growth Advisory.</div>', unsafe_allow_html=True)
st.write("")

modified_at = datetime.fromtimestamp(DATA_FILE.stat().st_mtime).strftime("%d %b %Y · %I:%M %p")
st.markdown(f'<span class="eyebrow">INTELLIGENCE DATA · LAST FILE UPDATE {modified_at}</span>', unsafe_allow_html=True)
st.write("")

# ---------------- KPI ROW ----------------
total_records = len(df)
competitor_count = df["Company"].nunique()
geo_count = df["Country"].nunique()
high_count = int((df["Impact"].astype(str).str.lower() == "high").sum())
k1,k2,k3,k4 = st.columns(4)
for col,label,value in [
    (k1,"Developments monitored",f"{total_records:,}"),
    (k2,"Competitors covered",f"{competitor_count:,}"),
    (k3,"Geographies",f"{geo_count:,}"),
    (k4,"High-impact signals",f"{high_count:,}"),
]:
    col.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)

st.write("")

# ---------------- AI INTELLIGENCE ----------------
st.markdown('<div class="eyebrow">AI ANALYSIS</div>',unsafe_allow_html=True)
st.markdown("### Executive Intelligence Brief")
st.markdown('<div class="page-sub">Synthesize the latest monitored developments into a concise, decision-oriented brief.</div>',unsafe_allow_html=True)
brief_clicked = st.button("✦ Generate AI intelligence brief", type="primary") or st.session_state.pop("welcome_brief_trigger", False)

if brief_clicked:
    api_key = get_groq_api_key()
    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets.")
    elif filtered.empty:
        st.warning("No records match the selected filters.")
    else:
        try:
            client = Groq(api_key=api_key)
            analysis_df = filtered.sort_values("Published", ascending=False).head(15)
            records = analysis_df[["Company","Country","Title","Published","Category","Impact","Source","Link"]].to_dict("records")
            prompt = f"""You are a competitive intelligence analyst supporting Aranca Growth Advisory Practice Leaders.
Analyze only the supplied public-source records. Do not invent facts or infer financial/client details not present.
DATA: {records}
Write a concise professional brief with:
1. Executive takeaway (2-3 sentences)
2. Priority developments (up to 5; company, geography, date, what happened)
3. Why it matters to Aranca (clearly label interpretation)
4. Recommended follow-up actions for Practice Leaders (specific, practical)
5. Information gaps / validation needed
Cite the supplied source URL inline beside each factual development when available.
Distinguish facts from analysis and recommendations. If evidence is weak, say so."""
            with st.spinner("Analyzing monitored developments…"):
                response = generate_with_retry(client, prompt)
            st.markdown('<div class="ai-panel">',unsafe_allow_html=True)
            st.markdown(response or "The model returned an empty response.")
            st.markdown("</div>",unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"AI brief generation failed: {exc}")
            if "503" in str(exc) or "unavailable" in str(exc).lower():
                st.info("Groq may be temporarily unavailable. Please retry shortly; transient server errors were retried.")
else:
    st.markdown('<div class="ai-panel"><b>Ready to analyze</b><br><span style="color:#687b8e">Generate a brief to identify priority developments, strategic relevance and suggested follow-up actions. The analysis uses the currently selected filters.</span></div>',unsafe_allow_html=True)

st.write("")

# ---------------- MULTI-FILE REPORT GENERATOR ----------------
st.write("")
st.markdown('<div class="eyebrow">DOCUMENT ANALYSIS</div>', unsafe_allow_html=True)
st.markdown("### Upload files & generate a concise report")
st.markdown('<div class="page-sub">Add multiple documents or data files. IntelPulse will combine their contents into one short, structured report.</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf", "docx", "txt", "csv", "xlsx"],
    accept_multiple_files=True,
    key="report_files",
)

def extract_uploaded_text(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file).to_string(index=False)
    if name.endswith(".xlsx"):
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        return "\\n".join(f"Sheet: {sheet}\\n{frame.to_string(index=False)}" for sheet, frame in sheets.items())
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(uploaded_file)
        return "\\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(uploaded_file)
        return "\\n".join(paragraph.text for paragraph in doc.paragraphs)
    return ""

if uploaded_files:
    st.caption(f"{len(uploaded_files)} file(s) selected: " + " · ".join(f.name for f in uploaded_files))
    report_focus = st.text_input(
        "Report focus (optional)",
        placeholder="e.g., Key competitor moves, risks, opportunities, and recommended actions",
        key="report_focus",
    )
    if st.button("Generate concise report", type="primary", key="multi_file_report"):
        api_key = get_groq_api_key()
        if not api_key:
            st.error("GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets.")
        else:
            try:
                file_contents = []
                for file in uploaded_files:
                    extracted = extract_uploaded_text(file).strip()
                    if extracted:
                        file_contents.append(f"\\n--- FILE: {file.name} ---\\n{extracted[:18000]}")
                    else:
                        st.warning(f"Could not extract readable text from {file.name}.")
                if not file_contents:
                    st.warning("No readable text was found in the uploaded files.")
                else:
                    combined = "\\n".join(file_contents)[:65000]
                    focus = report_focus.strip() or "Summarize the most important findings and implications."
                    prompt = f"""You are a precise business research analyst. Use ONLY the uploaded file content below.
Create a concise report (maximum 400 words) with:
- Executive summary (2-3 sentences)
- 4-6 key findings (bullets; mention source filename where relevant)
- Important risks, gaps, or conflicting information
- 3 practical recommended actions
Do not invent facts. Clearly label any interpretation. If a detail is absent, say it is not provided.
Report focus: {focus}
UPLOADED CONTENT:
{combined}"""
                    client = Groq(api_key=api_key)
                    with st.spinner("Reading files and preparing your report…"):
                        response = generate_with_retry(client, prompt)
                    st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
                    st.markdown(response or "The model returned an empty response.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.download_button(
                        "Download report as TXT",
                        data=(response or "").encode("utf-8"),
                        file_name="intelPulse_concise_report.txt",
                        mime="text/plain",
                    )
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")
                if "503" in str(exc) or "unavailable" in str(exc).lower():
                    st.info("Groq may be temporarily unavailable. Please retry shortly; transient server errors were retried.")
else:
    st.info("Upload PDF, DOCX, TXT, CSV or XLSX files to create a consolidated report.")

# ---------------- PRIORITY SIGNALS ----------------
left, right = st.columns([1.35,1])
with left:
    st.markdown('<div class="eyebrow">PRIORITY DEVELOPMENTS</div>',unsafe_allow_html=True)
    st.markdown("### High-impact signals")
    high = filtered[filtered["Impact"].astype(str).str.lower()=="high"].sort_values("Published",ascending=False)
    if high.empty:
        st.info("No high-impact signals in the current filtered view.")
    else:
        for _, row in high.head(5).iterrows():
            title = html.escape(str(row["Title"]))
            company_name = html.escape(str(row["Company"]))
            geo = html.escape(str(row["Country"]))
            date = html.escape(str(row["Published"]))
            cat = html.escape(str(row["Category"]))
            st.markdown(f'<div class="signal-card"><span class="tag high">HIGH IMPACT</span><div class="signal-title" style="margin-top:8px">{title}</div><div class="meta">{company_name} · {geo} · {cat} · {date}</div></div>',unsafe_allow_html=True)
            if str(row["Link"]).strip():
                st.markdown(f"[Open source ↗]({row['Link']})")
with right:
    st.markdown('<div class="eyebrow">MONITORING COVERAGE</div>',unsafe_allow_html=True)
    st.markdown("### Current view")
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.metric("Records matching filters",len(filtered))
    st.caption(f"{filtered['Company'].nunique()} competitors · {filtered['Country'].nunique()} geographies")
    st.caption("Coverage reflects collected public-source records; it is not a measure of total market activity.")
    st.markdown("</div>",unsafe_allow_html=True)

st.write("")
# ---------------- INTELLIGENCE FEED ----------------
st.markdown('<div class="eyebrow">SOURCE-TRACEABLE RECORDS</div>',unsafe_allow_html=True)
st.markdown("### Competitive intelligence feed")
st.caption("Developments are listed with company, geography, category, impact, publication date and source URL.")
display = filtered[["Company","Country","Title","Published","Category","Impact","Source","Link"]].copy()
st.dataframe(display, use_container_width=True, hide_index=True, column_config={
    "Link": st.column_config.LinkColumn("Source link",display_text="Open source"),
    "Title": st.column_config.TextColumn("Development",width="large"),
    "Published": st.column_config.TextColumn("Publication date",width="medium"),
})
st.download_button("Download filtered intelligence (CSV)",display.to_csv(index=False).encode("utf-8"),"intelpulse_filtered.csv","text/csv")

st.divider()
st.caption("Aranca IntelPulse · Competitive Intelligence Prototype · Public-source records require source-level verification.")
