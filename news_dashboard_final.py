import streamlit as st
import nltk
import pandas as pd
from gnews import GNews
from newspaper import Article
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import logging
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from io import BytesIO
from datetime import datetime
# openpyxl is used by pandas for .to_excel, ensure it's installed: pip install openpyxl
import openpyxl 

# --- Page Configuration ---
st.set_page_config(
    page_title="Briefing Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Configuration Constants ---
PAGE_LOAD_TIMEOUT = 15
SELENIUM_WAIT_AFTER_LOAD = 7

# --- Caching Functions ---
@st.cache_resource(show_spinner=False)
def load_summarizer():
    """Loads the BART summarization model and tokenizer."""
    try:
        MODEL_NAME = "facebook/bart-large-cnn"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Fatal Error: Could not load summarizer model. Error: {e}")
        st.stop()

@st.cache_data(show_spinner=False)
def setup_nltk():
    """Downloads NLTK 'punkt' if necessary."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


@st.cache_data
def convert_df_to_excel(df):
    """Converts a Pandas DataFrame to Excel bytes for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Summaries')
    processed_data = output.getvalue()
    return processed_data

@st.cache_data
def convert_text_to_txt(text):
    """Converts a string to .txt bytes for download."""
    return text.encode('utf-8')

@st.cache_data
def convert_df_to_txt(df, topic):
    """Converts DataFrame to a formatted .txt file."""
    txt_output = f"Article Summaries for: {topic}\n"
    txt_output += "=" * (len(topic) + 24) + "\n\n"
    
    for index, row in df.iterrows():
        txt_output += f"Title: {row['title']}\n"
        txt_output += f"Source: {row['publisher']}\n"
        txt_output += f"Date: {row['date']}\n"
        txt_output += f"URL: {row['url']}\n\n"
        txt_output += "Summary:\n"
        txt_output += f"{row['summary']}\n"
        txt_output += "-" * 80 + "\n\n"
        
    return txt_output.encode('utf-8')

# --- Helper Functions ---
# CRITICAL FIX: Removed @st.cache_resource. This ensures a new driver
# is created for each search, fixing the "timeout on second run" issue.
def setup_selenium_driver():
    """Initializes and returns a headless Selenium WebDriver."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
    options.add_argument("--log-level=3")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument('--blink-settings=imagesEnabled=false')
    logging.getLogger('WDM').setLevel(logging.ERROR)
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        return driver
    except Exception as e:
        st.error(f"Failed to initialize Selenium WebDriver: {e}")
        st.stop()


# MODIFIED to accept length parameters
def generate_summary(text_to_summarize, tokenizer, model, device, min_len=150, max_len=600):
    """Uses the loaded model to summarize the provided text."""
    if not text_to_summarize:
        return "[Input text was empty]"
    try:
        inputs = tokenizer(
            [text_to_summarize], max_length=1024, return_tensors="pt", truncation=True
        )
        input_ids = inputs.input_ids.to(device)
        summary_ids = model.generate(
            input_ids,
            max_length=max_len,     # Use parameter
            min_length=min_len,     # Use parameter
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary if summary else "[Summary generation failed]"
    except Exception as sum_error:
        if "out of memory" in str(sum_error).lower():
            return f"[Summarization failed: Out of Memory. Article might be too long.]"
        return f"[Error during summarization: {sum_error}]"

def hierarchical_master_summary(summaries_list, tokenizer, model, device):
    """
    Build a master briefing without losing article information:
      1) join all article summaries,
      2) chunk the big text into safe-size pieces,
      3) summarize each chunk,
      4) summarize the chunk-summaries into the final master briefing.

    This prevents token truncation and avoids skipping articles.
    """
    if not summaries_list:
        return "No summaries available to build a master briefing."

    # Join all summaries (space-separated)
    joined = " ".join(summaries_list)

    # Heuristic chunking by character length (safer than raw token counts)
    CHUNK_SIZE = 3500  # adjust if you see truncation/OOM in your environment
    chunks = []
    start = 0
    while start < len(joined):
        end = min(start + CHUNK_SIZE, len(joined))
        # Prefer splitting at sentence boundary if possible
        if end < len(joined):
            idx = joined.rfind('.', start, end)
            if idx > start:
                end = idx + 1
        chunk_text = joined[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        start = end

    chunk_summaries = []
    for c in chunks:
        # Summarize each chunk. Keep chunk-output relatively short to avoid OOM.
        try:
            chunk_summary = generate_summary(c, tokenizer, model, device, min_len=120, max_len=400)
        except Exception as e:
            chunk_summary = f"[Chunk summarization failed: {e}]"
        # fallback: if summarizer returned an error-like placeholder, keep chunk slice
        if not chunk_summary or chunk_summary.startswith("["):
            chunk_summaries.append(c[:2000])
        else:
            chunk_summaries.append(chunk_summary)

    # Combine chunk summaries and produce the final master summary
    combined = " ".join(chunk_summaries)
    try:
        final_summary = generate_summary(combined, tokenizer, model, device, min_len=250, max_len=800)
    except Exception as e:
        final_summary = f"[Final summarization failed: {e}]"

    # If summarizer failed and returned placeholder, return concatenated chunk summaries as fallback
    if not final_summary or final_summary.startswith("["):
        return "\n\n".join(chunk_summaries)
    return final_summary


# --- UI Styling (CSS) ---
st.markdown("""
<style>
    /* Import Satoshi Font */
    @import url('https://fonts.cdnfonts.com/css/satoshi');

    /* App background & font */
    .stApp {
        background-color: #0d0e12;
        background-image: radial-gradient(at 0% 0%, hsla(253, 16%, 7%, 1) 0, transparent 50%),
                          radial-gradient(at 50% 0%, hsla(220, 39%, 11%, 1) 0, transparent 50%),
                          radial-gradient(at 100% 0%, hsla(253, 16%, 7%, 1) 0, transparent 50%);
        font-family: 'Satoshi', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    /* Header / Branding */
    .header-container {
        text-align: left;
        padding: 1rem 0;
    }
    .brand-title {
        font-size: 3.2rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.6px;
        margin: 0;
    }
    .brand-tagline {
        font-size: 1.05rem;
        font-weight: 500;
        color: #a0a0a0;
        margin-top: 4px;
    }

    /* ========== SEARCH ROW LAYOUT ========== */
    .search-row { display: flex; gap: 8px; align-items: stretch; width: 100%; }
    .search-row > div { display: flex; align-items: stretch; width: 100%; }
    :root { --search-control-h: 56px; }
    .search-row .stTextInput,
    .search-row .stNumberInput { display: flex !important; align-items: center !important; width: 100%; margin: 0 !important; }
    .search-row .stTextInput input,
    .search-row .stNumberInput input,
    .search-row .stSelectbox select {
        height: var(--search-control-h) !important;
        min-height: var(--search-control-h) !important;
        box-sizing: border-box !important;
        padding: 0 0.9rem !important;
        font-size: 1.02rem !important;
        line-height: 1 !important;
        border-radius: 10px !important;
        background-color: #0f1116 !important;
        color: #e6eef3 !important;
        border: 1px solid #2b2f36 !important;
    }
    .search-row .stTextInput input { border-top-right-radius: 0 !important; border-bottom-right-radius: 0 !important; }
    .search-row .stNumberInput input {
        border-radius: 0 !important;
        border-left: 0 !important;
        border-right: 0 !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        text-align: center !important;
        width: 100%;
        max-width: 160px;
    }
    .search-row .stButton { display: flex !important; align-items: center !important; margin: 0 !important; }
    .search-row .stButton button {
        height: var(--search-control-h) !important;
        min-height: var(--search-control-h) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 1rem !important;
        box-sizing: border-box !important;
        line-height: 1 !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        border-top-left-radius: 0 !important;
        border-bottom-left-radius: 0 !important;
        border-radius: 10px !important;
        background-color: #25f2a1 !important;
        color: #0d0e12 !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        cursor: pointer !important;
    }
    .search-row .stButton button:hover { filter: brightness(0.98) !important; transform: translateY(-1px); }
    .search-row .stButton button svg,
    .search-row .stButton button img { height: 20px !important; width: 20px !important; display: block; margin-right: 8px; }
    .search-hint { color: #9aa3ad; font-size: 0.95rem; margin-top: 6px; }

    /* ========== RESULT & GENERAL STYLES ========== */
    .ai-status {
        text-align: center; font-size: 1.1rem; color: #a0a0a0; margin: 1.5rem 0;
        display: flex; justify-content: center; align-items: center; gap: 10px;
    }
    .ai-status-complete { color: #25f2a1; }
    .spinner {
        width: 20px; height: 20px; border: 3px solid #333;
        border-top-color: #25f2a1; border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    
    .stExpander { background-color: #0f1116; border: 1px solid #23252b; border-radius: 10px; margin-bottom: 1rem; padding: 0.85rem; }
    .stExpander header { font-size: 1.2rem; font-weight: 600; color: #e0e0e0; }
    .stExpander header:hover { color: #25f2a1; }
    .stExpander header button { color: #25f2a1; }
    
    /* Updated result-source to use flex for side-by-side layout */
    .result-source {
        font-size: 0.95rem; font-weight: 600; color: #a0a0a0;
        border-bottom: 1px solid #23252b; padding-bottom: 8px; margin-bottom: 8px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .result-summary { font-size: 1.02rem; line-height: 1.6; color: #e6eef3; }

    .results-header-container {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 2px solid #23252b; padding-bottom: 10px; margin-top: 1.25rem;
    }
    .results-header-container h2 { color: #FFFFFF; margin: 0; font-size: 1.5rem; }
    .results-header-container .timer { font-size: 0.98rem; color: #9aa3ad; font-weight: 500; }
    
    /* Master Summary Box */
    .master-summary-box {
        background-color: #0f1116;
        border: 1px solid #25f2a1;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .master-summary-box h2 {
        color: #25f2a1;
        font-size: 1.5rem;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    .master-summary-box p {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e6eef3;
    }

    /* Download button styling */
    .stDownloadButton button {
        background-color: transparent !important; color: #25f2a1 !important;
        border: 1px solid #25f2a1 !important; border-radius: 8px !important;
        padding: 8px 12px !important; font-weight: 600 !important;
    }
    .stDownloadButton button:hover {
        background-color: #25f2a1 !important; color: #0d0e12 !important;
    }
    
    /* NEW: Popover button styling */
    .stButton > button[data-baseweb="popover-anchor"] {
        background-color: transparent !important;
        color: #25f2a1 !important;
        border: 1px solid #25f2a1 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
    }
    .stButton > button[data-baseweb="popover-anchor"]:hover {
        background-color: #25f2a1 !important;
        color: #0d0e12 !important;
    }
    /* Style the buttons *inside* the popover */
    div[data-baseweb="popover"] .stDownloadButton button {
        background-color: #0f1116 !important;
        color: #e6eef3 !important;
        border: 1px solid #2b2f36 !important;
        width: 100%;
    }
    div[data-baseweb="popover"] .stDownloadButton button:hover {
        background-color: #2b2f36 !important;
        color: #25f2a1 !important;
        border: 1px solid #25f2a1 !important;
    }


    /* Responsive tweaks */
    @media (max-width: 900px) {
        .search-row { flex-direction: column; gap: 10px; }
        .search-row .stTextInput input { border-radius: 10px !important; border-right: 1px solid #2b2f36 !important; }
        .search-row .stSelectbox div[data-baseweb="select"] { border-radius: 10px !important; border-left: 1px solid #2b2f36 !important; border-right: 1px solid #2b2f36 !important; }
        .search-row .stButton button { border-radius: 10px !important; }
    }

</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header-container">
    <div class="brand-title">Briefing Engine</div>
    <div class="brand-tagline">Your AI-Powered News Briefing 🤖</div>
</div>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'overall_summary' not in st.session_state:
    st.session_state.overall_summary = None
if 'elapsed_time_str' not in st.session_state:
    st.session_state.elapsed_time_str = ""
if 'search_topic' not in st.session_state:
    st.session_state.search_topic = ""


# --- Load models and setup NLTK ---
tokenizer, model, device = load_summarizer()
setup_nltk()

# --- User Input ---
st.markdown('<div class="search-row">', unsafe_allow_html=True)
with st.form(key='search_form'):
    cols = st.columns([8, 2, 1])
    with cols[0]:
        search_query = st.text_input("Enter news topic", placeholder="e.g., Latest Developments in AI", label_visibility="visible", key='search_query_input')
    with cols[1]:
        target_article_count = st.number_input("Articles", min_value=1, max_value=25, value=10, step=5, format='%d', help='Choose how many articles to fetch (recommended 5-15)', key='target_article_count')
    with cols[2]:
        submitted = st.form_submit_button(label='🔎 Search', disabled=st.session_state.processing, use_container_width=True)

    st.markdown('<div class="search-hint">Tip: Press Enter or click <strong>Search</strong>. Use the number input to pick how many articles you want — smaller numbers are faster.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Process form submission
if submitted:
    if not search_query or not search_query.strip():
        st.warning("Please enter a search topic.")
    else:
        # Begin processing, clear all previous results
        st.session_state.processing = True
        st.session_state.target_count = int(target_article_count)
        st.session_state.search_topic = search_query 
        st.session_state.results_df = None
        st.session_state.overall_summary = None # Reset master summary
        st.session_state.elapsed_time_str = ""
        # Force a rerun so the processing block can pick up the state
        st.rerun()

# --- AI Status Updater & Processing Loop ---
status_container = st.empty()

if st.session_state.processing:
    start_time = time.time()
    driver = None
    results_list = []
    successful_summary_count = 0
    processed_article_count = 0
    all_results = []
    target_count = st.session_state.target_count
    current_search_topic = st.session_state.search_topic

    try:
        status_container.write('<div class="ai-status"><div class="spinner"></div> Finding relevant articles...</div>', unsafe_allow_html=True)
        google_news = GNews()
        try:
            all_results = google_news.get_news(current_search_topic)
        except Exception as gnews_error:
            status_container.write('<div class="ai-status ai-status-complete">❌ Error fetching news. Please try again.</div>', unsafe_allow_html=True)
            st.session_state.processing = False
            st.stop()

        if not all_results:
            status_container.write('<div class="ai-status ai-status-complete">⚠️ No articles found for this topic.</div>', unsafe_allow_html=True)
            st.session_state.processing = False
            st.stop()

        status_container.write('<div class="ai-status ai-status-complete">✅ Articles found.</div>', unsafe_allow_html=True)
        time.sleep(1)

        status_container.write('<div class="ai-status"><div class="spinner"></div> Initializing analysis engine...</div>', unsafe_allow_html=True)
        # This now creates a NEW driver every time, thanks to removal of @st.cache_resource
        driver = setup_selenium_driver()
        status_container.write('<div class="ai-status ai-status-complete">✅ Engine ready.</div>', unsafe_allow_html=True)
        time.sleep(1)

        for i, article_data in enumerate(all_results):
            if successful_summary_count >= target_count:
                break # Stop once we have enough *successful* articles

            processed_article_count += 1
            url = article_data['url']
            original_title = article_data['title']
            publisher = article_data['publisher']['title']
            
            # Get and format the date
            publish_date_str = article_data['published date']
            try:
                dt = datetime.strptime(publish_date_str, '%a, %d %b %Y %H:%M:%S %Z')
                formatted_date = dt.strftime('%B %d, %Y')
            except (ValueError, TypeError):
                formatted_date = "Date not available"

            summary_text = None
            parsed_title = original_title

            try:
                # UI Update: Stays on "Article 1" until it finds a *successful* article 1
                status_container.write(f'<div class="ai-status"><div class="spinner"></div> Reading article {successful_summary_count + 1} of {target_count}...</div>', unsafe_allow_html=True)
                driver.get(url)
                time.sleep(SELENIUM_WAIT_AFTER_LOAD)
                html_content = driver.page_source

                if not html_content or len(html_content) < 200:
                    continue # Skip if page is empty

                article = Article(url)
                article.download(input_html=html_content)
                article.parse()
                article_text = article.text
                if article.title: parsed_title = article.title

                status_container.write(f'<div class="ai-status ai-status-complete">✅ Reading article {successful_summary_count + 1}... Complete.</div>', unsafe_allow_html=True)
                time.sleep(0.5)

                # MODIFICATION: Only summarize if we have article text. Skip if not.
                if article_text and len(article_text) > 100: # Added length check
                    status_container.write(f'<div class="ai-status"><div class="spinner"></div> Summarizing article {successful_summary_count + 1}...</div>', unsafe_allow_html=True)
                    # Use default (shorter) summary lengths for individual articles
                    summary_text = generate_summary(article_text, tokenizer, model, device)
                    
                    status_container.write(f'<div class="ai-status ai-status-complete">✅ Summarization {successful_summary_count + 1}... Complete.</div>', unsafe_allow_html=True)
                    time.sleep(0.5)
                else:
                    # No article text found, so we will skip this loop iteration
                    continue 

            except Exception as e:
                print(f"Error processing article {url}: {e}")
                continue # Skip to next article on any error

            # Check for valid summary before appending
            if summary_text and not summary_text.startswith("["):
                results_list.append({
                    'title': parsed_title,
                    'summary': summary_text,
                    'publisher': publisher,
                    'url': url,
                    'date': formatted_date # Add the formatted date
                })
                successful_summary_count += 1 # Increment *only* on success

        st.session_state.results_df = pd.DataFrame(results_list)
        
        # --- NEW: Generate Master Summary ---
        if results_list:
            status_container.write(f'<div class="ai-status"><div class="spinner"></div> Creating master briefing...</div>', unsafe_allow_html=True)
            all_summaries_text = " ".join([d['summary'] for d in results_list])
            
            # Use LARGER length parameters for the master summary
            master_summary = generate_summary(all_summaries_text, tokenizer, model, device, min_len=300, max_len=800)
            
            if master_summary and not master_summary.startswith("["):
                st.session_state.overall_summary = master_summary
            else:
                st.session_state.overall_summary = "Could not generate a master summary for this topic."
            status_container.write(f'<div class="ai-status ai-status-complete">✅ Briefing complete!</div>', unsafe_allow_html=True)
            time.sleep(1)


    finally:
        if driver:
            driver.quit() # CRITICAL: Ensures browser closes every time
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        st.session_state.elapsed_time_str = f"Time taken: {minutes}m {seconds}s"
        st.session_state.processing = False
        status_container.empty()
        st.rerun()

# --- Display Results ---
# NEW STRUCTURE: This 'else' block ensures results *only* display
# when processing is *not* active.
else:
    # 1. Display Master Summary
    if st.session_state.overall_summary:
        st.markdown(f"""
        <div class="master-summary-box">
            <h2>Master Briefing for "{st.session_state.search_topic}"</h2>
            <p>{st.session_state.overall_summary}</p>
        </div>
        """, unsafe_allow_html=True)
        
        txt_data = convert_text_to_txt(st.session_state.overall_summary)
        st.download_button(
            label="Download Briefing (.txt) 💾",
            data=txt_data,
            file_name=f"{st.session_state.search_topic}_briefing.txt",
            mime='text/plain',
            key='download_master_summary'
        )


    # 2. Display Individual Article Summaries
    if st.session_state.results_df is not None:
        
        if st.session_state.results_df.empty:
            st.warning("No articles could be successfully summarized for this topic.")
        else:
            # Use st.expander for the collapsible individual summaries section
            with st.expander("Show Individual Article Summaries", expanded=True):
                header_cols = st.columns([3, 1])
                with header_cols[0]:
                    st.markdown(f"""
                    <div class="results-header-container">
                        <h2>Individual Articles</h2>
                        <div class="timer">{st.session_state.elapsed_time_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with header_cols[1]:
                    st.markdown(f'<div style="height: 38px;"></div>', unsafe_allow_html=True)
                    
                    # Prepare data for new download buttons
                    excel_data = convert_df_to_excel(st.session_state.results_df)
                    txt_data = convert_df_to_txt(st.session_state.results_df, st.session_state.search_topic)
                    
                    # NEW: Use st.popover for clean download options
                    with st.popover("💾 Download...", use_container_width=True):
                        st.download_button(
                            label="Download as Excel 💾",
                            data=excel_data,
                            file_name=f"briefing_export.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key='download_excel',
                            use_container_width=True
                        )
                        st.download_button(
                            label="Download as Text 📄",
                            data=txt_data,
                            file_name=f"briefing_export.txt",
                            mime='text/plain',
                            key='download_text',
                            use_container_width=True
                        )

                st.markdown("<br>", unsafe_allow_html=True)
                for index, row in st.session_state.results_df.iterrows():
                    # Use an inner expander for each article
                    with st.expander(f"{row['title']}"):
                        # Updated markdown to show source and date side-by-side
                        st.markdown(f"""
                            <div class='result-source'>
                                <span><strong>Source:</strong> {row['publisher']}</span>
                                <span>{row['date']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<div class='result-summary'>{row['summary']}</div>", unsafe_allow_html=True)
                        st.link_button("Read Full Article", row['url'], use_container_width=True)