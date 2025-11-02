import streamlit as st
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
from io import BytesIO
from datetime import datetime
import requests
import re
import json
import openpyxl

# --- NLTK is no longer needed ---

# --- Page Configuration ---
st.set_page_config(
    page_title="News Briefing Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Configuration Constants ---
PAGE_LOAD_TIMEOUT = 20
SELENIUM_WAIT_AFTER_LOAD = 12

# --- Caching Functions ---
# --- REMOVED setup_nltk ---

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
    
    df_valid = df[df['summary'].notna()]
    
    for index, row in df_valid.iterrows():
        txt_output += f"Title: {row['title']}\n"
        txt_output += f"Source: {row['publisher']}\n"
        txt_output += f"Date: {row['date']}\n"
        txt_output += f"URL: {row['url']}\n\n"
        txt_output += "Summary:\n"
        txt_output += f"{row['summary']}\n"
        
        if 'article_outlook' in row and pd.notna(row['article_outlook']):
            txt_output += f"\nAI Outlook: {row['article_outlook']}\n"
            txt_output += f"AI Reasoning: {row['article_reasoning']}\n"
            
        txt_output += "-" * 80 + "\n\n"
        
    return txt_output.encode('utf-8')

# --- Helper Functions ---

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
        print(f"Failed to initialize Selenium WebDriver: {e}")
        st.stop()

# --- DEEPSEEK API FUNCTIONS ---

# Make sure to set this in your Streamlit Secrets!
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
DEEPSEEK_MODEL = "deepseek/deepseek-chat" 

def create_deepseek_prompt(articles_data):
    """
    Formats the list of article titles and FULL TEXTS into a prompt
    for the Deepseek model to summarize AND analyze.
    
    articles_data is a list of dicts: [{'title': t, 'article_text': a}, ...]
    """
    
    input_block = ""
    for i, article in enumerate(articles_data):
        # Truncate article text to avoid excessively long prompts
        # This is a safety measure for the API context window.
        truncated_text = article['article_text'][:15000] 
        
        input_block += f"[ARTICLE_{i+1}]\n"
        input_block += f"Title: {article['title']}\n"
        # --- MODIFIED: Back to using 'Full_Text' ---
        input_block += f"Full_Text: {truncated_text}\n"
        input_block += f"[END_ARTICLE_{i+1}]\n"
        input_block += "---\n"

    prompt = f"""
[START_INPUT_ARTICLES]
{input_block}
[END_INPUT_ARTICLES]

You are an expert financial and news analyst. Based *only* on the 'Full_Text' provided for each article, perform a comprehensive analysis.

You MUST strictly follow this multi-step process and output format.
Do not add any conversational text before or after the formatted response.

**--- OUTPUT FORMAT ---**

[INDIVIDUAL_ANALYSIS]
Title: (Repeat the title of the first article)
Summary: (Generate a 150-200 word summary of this article's 'Full_Text'.)
Outlook: (State *only*: Positive, Negative, or Neutral)
Reasoning: (Provide a 1-sentence reasoning for this specific article's outlook.)
---
(Repeat this block for *every* article provided in the input, separated by '---')
[END_INDIVIDUAL_ANALYSIS]

[MASTER_BRIEFING]
(Generate a comprehensive, single-paragraph master summary synthesizing all the *new summaries* you just wrote.)
[END_MASTER_BRIEFING]

[OVERALL_SENTIMENT]
(State *only* one of the following: Positive, Negative, or Neutral, based on your new summaries)
[END_OVERALL_SENTIMENT]

[OVERALL_REASONING]
(Provide a 1-2 sentence explanation for the overall sentiment, based on the master briefing.)
[END_OVERALL_REASONING]

[KEY_POSITIVES]
- (List a key positive point from your new summaries. If none, write "No positives found.")
- (List another key positive point if applicable.)
[END_KEY_POSITIVES]

[KEY_NEGATIVES]
- (List a key negative point from your new summaries. If none, write "No negatives found.")
- (List another key negative point if applicable.)
[END_KEY_NEGATIVES]
"""
    return prompt

def call_deepseek_api(prompt_text):
    """Calls the OpenRouter API for the Deepseek model."""
    
    if not OPENROUTER_API_KEY:
        print("FATAL: OPENROUTER_API_KEY is not set. Please add it to Streamlit Secrets.")
        return None
        
    print("Calling Deepseek API...") # For server log
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": DEEPSEEK_MODEL, 
                "messages": [
                    {"role": "user", "content": prompt_text}
                ]
            })
        )
        
        response.raise_for_status() 
        data = response.json()
        content = data['choices'][0]['message']['content']
        print("API call successful.") # For server log
        return content
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")
    except Exception as e:
        print(f"An error occurred calling OpenRouter: {e}")
        
    return None

def parse_deepseek_response(response_text):
    """Parse the structured Deepseek response into a dict (best-effort)."""
    if not response_text:
        return None
    parsed = {}
    try:
        parsed['master_briefing'] = re.search(r'\[MASTER_BRIEFING\](.*?)\[END_MASTER_BRIEFING\]', response_text, re.DOTALL).group(1).strip()
    except:
        parsed['master_briefing'] = None
    try:
        parsed['overall_sentiment'] = re.search(r'\[OVERALL_SENTIMENT\](.*?)\[END_OVERALL_SENTIMENT\]', response_text, re.DOTALL).group(1).strip()
    except:
        parsed['overall_sentiment'] = None
    try:
        parsed['overall_reasoning'] = re.search(r'\[OVERALL_REASONING\](.*?)\[END_OVERALL_REASONING\]', response_text, re.DOTALL).group(1).strip()
    except:
        parsed['overall_reasoning'] = None

    # Extract positives & negatives
    try:
        positives_raw = re.search(r'\[KEY_POSITIVES\](.*?)\[END_KEY_POSITIVES\]', response_text, re.DOTALL).group(1)
        parsed['key_positives'] = [p.strip() for p in re.findall(r'^\s*-\s*(.+)$', positives_raw, re.MULTILINE)]
    except:
        parsed['key_positives'] = []

    try:
        negatives_raw = re.search(r'\[KEY_NEGATIVES\](.*?)\[END_KEY_NEGATIVES\]', response_text, re.DOTALL).group(1)
        parsed['key_negatives'] = [p.strip() for p in re.findall(r'^\s*-\s*(.+)$', negatives_raw, re.MULTILINE)]
    except:
        parsed['key_negatives'] = []

    # Parse individual analyses
    parsed['individual_analyses'] = []
    try:
        ind_raw = re.search(r'\[INDIVIDUAL_ANALYSIS\](.*?)\[END_INDIVIDUAL_ANALYSIS\]', response_text, re.DOTALL).group(1)
        blocks = [b.strip() for b in ind_raw.split('---') if b.strip()]
        for b in blocks:
            title = re.search(r'Title:\s*(.*)', b)
            summary = re.search(r'Summary:\s*(.*?)\s*Outlook:', b, re.DOTALL)
            outlook = re.search(r'Outlook:\s*(.*)', b)
            reasoning = re.search(r'Reasoning:\s*(.*)', b, re.DOTALL)
            if title:
                parsed['individual_analyses'].append({
                    'title': title.group(1).strip(),
                    'summary': summary.group(1).strip() if summary else None,
                    'outlook': outlook.group(1).strip() if outlook else None,
                    'reasoning': reasoning.group(1).strip() if reasoning else None
                })
        return parsed
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to parse API response. Error: {e}")
        print(f"--- Raw API Response ---\n{response_text}")
        return None

# --- END API FUNCTIONS ---


# --- UI Styling (CSS) ---
# (CSS is unchanged, so it is hidden here for brevity)
st.markdown("""
<style>
    /* ... all your original CSS ... */
    
    /* App background & font */
    .stApp {
        background-color: #0d0e12;
        background-image: radial-gradient(at 0% 0%, hsla(253, 16%, 7%, 1) 0, transparent 50%),
                          radial-gradient(at 50% 0%, hsla(220, 39%, 11%, 1) 0, transparent 50%),
                          radial-gradient(at 100% 0%, hsla(253, 16%, 7%, 1) 0, transparent 50%);
        font-family: 'Satoshi', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    .header-container { text-align: left; padding: 1rem 0; }
    .brand-title { font-size: 3.2rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.6px; margin: 0; }
    .brand-tagline { font-size: 1.05rem; font-weight: 500; color: #a0a0a0; margin-top: 4px; }
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
    .stDownloadButton button {
        background-color: transparent !important; color: #25f2a1 !important;
        border: 1px solid #25f2a1 !important; border-radius: 8px !important;
        padding: 8px 12px !important; font-weight: 600 !important;
    }
    .stDownloadButton button:hover {
        background-color: #25f2a1 !important; color: #0d0e12 !important;
    }
    .ai-outlook-block {
        background-color: #1a1c22;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .ai-outlook-block strong {
        color: #25f2a1; /* Neon green */
    }
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
if 'elapsed_time_str' not in st.session_state:
    st.session_state.elapsed_time_str = ""
if 'search_topic' not in st.session_state:
    st.session_state.search_topic = ""
if 'deepseek_result' not in st.session_state:
    st.session_state.deepseek_result = None

# --- REMOVED setup_nltk() call ---

# --- User Input ---
st.markdown('<div class="search-row">', unsafe_allow_html=True)
with st.form(key='search_form'):
    cols = st.columns([8, 2, 1])
    with cols[0]:
        search_query = st.text_input("Enter news topic", placeholder="e.g., Latest Developments in AI", label_visibility="visible", key='search_query_input')
    with cols[1]:
        # Back to a 10-article max, as full text is large
        target_article_count = st.number_input("Articles", min_value=2, max_value=10, value=4, step=2, format='%d', help='Choose how many articles to fetch (4-6 recommended)', key='target_article_count')
    with cols[2]:
        submitted = st.form_submit_button(label='🔎 Search', disabled=st.session_state.processing, use_container_width=True)

    st.markdown('<div class="search-hint">Tip: Press Enter or click <strong>Search</strong>. Use the number input to pick how many articles you want — smaller numbers are faster.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Process form submission
if submitted:
    if not search_query or not search_query.strip():
        st.warning("Please enter a search topic.")
    else:
        st.session_state.processing = True
        st.session_state.target_count = int(target_article_count)
        st.session_state.search_topic = search_query 
        st.session_state.results_df = None
        st.session_state.deepseek_result = None 
        st.session_state.elapsed_time_str = ""
        st.rerun()

# --- AI Status Updater & Processing Loop ---
status_container = st.empty()

if st.session_state.processing:
    start_time = time.time()
    driver = None
    results_list = []
    successful_article_count = 0
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

        all_results.sort(key=lambda x: datetime.strptime(x['published date'], '%a, %d %b %Y %H:%M:%S %Z'), reverse=True)

        status_container.write('<div class="ai-status ai-status-complete">✅ Articles Found.</div>', unsafe_allow_html=True)
        time.sleep(1)

        status_container.write('<div class="ai-status"><div class="spinner"></div> Booting Up Analysis Engine...</div>', unsafe_allow_html=True)
        driver = setup_selenium_driver()
        status_container.write('<div class="ai-status ai-status-complete">✅ Engine Ready.</div>', unsafe_allow_html=True)
        time.sleep(1)

        for i, article_data in enumerate(all_results):
            if successful_article_count >= target_count:
                break 

            processed_article_count += 1
            url = article_data['url']
            original_title = article_data['title']
            publisher = article_data['publisher']['title']
            
            publish_date_str = article_data['published date']
            try:
                dt = datetime.strptime(publish_date_str, '%a, %d %b %Y %H:%M:%S %Z')
                formatted_date = dt.strftime('%B %d, %Y')
            except (ValueError, TypeError):
                formatted_date = "Date not available"

            parsed_title = original_title
            article_text = "" 

            try:
                status_container.write(f'<div class="ai-status"><div class="spinner"></div> Reading article {successful_article_count + 1} of {target_count}...</div>', unsafe_allow_html=True)
                driver.get(url)
                time.sleep(SELENIUM_WAIT_AFTER_LOAD)
                html_content = driver.page_source

                if not html_content or len(html_content) < 200:
                    continue 

                article = Article(url)
                article.download(input_html=html_content)
                article.parse() # This finds the main article text
                
                article_text = article.text # This is the "clean" text
                if article.title: parsed_title = article.title

                status_container.write(f'<div class="ai-status ai-status-complete">✅ Reading Article {successful_article_count + 1}... Complete.</div>', unsafe_allow_html=True)
                time.sleep(0.5)
                
                # --- THIS IS YOUR REQUESTED CHECK ---
                # Skip article if it's too short (likely a
                # partial scrape, error page, or just a blurb)
                if not article_text or len(article_text) < 1000:
                    print(f"Skipping {url} (too short: {len(article_text)} chars)")
                    continue 
                # --- END OF CHECK ---

            except Exception as e:
                print(f"Error processing article {url}: {e}")
                continue 

            results_list.append({
                'title': parsed_title,
                'publisher': publisher,
                'url': url,
                'date': formatted_date,
                'article_text': article_text # Store the full "clean" text for the API
            })
            successful_article_count += 1 

        st.session_state.results_df = pd.DataFrame(results_list)
        
        st.session_state.deepseek_result = None 
        
        if results_list:
            status_container.write(f'<div class="ai-status"><div class="spinner"></div> Running AI Analysis...</div>', unsafe_allow_html=True)
            
            # Send the 'article_text' (which is the full, parsed text)
            api_input_data = [{'title': r['title'], 'article_text': r['article_text']} for r in results_list]
            
            api_prompt = create_deepseek_prompt(api_input_data)
            
            api_response_text = call_deepseek_api(api_prompt)
            
            if api_response_text:
                st.session_state.deepseek_result = parse_deepseek_response(api_response_text)
            
            if st.session_state.deepseek_result:
                status_container.write(f'<div class="ai-status ai-status-complete">✅ AI Analysis Complete.</div>', unsafe_allow_html=True)
                
                try:
                    analysis_map = {
                        item['title']: {
                            'summary': item['summary'], 
                            'article_outlook': item['outlook'], 
                            'article_reasoning': item['reasoning']
                        }
                        for item in st.session_state.deepseek_result.get('individual_analyses', [])
                    }
                    
                    df_analysis = st.session_state.results_df['title'].map(analysis_map).apply(pd.Series)
                    st.session_state.results_df = pd.concat([st.session_state.results_df, df_analysis], axis=1)
                
                except Exception as e:
                    st.warning(f"Could not merge individual analyses: {e}")
            else:
                status_container.write(f'<div class="ai-status ai-status-complete">❌ AI Analysis Failed.</div>', unsafe_allow_html=True)
                st.error("Failed to get or parse the response from the Deepseek API.")

            status_container.write(f'<div class="ai-status ai-status-complete">✅ Briefing Complete!</div>', unsafe_allow_html=True)
            time.sleep(1)

    finally:
        if driver:
            driver.quit() 
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        st.session_state.elapsed_time_str = f"Time taken: {minutes}m {seconds}s"
        st.session_state.processing = False
        status_container.empty()
        st.rerun()

# --- Display Results ---
else:
    # 1. Display Master Summary (from API)
    if st.session_state.deepseek_result:
        with st.expander("Master Briefing", expanded=True):
            briefing_text = st.session_state.deepseek_result.get('master_briefing', 'Master briefing could not be generated.')
            st.markdown(f"<p>{briefing_text}</p>", unsafe_allow_html=True)
            
            txt_data = convert_text_to_txt(briefing_text)
            st.download_button(
                label="Download Briefing",
                data=txt_data,
                file_name=f"{st.session_state.search_topic}_briefing.txt",
                mime='text/plain',
                key='download_master_summary'
            )
    
    # 2. Display Sentiment Overview (from API)
    if st.session_state.deepseek_result:
        with st.expander("Sentiment Overview", expanded=True):
            summary = st.session_state.deepseek_result
            st.markdown(f"**Overall:** {summary.get('overall_sentiment', 'N/A')}")
            st.markdown(f"**Reasoning:** *{summary.get('overall_reasoning', 'N/A')}*")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Key Positives")
                positives = summary.get('key_positives', ["No positives found."])
                if not positives: positives = ["No positives found."]
                for p in positives:
                    st.markdown(f"- {p}")
            
            with col2:
                st.markdown("#### Key Negatives")
                negatives = summary.get('key_negatives', ["No negatives found."])
                if not negatives: negatives = ["No negatives found."]
                for n in negatives:
                    st.markdown(f"- {n}")
            
    # 3. Display Individual Article Summaries
    if st.session_state.results_df is not None:
        
        # --- MODIFIED: This check is important now
        if 'summary' not in st.session_state.results_df.columns or st.session_state.results_df['summary'].isna().all():
             if not st.session_state.results_df.empty:
                st.warning("AI request timed out, please try again shortly.")
             else:
                st.warning("No in-depth content found for this search.")
        
        else:
            with st.expander("Individual Article Summaries", expanded=True):
                col1, col2, col3 = st.columns([8, 2, 2])
                with col1:
                    st.markdown(f'<div class="timer" style="text-align: left; padding-top: 8px;"><strong>{st.session_state.elapsed_time_str}</strong></div>', unsafe_allow_html=True)
                
                excel_data = convert_df_to_excel(st.session_state.results_df)
                txt_data = convert_df_to_txt(st.session_state.results_df, st.session_state.search_topic)
                
                with col2:
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name=f"briefing_export.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key='download_excel',
                        use_container_width=True
                    )
                with col3:
                    st.download_button(
                        label="Download Text",
                        data=txt_data,
                        file_name=f"briefing_export.txt",
                        mime='text/plain',
                        key='download_text',
                        use_container_width=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                
                for index, row in st.session_state.results_df.iterrows():
                    
                    if 'summary' not in row or pd.isna(row['summary']):
                        with st.expander(f"{row['title']} (Summary generation failed)"):
                            st.warning("This article couldn’t be summarized — try viewing the full version.")
                            st.link_button("Read Full Article", row['url'], use_container_width=True)
                        continue 

                    with st.expander(f"{row['title']}"):
                        st.markdown(f"""
                            <div class='result-source'>
                                <span><strong>Source:</strong> {row['publisher']}</span>
                                <span>{row['date']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"<div class='result-summary'>{row['summary']}</div>", unsafe_allow_html=True)
                        
                        if 'article_outlook' in row and pd.notna(row['article_outlook']):
                            st.markdown(f"""
                            <div class='ai-outlook-block'>
                                <strong>Sentiment:</strong> <b>{row['article_outlook']} </b><br>
                                <strong>Reasoning:</strong> {row['article_reasoning']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='ai-outlook-block'>
                                Failure</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.link_button("Read Full Article", row['url'], use_container_width=True)
