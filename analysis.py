import os

# DISABLING TOKENIZERS PARALLELISM TO PREVENT DEADLOCKS IN CLOUD RUN
# This is critical for avoiding hangs when using transformers in a threaded/multiprocess environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MEMORY OPTIMIZATION: Force CPU-only mode to reduce memory usage
# Comment out these lines if you have a GPU and want to use it
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import warnings
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
import pandas as pd
from google.cloud import bigquery
import sys
import time
import re
import logging
import torch

# MEMORY OPTIMIZATION: Limit CPU threads for lower memory footprint
torch.set_num_threads(2)

# Suppress tokenizer regex pattern warnings (known issue with DeBERTa tokenizers)
warnings.filterwarnings('ignore', message='.*incorrect regex pattern.*', category=FutureWarning)
from http.server import HTTPServer, BaseHTTPRequestHandler
from generate_topics import generate_topics_json
from analyzer import IssueAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class AnalysisServerHandler(BaseHTTPRequestHandler):
    """Server to handle health checks and trigger the analysis pipeline."""
    def do_GET(self):
        if self.path == '/':
            # Health check endpoint
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path.startswith('/run'):
            # Trigger analysis endpoint
            logger.info("Manual trigger received via /run")
            try:
                # Parse query parameters
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                config = {
                    'companies': params.get('companies', [None])[0],
                    'start_date': params.get('start_date', [None])[0],
                    'end_date': params.get('end_date', [None])[0],
                    'limit': params.get('limit', [None])[0],
                    'mode': params.get('mode', ['test'])[0]
                }

                logger.info(f"Starting pipeline with config: {config}")
                process_pipeline(config)

                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Pipeline execution complete. Check logs for details.")
            except Exception as e:
                logger.error(f"Error executing pipeline: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests for pipeline execution with JSON payload."""
        if self.path == '/run':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)

                if body:
                    config = json.loads(body.decode('utf-8'))
                else:
                    config = {}

                logger.info(f"POST trigger received with config: {config}")

                # Check if client wants CSV response
                return_csv = config.get('return_csv', False)

                if return_csv:
                    # Execute pipeline and return CSV data
                    csv_data = process_pipeline(config, return_data=True)

                    self.send_response(200)
                    self.send_header("Content-type", "text/csv")
                    self.send_header("Content-Disposition", "attachment; filename=analysis_results.csv")
                    self.end_headers()
                    self.wfile.write(csv_data.encode('utf-8'))
                else:
                    # Standard execution - write to BigQuery
                    process_pipeline(config)

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    response = {"status": "success", "message": "Pipeline execution complete"}
                    self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                logger.error(f"Error executing pipeline: {e}", exc_info=True)
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Redirect http.server logs to our logger
        logger.info("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format%args))

def run_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), AnalysisServerHandler)
    logger.info(f"Analysis server listening on port {port}...")
    server.serve_forever()

"""
analysis.py

Production pipeline for Cloud Run. Processes earnings call transcripts from BigQuery,
enriches them with classification and sentiment data, and writes back to BigQuery.

Features:
- Fragment Rejoining
- Q&A Session Clustering
- Role and Interaction Type Classification
- Aspect-Based Sentiment Analysis
"""

# =================================================================================================
# CONFIGURATION & SETUP
# =================================================================================================

# Always regenerate topics.json in production to ensure sync with CSV
generate_topics_json()

current_dir = os.getcwd() 
TOPICS_FILE = os.path.join(current_dir, 'topics.json')

# Threshold for vector similarity (0 to 1). 
SIMILARITY_THRESHOLD = 0.7

# Local Model Paths
INTERACTION_MODEL_PATH = os.path.join(current_dir, "models", "eng_type_class_v1")
ROLE_MODEL_PATH = os.path.join(current_dir, "models", "role_class_v1")
EMBEDDING_MODEL_PATH = os.path.join(current_dir, "models", "all-MiniLM-L6-v2")
# SENTIMENT_MODEL_PATH = os.path.join(current_dir, "models", "deberta-v3-base-absa-v1.1")  # COMMENTED OUT - Using VADER instead

# Initialize the modular analyzer for topic detection
# This ensures local CLI and Cloud Run use the exact same logic
# We pass the local EMBEDDING_MODEL_PATH for Cloud Run optimization
issue_analyzer = IssueAnalyzer(
    similarity_threshold=SIMILARITY_THRESHOLD,
    embedding_model=EMBEDDING_MODEL_PATH
)

# Human-Readable Label Mappings
INTERACTION_ID_MAP = {
    "LABEL_0": "Admin",
    "LABEL_1": "Answer",
    "LABEL_2": "Question"
}

ROLE_ID_MAP = {
    "LABEL_0": "Admin",
    "LABEL_1": "Analyst",
    "LABEL_2": "Executive",
    "LABEL_3": "Operator"
}

# BigQuery Configuration
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 500))
# Default to production-like behavior unless explicitly told otherwise in env
PRODUCTION_TESTING = os.environ.get("PRODUCTION_MODE", "false").lower() != "true"

BQ_PROJECT_ID = "sri-benchmarking-databases"
BQ_DATASET = "pressure_monitoring"
BQ_SOURCE_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_content"
BQ_METADATA_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_metadata"
BQ_DEST_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_enriched"
BQ_CORP_REF_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.corporation_reference"
TICKERS_FILE = os.path.join(current_dir, 'tickers.csv')

if PRODUCTION_TESTING:
    logger.info("RUNNING IN PRODUCTION TESTING MODE (Limited batches)")
else:
    logger.info("RUNNING IN FULL PRODUCTION MODE")

def load_tickers():
    """Load company tickers from tickers.csv"""
    if os.path.exists(TICKERS_FILE):
        df = pd.read_csv(TICKERS_FILE)
        # Remove any empty rows
        df = df.dropna(subset=['symbol'])
        df = df[df['symbol'].str.strip() != '']
        return df['symbol'].tolist()
    else:
        logger.warning(f"Tickers file not found at {TICKERS_FILE}, returning empty list")
        return []

# =================================================================================================
# MODEL LOADING
# =================================================================================================

print("Loading models...")

# Shared NLP and Embedding models are handled by the IssueAnalyzer
nlp = issue_analyzer.nlp
embedder = issue_analyzer.embedder

def load_model_safely(model_path, model_type="embedding"):
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model path not found: {model_path}")
        print("Ensure models are baked into the Docker image during build.")
        sys.exit(1)

    print(f"Loading {model_type} model from {model_path}")
    try:
        if model_type == "embedding":
            return SentenceTransformer(model_path)
        elif model_type == "sentiment":
            # Load tokenizer - suppress regex pattern warning (known DeBERTa issue, doesn't affect functionality)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*incorrect regex pattern.*')
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                return pipeline("text-classification", model=model_path, tokenizer=tokenizer)
        else:
            return pipeline("text-classification", model=model_path)
    except Exception as e:
        print(f"CRITICAL ERROR loading {model_type} model: {e}")
        sys.exit(1)

# Load all models strictly from local paths
# sentiment_analyzer = load_model_safely(SENTIMENT_MODEL_PATH, "sentiment")  # COMMENTED OUT - Using VADER instead
interaction_classifier = load_model_safely(INTERACTION_MODEL_PATH, "interaction")
role_classifier = load_model_safely(ROLE_MODEL_PATH, "role")

# Initialize VADER sentiment analyzer (fast and simple)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()
logger.info("Using VADER for sentiment analysis (fast mode)")

print("Models loaded successfully.")

# Topics logic is now handled by issue_analyzer
# Pre-computed embeddings and matcher are internal to the class
anchor_embeddings = issue_analyzer.anchor_embeddings
anchor_metadata = issue_analyzer.anchor_metadata
matcher = issue_analyzer.matcher
EXCLUSIONS_MAP = issue_analyzer.exclusions_map
ISSUE_AREA_MAP = issue_analyzer.issue_area_map

def rejoin_fragments(df):
    """Rejoins segments split by line breaks."""
    if df.empty:
        return df
    rejoined_rows = []
    current_row = df.iloc[0].to_dict()
    for i in range(1, len(df)):
        next_row = df.iloc[i].to_dict()
        msg = str(current_row['content']).strip()
        is_fragment = not any(msg.endswith(p) for p in ['.', '?', '!', '"', '"', '"'])
        if next_row['transcript_id'] == current_row['transcript_id'] and is_fragment:
            current_row['content'] = msg + " " + str(next_row['content']).strip()
        else:
            rejoined_rows.append(current_row)
            current_row = next_row
    rejoined_rows.append(current_row)
    return pd.DataFrame(rejoined_rows)

def ensure_complete_transcripts(df):
    """
    Ensure only complete transcripts are returned.
    If the dataframe cuts off mid-transcript, remove the incomplete one.

    Args:
        df: DataFrame with transcript data

    Returns:
        DataFrame with only complete transcripts
    """
    if df.empty:
        return df

    # Get unique transcript IDs in order of appearance
    transcript_ids = df['transcript_id'].unique()

    # Check if the last transcript is complete by counting paragraphs
    # A transcript is considered complete if we have all sequential paragraphs
    complete_transcript_ids = []

    for tid in transcript_ids:
        transcript_df = df[df['transcript_id'] == tid]
        paragraphs = sorted(transcript_df['paragraph_number'].unique())

        # Check if paragraphs are sequential from 1 (or 0)
        expected_paragraphs = list(range(paragraphs[0], paragraphs[0] + len(paragraphs)))

        if paragraphs == expected_paragraphs:
            complete_transcript_ids.append(tid)
        else:
            # This transcript is incomplete, don't include it
            logger.info(f"   Excluding incomplete transcript: {tid} (has {len(paragraphs)} non-sequential paragraphs)")

    # Return only complete transcripts
    return df[df['transcript_id'].isin(complete_transcript_ids)]

# =================================================================================================
# CORE ANALYSIS
# =================================================================================================

def analyze_batch(texts):
    """
    Optimized batch analysis for topics and sentiment.
    Returns a list of result lists (one per input text).
    """
    if not texts:
        return []

    # 1. Topic Detection (Vectorized)
    # This is already fast as it uses matrix multiplication
    query_embeddings = embedder.encode(texts, convert_to_tensor=True)
    all_scores = util.cos_sim(query_embeddings, anchor_embeddings) if anchor_embeddings is not None else None
    
    # Pre-filter topics for each text
    results_by_text = []
    sentiment_queue = [] # Pairs of (text, topic_label) for later batch inference
    
    for i, text in enumerate(texts):
        # First, try spaCy Matcher (Exact Match)
        doc = nlp(text)
        matches = matcher(doc)
        found_topics = set()
        if matches:
            for match_id, start, end in matches:
                found_topics.add(nlp.vocab.strings[match_id])
            
            text_results = []
            for topic in found_topics:
                # Check for exclusionary terms
                exclusions = EXCLUSIONS_MAP.get(topic, [])
                if any(ext.lower() in text.lower() for ext in exclusions):
                    logger.info(f"      [EXCLUSION] Dropping topic '{topic}' due to exclusionary term match.")
                    continue
                    
                text_results.append({"topic": topic, "idx": i})
                sentiment_queue.append({"text": text, "text_pair": topic, "text_idx": i, "topic_idx": len(text_results)-1})
            results_by_text.append(text_results)
            continue

        # Second, fallback to Vector Similarity
        if anchor_embeddings is None:
            results_by_text.append([])
            continue

        cos_scores = all_scores[i]
        text_results = []
        for idx, score in enumerate(cos_scores):
            if score.item() >= SIMILARITY_THRESHOLD:
                topic = anchor_metadata[idx][0]
                # Check for exclusionary terms
                exclusions = EXCLUSIONS_MAP.get(topic, [])
                if any(ext.lower() in text.lower() for ext in exclusions):
                    continue
                    
                text_results.append({"topic": topic, "score": score.item(), "idx": i, "matched_anchor": anchor_metadata[idx][1]})
        
        # Sort and take Top 3
        text_results.sort(key=lambda x: x['score'], reverse=True)
        unique = {}
        for r in text_results:
            if r['topic'] not in unique: unique[r['topic']] = r
        top_topics = list(unique.values())[:3]
        
        for r in top_topics:
            sentiment_queue.append({"text": text, "text_pair": r['topic'], "text_idx": i, "topic_idx": top_topics.index(r)})
        
        results_by_text.append(top_topics)

    # 2. VADER Sentiment Analysis (Fast and Simple)
    if sentiment_queue:
        logger.info(f"      Running VADER sentiment analysis for {len(sentiment_queue)} topic pairs...")
        start_time = time.time()

        # VADER is super fast - just analyze each text directly
        for item in sentiment_queue:
            text = str(item["text"])
            # Get VADER scores
            scores = sentiment_analyzer.polarity_scores(text)

            # Determine sentiment label based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'

            # Update the result
            meta = item
            target = results_by_text[meta["text_idx"]][meta["topic_idx"]]
            target["sentiment"] = label
            target["sentiment_score"] = abs(compound)  # Use absolute compound score
            target["all_scores"] = f"pos: {scores['pos']:.2f}, neu: {scores['neu']:.2f}, neg: {scores['neg']:.2f}, compound: {compound:.2f}"

        logger.info(f"      Finished sentiment analysis in {time.time() - start_time:.2f}s")

    return results_by_text

# =================================================================================================
# MAIN PIPELINE
# =================================================================================================

def process_pipeline(config=None, return_data=False):
    """
    Run the analysis pipeline with optional configuration.

    Args:
        config: Dict with optional keys:
            - companies: Comma-separated string of company symbols, or None for tickers list
            - start_date: Start date (YYYY-MM-DD) or None
            - end_date: End date (YYYY-MM-DD) or None
            - limit: Max records to process (complete transcripts only), or None
            - latest: Number of latest transcripts to pull (overrides limit)
            - earliest: Number of earliest transcripts to pull (overrides limit)
            - mode: 'test' (50 records) or 'full' (all records)
        return_data: If True, return CSV string instead of writing to BigQuery

    Returns:
        CSV string if return_data=True, otherwise None
    """
    config = config or {}
    logger.info(f"Models verified on disk. Starting pipeline logic with config: {config}")

    # Storage for all results if returning data
    all_accumulated_results = []

    client = bigquery.Client(project=BQ_PROJECT_ID)

    # Parse configuration
    companies_param = config.get('companies')
    if companies_param:
        companies = [s.strip() for s in companies_param.split(',')]
        logger.info(f"Using provided companies: {companies}")
    else:
        companies = load_tickers()
        logger.info(f"Using tickers list: {len(companies)} total")

    start_date = config.get('start_date')
    end_date = config.get('end_date')
    mode = config.get('mode', 'test')
    custom_limit = config.get('limit')
    latest = config.get('latest')
    earliest = config.get('earliest')

    # Latest and earliest override limit
    if latest or earliest:
        limit = None
        transcript_mode = f"Latest {latest}" if latest else f"Earliest {earliest}"
    elif custom_limit:
        limit = int(custom_limit)
        transcript_mode = f"Limit {limit} (complete transcripts)"
    elif mode == 'test':
        limit = 50
        transcript_mode = "Test mode (50 records, complete transcripts)"
    elif PRODUCTION_TESTING:
        limit = 20
        transcript_mode = "Production testing (20 records)"
    else:
        limit = BATCH_SIZE
        transcript_mode = f"Batch size {BATCH_SIZE}"

    logger.info(f"Analysis parameters:")
    logger.info(f"  Companies: {len(companies)}")
    logger.info(f"  Date range: {start_date or 'All'} to {end_date or 'All'}")
    logger.info(f"  Mode: {transcript_mode}")

    total_processed = 0
    current_session_id = 0
    current_analyst = "None"
    last_transcript_id = None

    try:
        while True:
            logger.info(f"Checking for new data to process in {BQ_SOURCE_TABLE}...")

            # Build WHERE clause
            where_clauses = []

            # Company filter
            if companies:
                companies_str = "', '".join(companies)
                where_clauses.append(f"m.symbol IN ('{companies_str}')")

            # Date filters
            if start_date:
                where_clauses.append(f"m.report_date >= '{start_date}'")
            if end_date:
                where_clauses.append(f"m.report_date <= '{end_date}'")

            # Combine WHERE clauses
            if where_clauses:
                where_clause = " AND ".join(where_clauses)
                where_filter = f"AND {where_clause}"
            else:
                where_filter = ""

            # Build query - include all metadata columns
            if return_data:
                # When returning CSV, don't check for existing records
                if latest:
                    # Get latest N transcripts
                    query = f"""
                        WITH selected_transcripts AS (
                            SELECT m.transcript_id
                            FROM `{BQ_METADATA_TABLE}` m
                            WHERE 1=1
                            {where_filter}
                            ORDER BY m.report_date DESC, m.symbol
                            LIMIT {latest}
                        )
                        SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content,
                               m.* EXCEPT(transcript_id, corporation, sector),
                               cr.corporation,
                               cr.sector
                        FROM `{BQ_SOURCE_TABLE}` t
                        JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
                        LEFT JOIN `{BQ_CORP_REF_TABLE}` cr ON m.symbol = cr.Ticker
                        WHERE t.transcript_id IN (SELECT transcript_id FROM selected_transcripts)
                        ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
                    """
                elif earliest:
                    # Get earliest N transcripts
                    query = f"""
                        WITH selected_transcripts AS (
                            SELECT m.transcript_id
                            FROM `{BQ_METADATA_TABLE}` m
                            WHERE 1=1
                            {where_filter}
                            ORDER BY m.report_date ASC, m.symbol
                            LIMIT {earliest}
                        )
                        SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content,
                               m.* EXCEPT(transcript_id),
                               cr.corporation,
                               cr.sector
                        FROM `{BQ_SOURCE_TABLE}` t
                        JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
                        LEFT JOIN `{BQ_CORP_REF_TABLE}` cr ON m.symbol = cr.Ticker
                        WHERE t.transcript_id IN (SELECT transcript_id FROM selected_transcripts)
                        ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
                    """
                else:
                    # Standard limit query (will be post-processed for complete transcripts)
                    query = f"""
                        SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content,
                               m.* EXCEPT(transcript_id),
                               cr.corporation,
                               cr.sector
                        FROM `{BQ_SOURCE_TABLE}` t
                        JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
                        LEFT JOIN `{BQ_CORP_REF_TABLE}` cr ON m.symbol = cr.Ticker
                        WHERE 1=1
                        {where_filter}
                        ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
                        LIMIT {limit * 2 if limit else limit}
                    """
            else:
                # Idempotent Query: Process only rows not already in destination
                if latest:
                    query = f"""
                        WITH selected_transcripts AS (
                            SELECT m.transcript_id
                            FROM `{BQ_METADATA_TABLE}` m
                            WHERE 1=1
                            {where_filter}
                            ORDER BY m.report_date DESC, m.symbol
                            LIMIT {latest}
                        )
                        SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content,
                               m.* EXCEPT(transcript_id, corporation, sector),
                               cr.corporation,
                               cr.sector
                        FROM `{BQ_SOURCE_TABLE}` t
                        JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
                        LEFT JOIN `{BQ_CORP_REF_TABLE}` cr ON m.symbol = cr.Ticker
                        LEFT JOIN (SELECT DISTINCT transcript_id, paragraph_number FROM `{BQ_DEST_TABLE}`) e
                        ON t.transcript_id = e.transcript_id AND t.paragraph_number = e.paragraph_number
                        WHERE e.transcript_id IS NULL
                        AND t.transcript_id IN (SELECT transcript_id FROM selected_transcripts)
                        ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
                    """
                elif earliest:
                    query = f"""
                        WITH selected_transcripts AS (
                            SELECT m.transcript_id
                            FROM `{BQ_METADATA_TABLE}` m
                            WHERE 1=1
                            {where_filter}
                            ORDER BY m.report_date ASC, m.symbol
                            LIMIT {earliest}
                        )
                        SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content,
                               m.* EXCEPT(transcript_id),
                               cr.corporation,
                               cr.sector
                        FROM `{BQ_SOURCE_TABLE}` t
                        JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
                        LEFT JOIN `{BQ_CORP_REF_TABLE}` cr ON m.symbol = cr.Ticker
                        LEFT JOIN (SELECT DISTINCT transcript_id, paragraph_number FROM `{BQ_DEST_TABLE}`) e
                        ON t.transcript_id = e.transcript_id AND t.paragraph_number = e.paragraph_number
                        WHERE e.transcript_id IS NULL
                        AND t.transcript_id IN (SELECT transcript_id FROM selected_transcripts)
                        ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
                    """
                else:
                    query = f"""
                        SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content,
                               m.* EXCEPT(transcript_id),
                               cr.corporation,
                               cr.sector
                        FROM `{BQ_SOURCE_TABLE}` t
                        JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
                        LEFT JOIN `{BQ_CORP_REF_TABLE}` cr ON m.symbol = cr.Ticker
                        LEFT JOIN (SELECT DISTINCT transcript_id, paragraph_number FROM `{BQ_DEST_TABLE}`) e
                        ON t.transcript_id = e.transcript_id AND t.paragraph_number = e.paragraph_number
                        WHERE e.transcript_id IS NULL
                        {where_filter}
                        ORDER BY t.transcript_id, t.paragraph_number
                        LIMIT {limit}
                    """
            
            logger.info(f"Executing query: {query}")
            # Use .result() to wait for job completion before converting to dataframe
            query_job = client.query(query)
            df = query_job.result().to_dataframe()
            logger.info(f"Successfully fetched {len(df)} rows from BigQuery.")
            
            if df.empty:
                if total_processed == 0:
                    logger.info("No new data to process. All records in source seem to exist in destination.")
                else:
                    logger.info(f"Finished processing all new data. Total segments: {total_processed}")
                break
            
            if PRODUCTION_TESTING:
                logger.info(f"--- PRODUCTION TESTING MODE: Processing {len(df)} segments ---")
            else:
                logger.info(f"--- Processing batch of {len(df)} segments (Total so far: {total_processed}) ---")

            # Ensure only complete transcripts (important when using limit)
            if not latest and not earliest:
                df = ensure_complete_transcripts(df)
                logger.info(f"   Ensured complete transcripts: {len(df)} segments from {df['transcript_id'].nunique()} complete transcripts")

            df = rejoin_fragments(df)
            logger.info(f"   (After rejoining: {len(df)} segments)")

            texts = df['content'].astype(str).tolist()

            # PREPROCESSING: Clean operator intro text BEFORE classification
            # This prevents classifier from seeing operator language and misclassifying analyst questions
            logger.info(f"   Preprocessing texts (stripping operator intros)...")
            operator_intro_pattern = r"^(?:We'll|We will|Let's|Certainly\.?)?\s*(?:go ahead and\s+)?(?:take|move to|now go to)\s+(?:our\s+)?(?:the\s+)?(?:first|next)?\s*question\s+(?:from|is from)\s+[^.!?]+[.!?]\s+"
            cleaned_texts = []
            cleaning_count = 0
            for text in texts:
                cleaned = re.sub(operator_intro_pattern, '', text, flags=re.IGNORECASE).strip()
                if len(cleaned) < len(text):
                    cleaning_count += 1
                cleaned_texts.append(cleaned if cleaned else text)  # Keep original if cleaning removed everything

            if cleaning_count > 0:
                logger.info(f"      Cleaned operator intro from {cleaning_count} segments")

            # Use cleaned texts for classification
            # Truncate for classifiers to avoid position embedding errors
            truncated_texts = [t[:512] for t in cleaned_texts]

            # 1. Batch Classification
            # MEMORY OPTIMIZATION: Using batch_size=2 instead of 4 to reduce memory usage
            logger.info(f"   Running batch interaction classification for {len(df)} segments...")
            start_time = time.time()
            int_results = interaction_classifier(truncated_texts, batch_size=2)
            logger.info(f"   Finished interaction classification in {time.time() - start_time:.2f}s")

            logger.info(f"   Running batch role classification for {len(df)} segments...")
            start_time = time.time()
            role_results = role_classifier(truncated_texts, batch_size=2)
            logger.info(f"   Finished role classification in {time.time() - start_time:.2f}s")

            # Update texts to use cleaned versions
            texts = cleaned_texts

            # 2. Batch Topic & Sentiment Analysis
            logger.info(f"   Running batch topic/sentiment analysis...")
            enrichment_results = analyze_batch(texts)

            # 3. Assemble and Checkpoint Upload
            all_results = []
            # Session tracking regexes
            # Updated to match actual transcript patterns
            SESSION_START_PATTERNS = [
                r"next question",  # Simplified - matches "next question is from" etc.
                r"first question",  # Matches "first question from"
                r"question (?:is |will be )?(?:coming )?from",  # "question is from", "question from"
                r"(?:we'll |we will )?(?:now )?(?:go to|take)",  # "we'll go to", "we'll take", "now go to"
                r"move to the line of",
                r"go to the line of",
                r"from the line of"
            ]
            session_start_regex = re.compile("|".join(SESSION_START_PATTERNS), re.IGNORECASE)

            # Checkpoint variables
            CHECKPOINT_SIZE = 10

            # Track previous interaction type for Answer validation
            previous_interaction = None
            previous_role = None

            for i, (_, row) in enumerate(df.iterrows()):
                # Reset session tracking if we've moved to a new transcript
                if last_transcript_id is not None and row['transcript_id'] != last_transcript_id:
                    current_session_id = 0
                    previous_interaction = None
                    previous_role = None
                last_transcript_id = row['transcript_id']

                int_res = int_results[i]['label']
                role_res = role_results[i]['label']
                interaction_type = INTERACTION_ID_MAP.get(int_res, int_res)
                role_label = ROLE_ID_MAP.get(role_res, role_res)
                # Use cleaned text from preprocessing step
                text = texts[i]

                # POST-PROCESSING: Validate "Answer" labels
                # An "Answer" should only follow a "Question" from an Analyst
                # This fixes executives' opening remarks being incorrectly labeled as "Answer"
                if interaction_type == "Answer":
                    if previous_interaction != "Question" or previous_role != "Analyst":
                        # Not a valid answer context - reclassify as Admin
                        interaction_type = "Admin"

                # POST-PROCESSING: Boost "Question" detection for Analyst segments
                # If an Analyst segment contains question indicators, prioritize labeling it as "Question"
                # This handles edge cases where operator intro text confused the classifier
                if role_label == "Analyst" and interaction_type != "Question":
                    question_indicators = [
                        "have two", "have a question", "wondering", "curious", "want to ask",
                        "can you", "could you", "would you", "will you",
                        "how do", "how does", "how should", "how would",
                        "what", "why", "when", "where", "which",
                        "talk about", "comment on", "thoughts on", "perspective on"
                    ]
                    lower_text = text.lower()
                    if any(indicator in lower_text for indicator in question_indicators):
                        logger.info(f"      [OVERRIDE] Changed Analyst {interaction_type} → Question (contains question indicators)")
                        interaction_type = "Question"

                # Session tracking:
                # We trigger a new session if we see strong "next question" pattern
                lower_text = text.lower()
                is_operator = role_label == "Operator"
                has_session_start_keyword = session_start_regex.search(lower_text)

                # Broad keywords for operator when they don't match the specific start patterns
                is_transition_text = any(k in lower_text for k in ["question", "line of", "analyst"])

                if (is_operator and is_transition_text) or has_session_start_keyword:
                    current_session_id += 1

                detected = enrichment_results[i]
                if not detected:
                    detected = [{
                        "topic": None,
                        "sentiment": None,
                        "sentiment_score": None,
                        "all_scores": None,
                        "similarity_score": None,
                        "matched_anchor": None
                    }]

                for d in detected:
                    res_row = {
                        "transcript_id": row['transcript_id'],
                        "paragraph_number": row['paragraph_number'],
                        "speaker": row['speaker'],
                        "qa_session_id": current_session_id,
                        "interaction_type": interaction_type,
                        "role": role_label,
                        "issue_area": ISSUE_AREA_MAP.get(d.get('topic'), "Unknown"),
                        "issue_subtopic": d.get('topic'),
                        "sentiment_label": d.get('sentiment'),
                        "sentiment_score": d.get('sentiment_score'),
                        "all_scores": d.get('all_scores'),
                        "similarity_score": d.get('similarity_score'),
                        "matched_anchor": d.get('matched_anchor'),
                        "content": text
                    }

                    # Add all metadata columns from BigQuery
                    for col in df.columns:
                        if col not in ['transcript_id', 'paragraph_number', 'speaker', 'content']:
                            res_row[col] = row[col]

                    all_results.append(res_row)

                # Update previous interaction tracking for next iteration (outside detected loop)
                previous_interaction = interaction_type
                previous_role = role_label

                # Checkpoint Upload or accumulation
                if (i + 1) % CHECKPOINT_SIZE == 0 or (i + 1) == len(df):
                    if all_results:
                        if return_data:
                            # Accumulate results for CSV return
                            all_accumulated_results.extend(all_results)
                            logger.info(f"   [Accumulate] Processed {len(all_results)} enrichment rows (Segment {i+1}/{len(df)})")
                        else:
                            # Upload to BigQuery
                            res_df = pd.DataFrame(all_results)
                            job_config = bigquery.LoadJobConfig(
                                write_disposition="WRITE_APPEND",
                                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
                            )
                            client.load_table_from_dataframe(res_df, BQ_DEST_TABLE, job_config=job_config).result()
                            logger.info(f"   [Checkpoint] Uploaded {len(res_df)} enrichment rows (Segment {i+1}/{len(df)})")
                        all_results = [] # Clear for next checkpoint

            total_processed += len(df)

            # Exit conditions
            if mode == 'test' or PRODUCTION_TESTING:
                logger.info(f"Test mode complete. Processed {total_processed} records.")
                break

            # For full mode, continue processing batches until no data left
            # This allows processing large datasets in manageable chunks

    except Exception as e:
        logger.error(f"FATAL ERROR in pipeline: {e}", exc_info=True)
        if return_data:
            raise  # Re-raise for proper error handling in API response

    logger.info(f"Pipeline execution complete. Total records processed: {total_processed}")

    # Return CSV data if requested
    if return_data:
        if all_accumulated_results:
            results_df = pd.DataFrame(all_accumulated_results)
            csv_data = results_df.to_csv(index=False)
            logger.info(f"Returning CSV with {len(results_df)} rows")
            return csv_data
        else:
            return "No results found\n"

if __name__ == "__main__":
    run_server()
