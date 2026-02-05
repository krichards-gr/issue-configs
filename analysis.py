import os

# DISABLING TOKENIZERS PARALLELISM TO PREVENT DEADLOCKS IN CLOUD RUN
# This is critical for avoiding hangs when using transformers in a threaded/multiprocess environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import pandas as pd
from google.cloud import bigquery
import sys
import time
import re
import logging
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from generate_topics import generate_topics_json


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
        elif self.path == '/run':
            # Trigger analysis endpoint
            logger.info("Manual trigger received via /run")
            try:
                # Run synchronously to ensure Cloud Run keeps CPU allocated
                # Using threading.Thread() creates background work that gets throttled
                # immediately after the response is sent unless "CPU always allocated"
                # is enabled. Synchronous processing is safer for batch jobs.
                logger.info("Starting pipeline synchronously...")
                process_pipeline()
                
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
SENTIMENT_MODEL_PATH = os.path.join(current_dir, "models", "deberta-v3-base-absa-v1.1")

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

if PRODUCTION_TESTING:
    logger.info("RUNNING IN PRODUCTION TESTING MODE (Limited batches)")
else:
    logger.info("RUNNING IN FULL PRODUCTION MODE")

# =================================================================================================
# MODEL LOADING
# =================================================================================================

print("Loading models...")
nlp = spacy.load("en_core_web_sm")

def load_model_safely(model_path, model_type="embedding"):
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model path not found: {model_path}")
        print("Ensure models are baked into the Docker image during build.")
        sys.exit(1)
        
    print(f"Loading {model_type} model from {model_path}")
    try:
        if model_type == "embedding":
            return SentenceTransformer(model_path)
        else:
            return pipeline("text-classification", model=model_path)
    except Exception as e:
        print(f"CRITICAL ERROR loading {model_type} model: {e}")
        sys.exit(1)

# Load all models strictly from local paths
embedder = load_model_safely(EMBEDDING_MODEL_PATH, "embedding")
sentiment_analyzer = load_model_safely(SENTIMENT_MODEL_PATH, "sentiment")
interaction_classifier = load_model_safely(INTERACTION_MODEL_PATH, "interaction")
role_classifier = load_model_safely(ROLE_MODEL_PATH, "role")

print("Models loaded successfully.")

# =================================================================================================
# DATA LOADING & UTILITIES
# =================================================================================================

def load_topics(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Topics file not found at {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('topics', [])

topics_data = load_topics(TOPICS_FILE)
# Create a map for quick exclusion lookups
# Create a map for quick exclusion lookups
EXCLUSIONS_MAP = {t['label']: t.get('exclusions', []) for t in topics_data}
# Create a map for issue area lookups
ISSUE_AREA_MAP = {t['label']: t.get('issue_area', 'Unknown') for t in topics_data}

# Prepare spaCy Matcher
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
for topic in topics_data:
    label = topic['label']
    patterns = topic.get('patterns', [])
    if patterns:
        matcher.add(label, patterns)

# Pre-compute embeddings for all anchor terms
all_anchors_text = []
anchor_metadata = [] 
for topic in topics_data:
    for anchor in topic.get('anchors', []):
        all_anchors_text.append(anchor)
        anchor_metadata.append((topic['label'], anchor))

if all_anchors_text:
    anchor_embeddings = embedder.encode(all_anchors_text, convert_to_tensor=True)
else:
    anchor_embeddings = None

def rejoin_fragments(df):
    """Rejoins segments split by line breaks."""
    if df.empty:
        return df
    rejoined_rows = []
    current_row = df.iloc[0].to_dict()
    for i in range(1, len(df)):
        next_row = df.iloc[i].to_dict()
        msg = str(current_row['content']).strip()
        is_fragment = not any(msg.endswith(p) for p in ['.', '?', '!', '"', '“', '”'])
        if next_row['transcript_id'] == current_row['transcript_id'] and is_fragment:
            current_row['content'] = msg + " " + str(next_row['content']).strip()
        else:
            rejoined_rows.append(current_row)
            current_row = next_row
    rejoined_rows.append(current_row)
    return pd.DataFrame(rejoined_rows)

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
                    
                text_results.append({"topic": topic, "score": score.item(), "idx": i})
        
        # Sort and take Top 3
        text_results.sort(key=lambda x: x['score'], reverse=True)
        unique = {}
        for r in text_results:
            if r['topic'] not in unique: unique[r['topic']] = r
        top_topics = list(unique.values())[:3]
        
        for r in top_topics:
            sentiment_queue.append({"text": text, "text_pair": r['topic'], "text_idx": i, "topic_idx": top_topics.index(r)})
        
        results_by_text.append(top_topics)

    # 2. Batch Sentiment Inference
    if sentiment_queue:
        logger.info(f"      Running batch sentiment analysis for {len(sentiment_queue)} topic pairs...")
        start_time = time.time()
        # Prepare inputs as parallel lists for better pipeline batch handling
        texts_input = [item["text"] for item in sentiment_queue]
        pairs_input = [item["text_pair"] for item in sentiment_queue]
        
        # Custom batch processing to handle text pairs correctly
        # (Pipeline's text_pair argument doesn't handle batches of pairs intuitively)
        batch_size = 4
        sent_results_flat = []
        
        try:
            for i in range(0, len(texts_input), batch_size):
                batch_texts = texts_input[i:i+batch_size]
                batch_pairs = pairs_input[i:i+batch_size]
                
                # Tokenize batch of pairs
                inputs = sentiment_analyzer.tokenizer(
                    batch_texts, 
                    text_pair=batch_pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                
                # Move inputs to same device as model
                device = sentiment_analyzer.model.device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Inference
                with torch.no_grad():
                    outputs = sentiment_analyzer.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Process results
                id2label = sentiment_analyzer.model.config.id2label
                for j in range(len(batch_texts)):
                    # Get top prediction (max score)
                    score, label_idx = torch.max(probs[j], dim=0)
                    label = id2label[label_idx.item()]
                    sent_results_flat.append({"label": label, "score": score.item()})

            logger.info(f"      Finished sentiment analysis in {time.time() - start_time:.2f}s")
            
            for i, res in enumerate(sent_results_flat):
                meta = sentiment_queue[i]
                target = results_by_text[meta["text_idx"]][meta["topic_idx"]]
                target["sentiment"] = res['label']
                target["sentiment_score"] = res['score']
                
        except Exception as e:
            logger.error(f"Error during sentiment batch processing: {e}")
            # Fallback or skip
            pass

    return results_by_text

# =================================================================================================
# MAIN PIPELINE
# =================================================================================================

def process_pipeline():
    logger.info("Models verified on disk. Starting pipeline logic...")
    client = bigquery.Client(project=BQ_PROJECT_ID)
    
    total_processed = 0
    current_session_id = 0
    current_analyst = "None"
    last_transcript_id = None
    
    try:
        while True:
            logger.info(f"Checking for new data to process in {BQ_SOURCE_TABLE}...")
            # Determine query limit based on testing mode
            current_limit = 20 if PRODUCTION_TESTING else BATCH_SIZE
            
            # Idempotent Query: Process only rows not already in destination
            query = f"""
                SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content
                FROM `{BQ_SOURCE_TABLE}` t
                LEFT JOIN (SELECT DISTINCT transcript_id, paragraph_number FROM `{BQ_DEST_TABLE}`) e
                ON t.transcript_id = e.transcript_id AND t.paragraph_number = e.paragraph_number
                WHERE e.transcript_id IS NULL
                ORDER BY t.transcript_id, t.paragraph_number
                LIMIT {current_limit}
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
        
            df = rejoin_fragments(df)
            logger.info(f"   (After rejoining: {len(df)} segments)")

            texts = df['content'].astype(str).tolist()
            # Truncate for classifiers to avoid position embedding errors
            truncated_texts = [t[:512] for t in texts]

            # 1. Batch Classification
            logger.info(f"   Running batch interaction classification for {len(df)} segments...")
            start_time = time.time()
            int_results = interaction_classifier(truncated_texts, batch_size=4)
            logger.info(f"   Finished interaction classification in {time.time() - start_time:.2f}s")

            logger.info(f"   Running batch role classification for {len(df)} segments...")
            start_time = time.time()
            role_results = role_classifier(truncated_texts, batch_size=4)
            logger.info(f"   Finished role classification in {time.time() - start_time:.2f}s")

            # 2. Batch Topic & Sentiment Analysis
            logger.info(f"   Running batch topic/sentiment analysis...")
            enrichment_results = analyze_batch(texts)

            # 3. Assemble and Checkpoint Upload
            all_results = []
            # Session tracking regexes
            SESSION_START_PATTERNS = [
                r"next question (?:comes|is coming)",
                r"next we (?:have|will go to)",
                r"question (?:comes|is coming) from",
                r"your first question (?:comes|is coming)",
                r"move to the line of",
                r"go to the line of",
                r"from the line of",
                r"comes? from the line of"
            ]
            session_start_regex = re.compile("|".join(SESSION_START_PATTERNS), re.IGNORECASE)
            
            # Analyst extraction regex
            # Optimized to match common "Operator" phrasing
            intro_regex = re.compile(
                r"(?:line of|comes from|is from|from|at)\s+(?:the line of\s+)?([^,.]+?)\s+(?:with|from|at|is coming)", 
                re.IGNORECASE
            )

            # Checkpoint variables
            CHECKPOINT_SIZE = 10 
            
            for i, (_, row) in enumerate(df.iterrows()):
                # Reset session tracking if we've moved to a new transcript
                if last_transcript_id is not None and row['transcript_id'] != last_transcript_id:
                    current_session_id = 0
                    current_analyst = "None"
                last_transcript_id = row['transcript_id']

                int_res = int_results[i]['label']
                role_res = role_results[i]['label']
                interaction_type = INTERACTION_ID_MAP.get(int_res, int_res)
                role_label = ROLE_ID_MAP.get(role_res, role_res)
                text = str(row['content'])

                # Session tracking:
                # We trigger a new session if:
                # 1. Role is Operator (strong role signal) AND we see any Q&A keyword
                # 2. OR if we see a very strong "next question" pattern (strong text signal) regardless of role
                lower_text = text.lower()
                is_operator = role_label == "Operator"
                has_session_start_keyword = session_start_regex.search(lower_text)
                
                # Broad keywords for operator when they don't match the specific start patterns
                is_transition_text = any(k in lower_text for k in ["question", "line of", "analyst"])

                if (is_operator and is_transition_text) or has_session_start_keyword:
                    current_session_id += 1
                    match = intro_regex.search(text)
                    if match:
                        current_analyst = match.group(1).strip()
                    elif is_operator and "question" in lower_text:
                        # Fallback if regex fails but we know it's a question transition
                        current_analyst = "Unknown Analyst"
                    # else: keep current_analyst from previous turn or leave as is if it's the operator again

                detected = enrichment_results[i]
                if not detected: 
                    detected = [{"topic": None, "sentiment": None, "sentiment_score": None}]

                for d in detected:
                    all_results.append({
                        "transcript_id": row['transcript_id'],
                        "paragraph_number": row['paragraph_number'],
                        "speaker": row['speaker'],
                        "qa_session_id": current_session_id,
                        "qa_session_label": current_analyst,
                        "interaction_type": interaction_type,
                        "role": role_label,
                        "role": role_label,
                        "topic": d.get('topic'),
                        "issue_area": ISSUE_AREA_MAP.get(d.get('topic'), "Unknown"),
                        "issue_subtopic": d.get('topic'),
                        "sentiment_label": d.get('sentiment'),
                        "sentiment_score": d.get('sentiment_score')
                    })

                # Checkpoint Upload
                if (i + 1) % CHECKPOINT_SIZE == 0 or (i + 1) == len(df):
                    if all_results:
                        res_df = pd.DataFrame(all_results)
                        job_config = bigquery.LoadJobConfig(
                            write_disposition="WRITE_APPEND",
                            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
                        )
                        client.load_table_from_dataframe(res_df, BQ_DEST_TABLE, job_config=job_config).result()
                        logger.info(f"   [Checkpoint] Uploaded {len(res_df)} enrichment rows (Segment {i+1}/{len(df)})")
                        all_results = [] # Clear for next checkpoint

            total_processed += len(df)
            
            # Exit if in testing mode after first batch
            if PRODUCTION_TESTING:
                logger.info("Production testing complete. Set 'PRODUCTION_MODE=true' env var for full run.")
                break
    except Exception as e:
        logger.error(f"FATAL ERROR in pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    run_server()
