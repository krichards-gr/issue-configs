import json
import spacy
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import sys
import torch
from generate_topics import generate_topics_json

"""
local_analysis.py

This script demonstrates the logic used in the production pipeline for topic matching.
It is synchronized with the structure of the earnings-call-transcript-analysis repo.
"""

# =================================================================================================
# CONFIGURATION & SETUP
# =================================================================================================

# Automatically regenerate topics.json from the CSV inputs
generate_topics_json()

TOPICS_FILE = 'topics.json'
SIMILARITY_THRESHOLD = 0.7

# =================================================================================================
# MODEL LOADING
# =================================================================================================

print("Loading models (this may take a moment)...")
try:
    nlp = spacy.load("en_core_web_sm")
    # Using a fast, small model for local demonstration
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"ERROR: Could not load models. Ensure 'en_core_web_sm' is downloaded.")
    print(f"Run: python -m spacy download en_core_web_sm")
    sys.exit(1)

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
EXCLUSIONS_MAP = {t['label']: t.get('exclusions', []) for t in topics_data}
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
    print(f"Pre-computing embeddings for {len(all_anchors_text)} anchor terms...")
    anchor_embeddings = embedder.encode(all_anchors_text, convert_to_tensor=True)
else:
    anchor_embeddings = None

# =================================================================================================
# CORE ANALYSIS
# =================================================================================================

def analyze_text(text):
    """
    Performs topic matching (Exact Match & Vector Similarity) for a single text.
    """
    doc = nlp(text)
    
    # 1. spaCy Matcher (Exact Patterns)
    matches = matcher(doc)
    found_topics = set()
    for match_id, start, end in matches:
        found_topics.add(nlp.vocab.strings[match_id])

    # 2. Vector Similarity Fallback / Augmentation
    if anchor_embeddings is not None:
        query_embedding = embedder.encode(text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, anchor_embeddings)[0]
        
        for idx, score in enumerate(cos_scores):
            if score.item() >= SIMILARITY_THRESHOLD:
                topic_label = anchor_metadata[idx][0]
                found_topics.add(topic_label)

    # 3. Apply Exclusions
    results = []
    for topic in found_topics:
        exclusions = EXCLUSIONS_MAP.get(topic, [])
        is_excluded = False
        for ext in exclusions:
            if ext.lower() in text.lower():
                is_excluded = True
                break
        
        if not is_excluded:
            results.append({
                "topic": topic,
                "issue_area": ISSUE_AREA_MAP.get(topic, "Unknown")
            })
            
    return results

def run_demo():
    print("\n--- Starting Analysis Demo ---\n")
    
    sample_texts = [
        "Our committed net zero targets are driving our strategy.",
        "The patient's plastic surgery went as planned.", # Should be excluded (Climate Change)
        "We are seeing strong growth in climate disclosure and carbon credits.",
        "We prioritize workers' rights and fair wages across our operations."
    ]

    for text in sample_texts:
        print(f"TEXT: \"{text}\"")
        detected = analyze_text(text)
        
        if not detected:
            print("  - No topics detected.")
        else:
            for d in detected:
                print(f"  [RESULT] Detected Topic: '{d['topic']}' (Area: {d['issue_area']})")
        
        print("-" * 40)

if __name__ == "__main__":
    run_demo()
