import json
import os
import spacy
from sentence_transformers import SentenceTransformer, util
import torch

def run_high_fidelity_demonstration():
    """
    This script demonstrates the EXACT logic used in the production pipeline.
    It uses:
    1. spaCy Matcher for precise keyword patterns.
    2. SentenceTransformers for semantic similarity of anchor terms.
    3. Exclusionary logic for noise reduction.
    """
    print("=== Production-Grade Issue Configs Demo ===\n")

    # 1. Configuration & Paths
    # In a real scenario, you would point these to your model storage
    # Here we assume the models are already downloaded or available via HuggingFace
    print("Loading models (this may take a moment)...")
    try:
        nlp = spacy.load("en_core_web_sm")
        # Use a small, fast model for the demo
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"ERROR: Could not load models. Ensure 'en_core_web_sm' is downloaded.")
        print(f"Run: python -m spacy download en_core_web_sm")
        return

    # 2. Load the Topic Configuration
    config_path = 'test_topics.json'
    if not os.path.exists(config_path):
        print(f"ERROR: '{config_path}' not found. Please run 'generator.py' first!")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
        topics_data = config_data.get('topics', [])

    # 3. Setup spaCy Matcher
    from spacy.matcher import Matcher
    matcher = Matcher(nlp.vocab)
    
    # Store exclusions and anchors for quick lookup
    exclusions_map = {}
    anchor_metadata = [] # (topic_label, anchor_text)
    all_anchors_text = []

    for topic in topics_data:
        label = topic['label']
        exclusions_map[label] = topic.get('exclusions', [])
        
        # Add Patterns to Matcher
        patterns = topic.get('patterns', [])
        if patterns:
            matcher.add(label, patterns)
            
        # Collect Anchors for Embedding
        anchors = topic.get('anchors', [])
        for anchor in anchors:
            all_anchors_text.append(anchor)
            anchor_metadata.append((label, anchor))

    # Pre-compute Embeddings for all anchor terms at once (Efficient!)
    print(f"Pre-computing embeddings for {len(all_anchors_text)} anchor terms...")
    anchor_embeddings = embedder.encode(all_anchors_text, convert_to_tensor=True)

    # 4. Define Sample Text
    sample_texts = [
        "Our committed net zero targets are driving our strategy.",
        "The patient's plastic surgery went as planned.", # Should be excluded
        "We are seeing strong growth in climate disclosure and carbon credits."
    ]

    print("\n--- Starting Analysis ---\n")

    for text in sample_texts:
        print(f"TEXT: \"{text}\"")
        doc = nlp(text)
        
        # A. Pattern Matching (Priority)
        matches = matcher(doc)
        found_topics = set()
        for match_id, start, end in matches:
            topic_label = nlp.vocab.strings[match_id]
            found_topics.add(topic_label)

        # B. Semantic Similarity Matching (Fallback or Augmentation)
        query_embedding = embedder.encode(text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, anchor_embeddings)[0]
        
        similarity_threshold = 0.7
        for idx, score in enumerate(cos_scores):
            if score.item() >= similarity_threshold:
                topic_label, matched_anchor = anchor_metadata[idx]
                found_topics.add(topic_label)

        # C. Apply Exclusions & Output Results
        if not found_topics:
            print("  - No topics detected.")
        else:
            for topic in found_topics:
                # Check for negative filtering (Exclusions)
                exclusions = exclusions_map.get(topic, [])
                is_excluded = False
                for ext in exclusions:
                    if ext.lower() in text.lower():
                        print(f"  [EXCLUSION] Dropped topic '{topic}' because of term: '{ext}'")
                        is_excluded = True
                        break
                
                if not is_excluded:
                    print(f"  [RESULT] Detected Topic: '{topic}'")
        
        print("-" * 40)

if __name__ == "__main__":
    run_high_fidelity_demonstration()
