import json
import spacy
from sentence_transformers import SentenceTransformer, util
import os
import sys
import torch
from generate_topics import generate_all

class IssueAnalyzer:
    def __init__(self, topics_file='topics.json', similarity_threshold=0.7, nlp_model="en_core_web_sm", embedding_model="all-MiniLM-L6-v2"):
        # Always regenerate topics.json to ensure latest configs are used
        # generate_all(from_raw=True)

        self.topics_file = topics_file
        self.similarity_threshold = similarity_threshold

        print(f"Loading models ({nlp_model}, {embedding_model})...")
        try:
            # MEMORY OPTIMIZATION: Disable unused spaCy components to reduce memory
            # Only keep tokenizer and basic attributes needed for matcher
            self.nlp = spacy.load(nlp_model, disable=["parser", "ner"])
            self.embedder = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"ERROR: Could not load models: {e}")
            sys.exit(1)

        self.load_config()

    def load_config(self):
        if not os.path.exists(self.topics_file):
            print(f"Error: Topics file not found at {self.topics_file}")
            self.topics_data = []
        else:
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.topics_data = data.get('topics', [])

        self.exclusions_map = {t['label']: t.get('exclusions', []) for t in self.topics_data}
        self.issue_area_map = {t['label']: t.get('issue_area', 'Unknown') for t in self.topics_data}
        self.issue_subtopic_map = {t['label']: t.get('issue_subtopic', 'Unknown') for t in self.topics_data}

        # Prepare spaCy Matcher
        from spacy.matcher import Matcher
        self.matcher = Matcher(self.nlp.vocab)
        for topic in self.topics_data:
            label = topic['label']
            patterns = topic.get('patterns', [])
            if patterns:
                self.matcher.add(label, patterns)

        # Pre-compute embeddings for all anchor terms
        self.all_anchors_text = []
        self.anchor_metadata = []
        for topic in self.topics_data:
            for anchor in topic.get('anchors', []):
                self.all_anchors_text.append(anchor)
                self.anchor_metadata.append((topic['label'], anchor))

        if self.all_anchors_text:
            print(f"Pre-computing embeddings for {len(self.all_anchors_text)} anchor terms...")
            self.anchor_embeddings = self.embedder.encode(self.all_anchors_text, convert_to_tensor=True)
        else:
            self.anchor_embeddings = None

    def analyze_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return []

        doc = self.nlp(text)

        # 1. spaCy Matcher (Exact Patterns)
        matches = self.matcher(doc)
        found_topics = set()
        for match_id, start, end in matches:
            found_topics.add(self.nlp.vocab.strings[match_id])

        # 2. Vector Similarity Fallback / Augmentation
        if self.anchor_embeddings is not None:
            query_embedding = self.embedder.encode(text, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.anchor_embeddings)[0]

            for idx, score in enumerate(cos_scores):
                if score.item() >= self.similarity_threshold:
                    topic_label = self.anchor_metadata[idx][0]
                    found_topics.add(topic_label)

        # 3. Apply Exclusions
        results = []
        for topic in found_topics:
            exclusions = self.exclusions_map.get(topic, [])
            is_excluded = False
            for ext in exclusions:
                if ext.lower() in text.lower():
                    is_excluded = True
                    break

            if not is_excluded:
                results.append({
                    "topic": topic,
                    "issue_area": self.issue_area_map.get(topic, "Unknown"),
                    "issue_subtopic": self.issue_subtopic_map.get(topic, "Unknown")
                })

        return results
