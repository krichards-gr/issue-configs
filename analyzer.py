"""
analyzer.py

Core NLP engine for issue topic detection.

The IssueAnalyzer class loads the compiled topics.json config and applies a
three-step detection process to each piece of text:

  1. Exact pattern matching  — fast, high-precision keyword/phrase matching
                               via spaCy's token Matcher.
  2. Semantic similarity     — catches paraphrases and near-synonyms using
                               cosine similarity against anchor phrase embeddings
                               (SentenceTransformer). Only runs if anchor phrases
                               are defined and no exact match was found.
  3. Exclusion filtering     — drops any topic whose exclusionary terms appear
                               in the text, reducing false positives.

This class is shared by label_files.py (local CLI) and analysis.py (Cloud Run),
ensuring identical detection logic in both environments.
"""

import json
import os
import sys
import torch
import spacy
from spacy.matcher import Matcher
from sentence_transformers import SentenceTransformer, util
from generate_topics import generate_all  # noqa: F401 (imported for optional regeneration)


class IssueAnalyzer:
    def __init__(
        self,
        topics_file='topics.json',
        similarity_threshold=0.7,
        nlp_model="en_core_web_sm",
        embedding_model="all-MiniLM-L6-v2"
    ):
        """
        Initialize the analyzer by loading NLP models and the topics config.

        Args:
            topics_file:          Path to the compiled topics.json config file.
            similarity_threshold: Minimum cosine similarity score (0–1) for a
                                  semantic anchor match to be accepted.
            nlp_model:            spaCy model to use for tokenization and matching.
            embedding_model:      SentenceTransformer model for anchor embeddings.
        """
        self.topics_file = topics_file
        self.similarity_threshold = similarity_threshold

        print(f"Loading NLP models ({nlp_model}, {embedding_model})...")
        try:
            # Disable unused spaCy pipeline components to reduce memory usage.
            # We only need the tokenizer and vocabulary for the Matcher.
            self.nlp = spacy.load(nlp_model, disable=["parser", "ner"])
            self.embedder = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"Error: Could not load NLP models: {e}")
            sys.exit(1)

        self._load_config()

    def _load_config(self):
        """
        Load topics.json and set up internal data structures:
          - spaCy Matcher populated with all keyword patterns
          - Pre-computed embeddings for all anchor phrases
          - Lookup maps for exclusions, issue areas, and subtopic labels
        """
        if not os.path.exists(self.topics_file):
            print(f"Warning: Topics config not found at '{self.topics_file}'. "
                  "Run 'python generate_topics.py' to generate it.")
            self.topics_data = []
        else:
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.topics_data = data.get('topics', [])

        # Build lookup maps for quick access during analysis
        self.exclusions_map = {t['label']: t.get('exclusions', []) for t in self.topics_data}
        self.issue_area_map = {t['label']: t.get('issue_area', 'Unknown') for t in self.topics_data}
        self.issue_subtopic_map = {t['label']: t.get('issue_subtopic', 'Unknown') for t in self.topics_data}

        # Register all keyword patterns with the spaCy Matcher.
        # Each pattern is a list of token dicts, e.g. [{"LOWER": "climate"}, {"LOWER": "change"}].
        self.matcher = Matcher(self.nlp.vocab)
        for topic in self.topics_data:
            patterns = topic.get('patterns', [])
            if patterns:
                self.matcher.add(topic['label'], patterns)

        # Pre-compute embeddings for all anchor phrases upfront so that
        # analyze_text() can do a single matrix similarity operation per call.
        self.all_anchors_text = []
        self.anchor_metadata = []  # parallel list: (topic_label, anchor_phrase)
        for topic in self.topics_data:
            for anchor in topic.get('anchors', []):
                self.all_anchors_text.append(anchor)
                self.anchor_metadata.append((topic['label'], anchor))

        if self.all_anchors_text:
            print(f"Pre-computing embeddings for {len(self.all_anchors_text)} anchor phrases...")
            self.anchor_embeddings = self.embedder.encode(
                self.all_anchors_text, convert_to_tensor=True
            )
        else:
            self.anchor_embeddings = None

    def analyze_text(self, text):
        """
        Detect issue topics in a piece of text.

        Detection runs in three steps:
          1. Exact match — spaCy Matcher checks for keyword/phrase hits.
          2. Semantic fallback — if no exact match, cosine similarity against
             anchor phrase embeddings (only if anchors are configured).
          3. Exclusion filter — drops any topic whose exclusionary terms appear
             anywhere in the text.

        Args:
            text: The string to analyze.

        Returns:
            List of dicts, one per detected topic (after exclusions):
              [{"topic": str, "issue_area": str, "issue_subtopic": str}, ...]
            Returns an empty list if no topics are detected.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        doc = self.nlp(text)

        # Step 1: Exact keyword/phrase matching via spaCy Matcher
        matches = self.matcher(doc)
        found_topics = set()
        for match_id, start, end in matches:
            found_topics.add(self.nlp.vocab.strings[match_id])

        # Step 2: Semantic similarity fallback (only if no exact matches were found)
        if not found_topics and self.anchor_embeddings is not None:
            query_embedding = self.embedder.encode(text, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.anchor_embeddings)[0]
            for idx, score in enumerate(cos_scores):
                if score.item() >= self.similarity_threshold:
                    found_topics.add(self.anchor_metadata[idx][0])

        # Step 3: Exclusion filtering — drop topics whose exclusionary terms appear in the text
        results = []
        text_lower = text.lower()
        for topic in found_topics:
            exclusions = self.exclusions_map.get(topic, [])
            if any(exc in text_lower for exc in exclusions):
                continue  # This topic is disqualified for this text
            results.append({
                "topic": topic,
                "issue_area": self.issue_area_map.get(topic, "Unknown"),
                "issue_subtopic": self.issue_subtopic_map.get(topic, "Unknown")
            })

        return results
