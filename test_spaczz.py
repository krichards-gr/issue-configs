import spacy
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
matcher = FuzzyMatcher(nlp.vocab)

# Use list of Doc objects for fuzzy matching
matcher.add("CLIMATE_CHANGE", [nlp("climate change")])

doc = nlp("I believe in climite changr because science.")
matches = matcher(doc)

for match_id, start, end, ratio, pattern in matches:
    print(f"Match: {nlp.vocab.strings[match_id]}, Text: '{doc[start:end]}', Ratio: {ratio}, Pattern: '{pattern}'")
