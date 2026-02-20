# Issue Definitions Configuration Framework

A local toolkit for defining, compiling, and applying issue topic labels to text data using keyword patterns, semantic anchors, and exclusionary filters.

---

## Quickstart

### Label a file right now

```bash
python label_files.py my_data.csv
```

You'll be prompted to select which column contains the text to analyze. Results are saved to `outputs/`.

> **First run only:** Dependencies and NLP models (~80MB) are downloaded automatically.

---

### Updated your topic definitions? Recompile first.

If you've edited `issue_config_inputs_raw.csv`, pass `--from_raw` to regenerate the compiled config before labeling:

```bash
python label_files.py my_data.csv --from_raw
```

Or regenerate the config on its own (without labeling anything):

```bash
python generate_topics.py --from_raw
```

---


## How It Works

The system follows a **Source → Compile → Apply** pipeline:

```
issue_config_inputs_raw.csv   ←── Edit this to define topics
         │
         │  (--from_raw)
         ▼
updated_issue_config_inputs.csv   ←── Intermediate long-format CSV
         │
         ▼
      topics.json   ←── Compiled config used by the analyzer
         │
         ▼
    label_files.py   ──► outputs/your_file_topic_labeled_YYYYMMDD.csv
```

When you run without `--from_raw`, the pipeline **skips the first step** and reads directly from the intermediate CSV. This is faster and appropriate when you haven't changed the raw inputs.

---

## Repository Layout

```
issue-configs/
│
├── issue_config_inputs_raw.csv      # SOURCE OF TRUTH — edit topics here
├── updated_issue_config_inputs.csv  # Intermediate CSV (auto-generated)
├── topics.json                      # Compiled NLP config (auto-generated)
│
├── label_files.py       # CLI tool: label any CSV or XLSX file
├── generate_topics.py   # CLI tool: compile topics.json from CSV inputs
├── analyzer.py          # Core detection engine (used by all scripts)
├── setup_utils.py       # Auto-installer for dependencies and models
├── local_analysis.py    # Smoke test: run detector against sample texts
│
├── test_data.csv        # Sample data for testing
├── requirements.txt     # Python dependencies
│
└── (Production / Cloud Run)
    ├── analysis.py       # Cloud Run pipeline (BigQuery → enriched output)
    ├── Dockerfile        # Container definition
    └── cloudbuild.yaml   # GCP CI/CD build configuration
```

---

## Defining Topics (`issue_config_inputs_raw.csv`)

Each row defines one **issue subtopic**. Columns:

| Column | Description |
| :--- | :--- |
| `issue_area` | Broad category (e.g. `Climate Change & ESG`) |
| `issue_subtopic` | Specific topic label (e.g. `Climate Change`) |
| `pattern` | Semicolon-separated keywords/phrases for exact matching |
| `exclusionary_term` | Semicolon-separated terms that disqualify a match |
| `anchor_phrases` | Semicolon-separated phrases for semantic similarity fallback |

**Example row:**
```
issue_area: Climate Change & ESG
issue_subtopic: Climate Change
pattern: climate change;#climatechange;net zero;carbon emissions
exclusionary_term: surgery
anchor_phrases: environmental impact;emissions reduction
```

### How matching works

| Method | Input column | Logic |
| :--- | :--- | :--- |
| **Exact match** | `pattern` | spaCy token matcher — fast and precise |
| **Semantic fallback** | `anchor_phrases` | Cosine similarity via SentenceTransformer (threshold: 0.7) |
| **Exclusion filter** | `exclusionary_term` | If found in text, the topic label is dropped |

Multi-word patterns (e.g. `climate change`) are matched as token sequences, not substrings.

---

## CLI Reference

### `label_files.py` — Label a file

```
python label_files.py <input_file> [--column COLUMN] [--from_raw]

Arguments:
  input_file          Path to a .csv or .xlsx file to label

Options:
  --column COLUMN     Name of the column containing text to analyze.
                      If omitted, you will be prompted to choose interactively.
  --from_raw          Re-read issue_config_inputs_raw.csv and regenerate
                      updated_issue_config_inputs.csv and topics.json before
                      labeling. Use this whenever you have edited the raw inputs.
```

**Output columns added:**
- `issue_subtopics` — detected topic labels, semicolon-separated
- `issue_areas` — corresponding issue area labels, semicolon-separated

---

### `generate_topics.py` — Compile config only

```
python generate_topics.py [--from_raw]

Options:
  --from_raw    Re-read issue_config_inputs_raw.csv and regenerate the
                intermediate CSV before compiling topics.json.
                Without this flag, compiles from the existing intermediate CSV.
```

---

## Production Notes

`analysis.py` is the Cloud Run production pipeline. It reads earnings call transcripts from BigQuery, applies topic detection via `IssueAnalyzer`, and writes enriched results back to BigQuery. It is **not** intended for local use.

`Dockerfile` and `cloudbuild.yaml` handle container builds and GCP deployment — see those files for configuration details.
