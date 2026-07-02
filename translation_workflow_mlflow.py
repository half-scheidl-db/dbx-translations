# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Architecture Overview
# MAGIC %md
# MAGIC # Translation Workflow with MLflow Tracing & LLM-as-a-Judge
# MAGIC
# MAGIC A production-ready translation pipeline for PDF documents, using:
# MAGIC - **MLflow Tracing** for full observability of every LLM call
# MAGIC - **MLflow GenAI Evaluate** (`mlflow.genai.evaluate()`) for structured quality assessment
# MAGIC - **Custom Scorers** replacing raw `ai_query()` judge prompts
# MAGIC - **Evaluation Datasets** for reproducible benchmarks
# MAGIC - **Production Monitoring** with registered scorers
# MAGIC
# MAGIC ### Pipeline Architecture
# MAGIC ```
# MAGIC ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐
# MAGIC │  PDF Docs   │───▶│  Parse PDF  │───▶│  Translate   │───▶│   Back      │───▶│  MLflow Evaluate │
# MAGIC │  (Volume)   │    │  (ai_parse  │    │  to English  │    │  Translate  │    │  (Custom Scorer) │
# MAGIC │             │    │  _document) │    │  (ai_query)  │    │  (ai_query) │    │                  │
# MAGIC └─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘    └──────────────────┘
# MAGIC        │                                                                              │
# MAGIC        │              ┌──────────────────────────────────────────────────────┐         │
# MAGIC        └─────────────▶│  MLflow Experiment: traces, metrics, eval datasets   │◀────────┘
# MAGIC                       └──────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Key Improvements over Raw `ai_query()` Judge
# MAGIC | Aspect | Old Approach | New Approach |
# MAGIC | --- | --- | --- |
# MAGIC | Judging | Raw `ai_query()` with hand-crafted prompt | `mlflow.genai.evaluate()` with typed scorers |
# MAGIC | Observability | None | Full MLflow tracing (latency, tokens, spans) |
# MAGIC | Reproducibility | Ad-hoc runs | Versioned evaluation datasets |
# MAGIC | Monitoring | Manual queries | Registered scorers with auto-sampling |
# MAGIC | Experiment Tracking | None | MLflow experiments for A/B model comparison |

# COMMAND ----------

# DBTITLE 1,Setup: Install MLflow and Dependencies
# MAGIC %pip install --upgrade mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
import mlflow
from mlflow.entities import SpanType

# =============================================================================
# Configuration
# =============================================================================

# Source volume containing PDFs
SOURCE_VOLUME = "/Volumes/sandbox/hfsl_icf/icf"

# Output catalog/schema for pipeline tables
OUTPUT_CATALOG = "sandbox"
OUTPUT_SCHEMA = "hfsl_icf"

# LLM endpoints — use the cheapest model that handles each task
LLM_ENDPOINT = "databricks-gemini-3-5-flash"       # Translation + Judge (needs quality)
LLM_ENDPOINT_FAST = "databricks-claude-haiku-4-5"    # Language detection (trivial task)

# Endpoints for model comparison (cheap → mid → premium, 3 providers)
COMPARISON_ENDPOINTS = [
    "databricks-gpt-5-4-nano",        # Cheapest: OpenAI nano (lightweight)
    "databricks-gemini-3-5-flash",    # Mid: Google (fast & efficient)
    "databricks-claude-sonnet-4-5",   # Premium: Anthropic (hybrid reasoning)
]

# Quality threshold (binary: pass/fail is more calibrated than 1-10 scales)
QUALITY_THRESHOLD = 7

# MLflow Experiment - tracks all runs, traces, and evaluations
CURRENT_USER = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
EXPERIMENT_PATH = f"/Users/{CURRENT_USER}/translation_quality_experiment"
mlflow.set_experiment(EXPERIMENT_PATH)

spark.sql(f"""USE CATALOG {OUTPUT_CATALOG}""")
spark.sql(f"""USE SCHEMA {OUTPUT_SCHEMA}""")
print(f"Source Volume: {SOURCE_VOLUME}")
print(f"Output: {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}")
print(f"LLM Endpoint: {LLM_ENDPOINT}")
print(f"MLflow Experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# DBTITLE 1,Step 1: Parse PDFs from Volume (SQL)
# MAGIC %sql
# MAGIC -- =============================================================================
# MAGIC -- STEP 1: Parse PDFs from Volume
# MAGIC -- =============================================================================
# MAGIC -- Uses ai_parse_document() to extract text content from binary PDFs.
# MAGIC -- Results are persisted for downstream Python-based translation pipeline.
# MAGIC
# MAGIC CREATE OR REPLACE TABLE sandbox.hfsl_icf.parsed_documents AS
# MAGIC WITH raw_files AS (
# MAGIC   SELECT path, content
# MAGIC   FROM READ_FILES('/Volumes/sandbox/hfsl_icf/icf/*.pdf', format => 'binaryFile')
# MAGIC ),
# MAGIC parsed AS (
# MAGIC   SELECT
# MAGIC     path AS source_path,
# MAGIC     ai_parse_document(content) AS parsed_result
# MAGIC   FROM raw_files
# MAGIC )
# MAGIC SELECT
# MAGIC   source_path,
# MAGIC   parsed_result,
# MAGIC   regexp_extract(source_path, '([^/]+)$', 1) AS file_name,
# MAGIC   concat_ws(
# MAGIC     '\n\n',
# MAGIC     transform(
# MAGIC       filter(
# MAGIC         CAST(parsed_result:document:elements AS ARRAY<STRUCT<type: STRING, content: STRING>>),
# MAGIC         x -> x.type IN ('text', 'title', 'section_header')
# MAGIC       ),
# MAGIC       x -> x.content
# MAGIC     )
# MAGIC   ) AS extracted_text,
# MAGIC   SIZE(CAST(parsed_result:document:pages AS ARRAY<STRUCT<id: INT, image_uri: STRING>>)) AS num_pages,
# MAGIC   current_timestamp() AS parsed_at
# MAGIC   FROM parsed

# COMMAND ----------

# DBTITLE 1,Step 2: Translation Pipeline with MLflow Tracing
# =============================================================================
# STEP 2: Translation Pipeline with Full MLflow Tracing
# =============================================================================
# Each LLM call is wrapped with @mlflow.trace for observability.
# This gives you: latency per step, token usage, full input/output logging,
# and the ability to debug failures at any point in the pipeline.

import json
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType




@mlflow.trace(span_type=SpanType.LLM, name="detect_language")
def detect_language(text: str) -> str:
    """Detect the primary language of text using LLM."""
    prompt = (
        "Identify the primary language of the following text. "
        "Return ONLY the ISO 639-1 language code (e.g., 'fr', 'de', 'ja', 'zh', 'es'). "
        f"If English, return 'en'. Text: {text[:2000]}"
    )
    escaped_prompt = prompt.replace("'", "''")
    result = spark.sql(
        f"SELECT ai_query('{LLM_ENDPOINT_FAST}', '{escaped_prompt}') AS lang"
    ).collect()[0]["lang"]
    return result.strip()


@mlflow.trace(span_type=SpanType.LLM, name="translate_to_english")
def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English."""
    if source_lang.strip() == "en":
        return text
    prompt = (
        "You are a professional translator. Translate the following text to English. "
        "Maintain the original meaning, tone, and structure. "
        "Preserve technical terminology. Only output the translated text.\n\n"
        f"Source language: {source_lang}\n\nText:\n{text}"
    )
    escaped_prompt = prompt.replace("'", "''")
    result = spark.sql(
        f"SELECT ai_query('{LLM_ENDPOINT}', '{escaped_prompt}') AS translation"
    ).collect()[0]["translation"]
    return result


@mlflow.trace(span_type=SpanType.LLM, name="back_translate")
def back_translate(english_text: str, target_lang: str) -> str:
    """Back-translate English text to the original language."""
    if target_lang.strip() == "en":
        return english_text
    prompt = (
        f"You are a professional translator. Translate to {target_lang}. "
        "Maintain meaning, tone, and structure. Only output the translated text.\n\n"
        f"Text:\n{english_text}"
    )
    escaped_prompt = prompt.replace("'", "''")
    result = spark.sql(
        f"SELECT ai_query('{LLM_ENDPOINT}', '{escaped_prompt}') AS back_translation"
    ).collect()[0]["back_translation"]
    return result


@mlflow.trace(span_type=SpanType.CHAIN, name="translation_pipeline")
def run_translation_pipeline(file_name: str, text: str) -> dict:
    """Run the full translation pipeline for a single document.
    
    Each sub-step is individually traced, creating a rich span tree:
    translation_pipeline
      ├─ detect_language
      ├─ translate_to_english
      └─ back_translate
    """
    # Tag trace with document metadata
    mlflow.update_current_trace(
        metadata={"file_name": file_name, "pipeline_version": "v2_mlflow"}
    )
    
    lang = detect_language(text)
    english = translate_to_english(text, lang)
    back_trans = back_translate(english, lang)
    
    return {
        "file_name": file_name,
        "original_text": text,
        "detected_language": lang,
        "english_translation": english,
        "back_translation": back_trans,
    }


print("Translation pipeline functions defined with MLflow tracing.")
print("Each call generates a trace visible in the MLflow Experiment UI.")

# COMMAND ----------

# DBTITLE 1,Test: Validate detect_language call
# One-time validation of detect_language
test_result = detect_language("Bonjour le monde, ceci est un test de détection de langue.")
print(f"Detected language: '{test_result}'")
assert test_result.strip() in ("fr", "fra"), f"Expected 'fr', got '{test_result}'"

# COMMAND ----------


display(spark.sql(f"""
SELECT
  file_name,
  extracted_text,
  ai_query(
    '{LLM_ENDPOINT}',
    CONCAT(
      'You are a professional translator. Translate the following text to English. ',
      'Maintain the original meaning, tone, and structure. ',
      'Preserve technical terminology. Only output the translated text.\n\nText:\n',
      extracted_text
    )
  ) AS english_translation
FROM  parsed_documents
WHERE extracted_text IS NOT NULL LIMIT 1"""))


# COMMAND ----------

# DBTITLE 1,Step 3: Execute Translation Pipeline
# =============================================================================
# STEP 3: Batch Translation via SQL (Spark-Parallel)
# =============================================================================
# Instead of row-by-row Python calls, run ALL ai_query() calls as SQL column
# expressions. Spark parallelizes them internally via adaptive thread pools.
#
# Reference baseline (row-by-row approach):
#   AR SI-IC main v 1.0 ar es.pdf (es): 296s
#   BE SI-IC Main SIIC DU V6.0.pdf (nl): 333s
#   Total: ~629s for 2 documents (sequential)

import time

start_time = time.time()

# --- Phase 1: Detect language + forward translate (parallel across all docs) ---
spark.sql(f"""
CREATE OR REPLACE TABLE {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translations_batch AS
SELECT
  file_name,
  extracted_text,
  TRIM(ai_query(
    '{LLM_ENDPOINT_FAST}',
    CONCAT(
      'Identify the primary language of the following text. ',
      'Return ONLY the ISO 639-1 language code (e.g., fr, de, ja, zh, es). ',
      'If English, return en. Text: ',
      LEFT(extracted_text, 2000)
    )
  )) AS detected_language,
  ai_query(
    '{LLM_ENDPOINT}',
    CONCAT(
      'You are a professional translator. Translate the following text to English. ',
      'Maintain the original meaning, tone, and structure. ',
      'Preserve technical terminology. Only output the translated text.\n\nText:\n',
      extracted_text
    )
  ) AS english_translation
FROM {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.parsed_documents
WHERE extracted_text IS NOT NULL
""")

phase1_time = time.time() - start_time
print(f"Phase 1 (detect + translate): {phase1_time:.1f}s")


# COMMAND ----------


# --- Phase 2: Back-translate (parallel, skip English docs) ---
spark.sql(f"""
CREATE OR REPLACE TABLE {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translations_with_backtrans AS
SELECT
  *,
  CASE
    WHEN TRIM(detected_language) = 'en' THEN english_translation
    ELSE ai_query(
      '{LLM_ENDPOINT}',
      CONCAT(
        'You are a professional translator. Translate the following English text to ',
        detected_language,
        '. Maintain meaning, tone, and structure. Only output the translated text.\n\nText:\n',
        english_translation
      )
    )
  END AS back_translation
FROM {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translations_batch
""")

total_time = time.time() - start_time
print(f"Phase 2 (back-translate): {total_time - phase1_time:.1f}s")
print(f"\nTotal batch pipeline: {total_time:.1f}s")
print(f"Baseline (row-by-row): ~629s")
print(f"Speedup: {629 / max(total_time, 1):.1f}x")

# Preview results (stay in Spark — no pandas conversion)
results_table = f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translations_with_backtrans"
print(f"\nResults persisted to: {results_table}")
display(spark.table(results_table).select("file_name", "detected_language"))

# COMMAND ----------

# DBTITLE 1,Step 4: MLflow Evaluate - LLM as a Judge (Custom Scorers)
# =============================================================================
# STEP 4: Import Scorers from scorers.py
# =============================================================================
# Scorers are defined in a separate module for reuse across notebooks,
# jobs, and production monitoring without re-executing this notebook.
#
# Source: /Workspace/Users/hfsl@novonordisk.com/dbx-translations/scorers.py

import importlib.util
import sys

# Load scorers module from workspace file path
_scorers_path = "/Workspace/Users/hfsl@novonordisk.com/dbx-translations/scorers.py"
_spec = importlib.util.spec_from_file_location("scorers", _scorers_path)
scorers = importlib.util.module_from_spec(_spec)
sys.modules["scorers"] = scorers
_spec.loader.exec_module(scorers)

# Pass the configured endpoint to the scorers module
scorers.LLM_ENDPOINT = LLM_ENDPOINT

from scorers import semantic_preservation, completeness_check, translation_guidelines

print("Scorers imported from scorers.py:")
print("  1. semantic_preservation - LLM judges original vs back-translation")
print("  2. completeness_check - Heuristic length-based omission detection")
print("  3. translation_guidelines - Guidelines-based quality assessment")
print(f"  LLM endpoint for judge: {scorers.LLM_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Test: Validate semantic_preservation scorer
# One-time validation of semantic_preservation scorer
test_inputs = {
    "original_text": "Bonjour le monde. Ceci est un document de test pour la traduction.",
    "detected_language": "fr",
    "file_name": "test.pdf",
}
test_outputs = {
    "english_translation": "Hello world. This is a test document for translation.",
    "back_translation": "Bonjour le monde. Ceci est un document de test pour la traduction.",
}

result = semantic_preservation(test_inputs, test_outputs)
print(f"Passed: {result.value}")
print(f"Rationale: {result.rationale}")

# COMMAND ----------

# DBTITLE 1,Step 5: Run MLflow Evaluation
# =============================================================================
# STEP 5: Run MLflow Evaluation
# =============================================================================
# Execute the evaluation using mlflow.genai.evaluate()
# This produces a structured evaluation run with per-row scores and rationales.

from pyspark.sql import functions as F

# Build eval data directly from Spark (no pandas needed)
# Only collect the columns required for scoring, truncated at source
results_table = f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translations_with_backtrans"
rows = (
    spark.table(results_table)
    .select(
        "file_name",
        F.substring("extracted_text", 1, 5000).alias("original_text"),
        "detected_language",
        F.substring("english_translation", 1, 5000).alias("english_translation"),
        F.substring("back_translation", 1, 5000).alias("back_translation"),
    )
    .collect()
)

eval_data = [
    {
        "inputs": {
            "original_text": r["original_text"],
            "detected_language": r["detected_language"],
            "file_name": r["file_name"],
        },
        "outputs": {
            "english_translation": r["english_translation"],
            "back_translation": r["back_translation"],
        },
    }
    for r in rows
]

# Run evaluation with all scorers
with mlflow.start_run(run_name="translation_quality_eval") as run:
    mlflow.log_params({
        "llm_endpoint": LLM_ENDPOINT,
        "num_documents": len(eval_data),
        "pipeline_version": "v2_mlflow",
    })
    
    eval_result = mlflow.genai.evaluate(
        data=eval_data,
        scorers=[
            semantic_preservation,
            completeness_check,
            translation_guidelines,
        ],
    )

print(f"Evaluation complete. Run ID: {run.info.run_id}")
print(f"View results in MLflow Experiment: {EXPERIMENT_PATH}")
print("\n--- Aggregate Metrics ---")
display(eval_result.metrics)

# COMMAND ----------

# DBTITLE 1,Step 6: Analyze Evaluation Results
# =============================================================================
# STEP 6: Analyze Evaluation Results & Generate Final Output
# =============================================================================
# The evaluation results table has per-row scores from each scorer.
# Use this to flag documents for human review.

# Get detailed per-row results
eval_table = eval_result.result_df

# Extract inputs from the request dict column
eval_table["file_name"] = eval_table["request"].apply(lambda r: r.get("file_name", "") if isinstance(r, dict) else "")
eval_table["detected_language"] = eval_table["request"].apply(lambda r: r.get("detected_language", "") if isinstance(r, dict) else "")

# Add status flags based on scorer results
def determine_status(row):
    """Map scorer results to actionable statuses."""
    if row.get("detected_language", "").strip() == "en":
        return "SKIPPED"
    
    semantic_pass = row.get("semantic_preservation/value", False)
    completeness_pass = row.get("completeness_check/value", False)
    
    if semantic_pass and completeness_pass:
        return "APPROVED"
    elif not semantic_pass and not completeness_pass:
        return "REJECTED"
    else:
        return "NEEDS_REVIEW"

eval_table["status"] = eval_table.apply(determine_status, axis=1)

# Summary
print("Translation Quality Summary:")
print(eval_table["status"].value_counts().to_string())
print("\n--- Documents Needing Review ---")
review_docs = eval_table[eval_table["status"].isin(["REJECTED", "NEEDS_REVIEW"])]
if len(review_docs) > 0:
    display(review_docs[["file_name", "status", 
                         "semantic_preservation/rationale",
                         "completeness_check/rationale"]].head(10))
else:
    print("All documents passed quality checks!")

# COMMAND ----------

# DBTITLE 1,Step 7: Persist Final Results to Delta
# =============================================================================
# STEP 7: Persist Final Results to Delta Table
# =============================================================================
# Save the evaluated translations with quality metadata to a Delta table.

final_df = spark.table(f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translations_with_backtrans").toPandas()
final_df["status"] = eval_table["status"].values
final_df["semantic_preservation_rationale"] = eval_table["semantic_preservation/rationale"].values
final_df["completeness_rationale"] = eval_table["completeness_check/rationale"].values

# Write to Delta
final_spark_df = spark.createDataFrame(final_df)
final_spark_df.write.mode("overwrite").saveAsTable(
    f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translation_final_v2"
)

print(f"Results saved to {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translation_final_v2")
display(
    spark.table(f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translation_final_v2")
    .select("file_name", "detected_language", "status", "semantic_preservation_rationale")
)

# COMMAND ----------

# DBTITLE 1,Step 8: Create Evaluation Dataset for Reproducibility
# =============================================================================
# STEP 8: Create Versioned Evaluation Dataset
# =============================================================================
# MLflow evaluation datasets allow you to:
# - Reproduce evaluations exactly
# - Track quality over time as models/prompts change
# - Share test cases across team members
# - Add edge cases as they're discovered

from pyspark.sql import functions as F
from mlflow.genai.datasets import create_dataset, get_dataset

# Create or load persistent evaluation dataset
try:
    dataset = create_dataset(
        uc_table_name=f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translation_eval_dataset",
    )
except Exception:
    dataset = get_dataset(uc_table_name=f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translation_eval_dataset")

# Populate with current results as baseline test cases
rows = (
    spark.table(f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translations_with_backtrans")
    .select("file_name", F.substring("extracted_text", 1, 3000).alias("original_text"), "detected_language")
    .collect()
)

test_cases = []
for row in rows:
    test_cases.append({
        "inputs": {
            "original_text": row["original_text"],
            "detected_language": row["detected_language"],
            "file_name": row["file_name"],
        },
        "expectations": {
            "expected_language": row["detected_language"],
            "should_translate": row["detected_language"].strip() != "en",
        },
    })

dataset.merge_records(test_cases)

print(f"Evaluation dataset created with {len(test_cases)} test cases.")
print("Use this dataset to benchmark future model/prompt changes.")
print(f"Table: {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.translation_eval_dataset")

# COMMAND ----------

# DBTITLE 1,Step 9: Register Scorers for Production Monitoring
# =============================================================================
# STEP 9: Production Monitoring with Registered Scorers
# =============================================================================
# Register scorers so they automatically evaluate new traces at a sample rate.
# This provides continuous quality monitoring without manual intervention.
#
# Once registered, any new trace logged to this experiment will be
# automatically scored at the configured sample rate.

from mlflow.genai.scorers import ScorerSamplingConfig

# Register the semantic preservation scorer for production monitoring
registered_semantic = semantic_preservation.register(name="translation_semantic_preservation")
registered_semantic.start(
    sampling_config=ScorerSamplingConfig(sample_rate=1.0)  # Score 100% of traces
)

# Register the completeness scorer
registered_completeness = completeness_check.register(name="translation_completeness")
registered_completeness.start(
    sampling_config=ScorerSamplingConfig(sample_rate=1.0)
)

# Register guidelines scorer at lower sample rate (it's more expensive)
registered_guidelines = translation_guidelines.register(name="translation_guidelines_monitor")
registered_guidelines.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5)  # Score 50% of traces
)

print("Production monitoring scorers registered and started:")
print("  - translation_semantic_preservation (100% sample rate)")
print("  - translation_completeness (100% sample rate)")
print("  - translation_guidelines_monitor (50% sample rate)")
print("\nAll future traces in this experiment will be automatically evaluated.")

# COMMAND ----------

# DBTITLE 1,Step 10: Model Comparison with Timing and Token Metrics
# =============================================================================
# STEP 10: Model Comparison with Timing & Token Metrics
# =============================================================================
# Compare translation quality AND performance across LLM endpoints.
# Logs wall-clock time and estimated token usage per model to MLflow.

import time
from pyspark.sql import functions as F


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for Latin scripts."""
    if not text:
        return 0
    return len(text) // 4


def run_model_comparison(endpoints: list[str], num_docs: int = 3):
    """Run translation + evaluation across multiple LLM endpoints.
    
    For each endpoint, logs to MLflow:
    - translation_duration_s: total wall-clock time
    - avg_duration_per_doc_s: per-document average
    - estimated_input_tokens / estimated_output_tokens / estimated_total_tokens
    - tokens_per_second: throughput estimate
    - Quality scores from completeness_check and translation_guidelines
    """
    sample_rows = (
        spark.table(f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.parsed_documents")
        .where("extracted_text IS NOT NULL")
        .limit(num_docs)
        .select("file_name", "extracted_text")
        .collect()
    )
    print(f"Comparing {len(endpoints)} endpoints on {len(sample_rows)} documents...\n")
    
    for endpoint in endpoints:
        print(f"  Running: {endpoint}")
        
        with mlflow.start_run(run_name=f"comparison_{endpoint}"):
            mlflow.log_params({
                "llm_endpoint": endpoint,
                "num_documents": len(sample_rows),
                "comparison_type": "model_ab_test",
            })
            
            # --- Measure translation time (batch SQL) ---
            t_start = time.time()
            
            translation_df = spark.sql(f"""
                SELECT
                  file_name,
                  extracted_text,
                  ai_query(
                    '{endpoint}',
                    CONCAT(
                      'You are a professional translator. Translate the following text to English. ',
                      'Maintain the original meaning, tone, and structure. ',
                      'Preserve technical terminology. Only output the translated text.\n\nText:\n',
                      extracted_text
                    )
                  ) AS english_translation
                FROM {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.parsed_documents
                WHERE extracted_text IS NOT NULL
                LIMIT {num_docs}
            """)
            results = translation_df.collect()
            
            translation_duration = time.time() - t_start
            
            # --- Estimate token usage ---
            total_input_tokens = 0
            total_output_tokens = 0
            for row in results:
                # Input: system prompt (~50 tokens) + source text
                total_input_tokens += 50 + estimate_tokens(row["extracted_text"])
                # Output: translated text
                total_output_tokens += estimate_tokens(row["english_translation"] or "")
            
            total_tokens = total_input_tokens + total_output_tokens
            
            # --- Log timing and token metrics ---
            mlflow.log_metrics({
                "translation_duration_s": round(translation_duration, 1),
                "avg_duration_per_doc_s": round(translation_duration / max(len(results), 1), 1),
                "estimated_input_tokens": total_input_tokens,
                "estimated_output_tokens": total_output_tokens,
                "estimated_total_tokens": total_tokens,
                "tokens_per_second": round(total_tokens / max(translation_duration, 0.1), 0),
            })
            
            # --- Run quality evaluation ---
            eval_data = [
                {
                    "inputs": {
                        "original_text": row["extracted_text"][:3000],
                        "detected_language": "unknown",
                        "file_name": row["file_name"],
                    },
                    "outputs": {
                        "english_translation": (row["english_translation"] or "")[:3000],
                        "back_translation": "",
                    },
                }
                for row in results
            ]
            
            mlflow.genai.evaluate(
                data=eval_data,
                scorers=[completeness_check, translation_guidelines],
            )
        
        print(f"    Duration: {translation_duration:.1f}s | "
              f"Tokens: ~{total_tokens:,} | "
              f"Throughput: ~{total_tokens / max(translation_duration, 0.1):.0f} tok/s")
    
    print("\nComparison complete. View side-by-side in MLflow Experiment UI.")


# --- Run comparison: 3 providers, cheap -> mid -> premium ---
run_model_comparison(COMPARISON_ENDPOINTS, num_docs=2)

# COMMAND ----------

# DBTITLE 1,Deployment Summary
# MAGIC %md
# MAGIC ## Deployment & Next Steps
# MAGIC
# MAGIC ### What This Notebook Adds Over the Original
# MAGIC
# MAGIC | Feature | Benefit |
# MAGIC | --- | --- |
# MAGIC | **MLflow Tracing** (`@mlflow.trace`) | Full observability: latency, tokens, errors per LLM call |
# MAGIC | **`mlflow.genai.evaluate()`** | Structured evaluation with typed scorers, rationale, and metrics |
# MAGIC | **Custom Scorers** | Reusable, versioned judge logic (not brittle prompt strings) |
# MAGIC | **Binary Pass/Fail** | More calibrated than 1-10 numeric scales |
# MAGIC | **Evaluation Datasets** | Reproducible benchmarks, versioned in Unity Catalog |
# MAGIC | **Production Monitoring** | Auto-score new traces without manual intervention |
# MAGIC | **Model Comparison** | A/B test LLM endpoints with identical evaluation criteria |
# MAGIC
# MAGIC ### Production Deployment Options
# MAGIC
# MAGIC 1. **Lakeflow Spark Declarative Pipeline** — Use SQL-based pipeline for the translation steps, with this notebook as a post-processing quality gate
# MAGIC 2. **Scheduled Job** — Run this notebook on a schedule; MLflow monitoring catches quality regressions automatically
# MAGIC 3. **Hybrid** — SDP for ingestion/translation, MLflow evaluate as a downstream quality check
# MAGIC
# MAGIC ### Monitoring in Production
# MAGIC
# MAGIC Once scorers are registered (Step 9), you can:
# MAGIC - View quality trends in the MLflow Experiment UI
# MAGIC - Set alerts on scorer pass rates dropping below thresholds
# MAGIC - Compare quality across model versions when upgrading endpoints
# MAGIC - Use `mlflow.genai.datasets.create_dataset()` to accumulate edge cases over time
