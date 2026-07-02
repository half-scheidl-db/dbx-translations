# Design Decisions: Translation Workflow with MLflow

## Decision: Batch SQL over Row-by-Row Python for Translation

### Problem

The initial implementation converted the parsed documents table to pandas, then
iterated row-by-row calling `ai_query()` via individual `spark.sql()` invocations:

```python
# ANTI-PATTERN: Sequential row-by-row calls (slow)
docs_df = spark.table("sandbox.hfsl_icf.parsed_documents").toPandas()

results = []
for _, row in docs_df.iterrows():
    result = run_translation_pipeline(
        file_name=row["file_name"],
        text=row["extracted_text"]
    )
    results.append(result)
```

**Measured baseline (2 documents):**

| Document | Duration |
| --- | --- |
| AR SI-IC main v 1.0 ar es.pdf (lang: es) | 296s |
| BE SI-IC Main SIIC DU V6.0.pdf (lang: nl) | 333s |

This approach is inherently slow because:
1. Each `spark.sql()` call has round-trip overhead (plan, execute, collect)
2. Documents are processed sequentially — no parallelism
3. Each document requires 3 sequential LLM calls (detect, translate, back-translate)
4. Total: ~6 individual LLM calls serialized for 2 documents

### Solution: Batch SQL with Spark Parallelism

Run all `ai_query()` calls as SQL column expressions across the full table.
Spark parallelizes the calls internally, and there is no Python-loop overhead:

```sql
-- STEP 1: Detect language + translate in one pass
CREATE OR REPLACE TABLE sandbox.hfsl_icf.translations_batch AS
SELECT
  file_name,
  extracted_text,
  ai_query(
    'databricks-claude-sonnet-4',
    CONCAT(
      'Identify the primary language of the following text. ',
      'Return ONLY the ISO 639-1 language code. Text: ',
      LEFT(extracted_text, 2000)
    )
  ) AS detected_language,
  ai_query(
    'databricks-claude-sonnet-4',
    CONCAT(
      'You are a professional translator. Translate the following text to English. ',
      'Maintain the original meaning, tone, and structure. ',
      'Preserve technical terminology. Only output the translated text.\n\nText:\n',
      extracted_text
    )
  ) AS english_translation
FROM sandbox.hfsl_icf.parsed_documents
WHERE extracted_text IS NOT NULL;
```

```sql
-- STEP 2: Back-translate (depends on step 1 results)
CREATE OR REPLACE TABLE sandbox.hfsl_icf.translations_with_backtrans AS
SELECT
  *,
  CASE
    WHEN TRIM(detected_language) = 'en' THEN english_translation
    ELSE ai_query(
      'databricks-claude-sonnet-4',
      CONCAT(
        'You are a professional translator. Translate the following English text to ',
        detected_language,
        '. Maintain meaning, tone, and structure. Only output the translated text.\n\nText:\n',
        english_translation
      )
    )
  END AS back_translation
FROM sandbox.hfsl_icf.translations_batch;
```

### MLflow Integration: Post-Hoc Evaluation

MLflow `genai.evaluate()` is **decoupled from how data was produced**. We can:
1. Run translations in fast batch SQL (no tracing overhead)
2. Feed materialized results into `mlflow.genai.evaluate()` afterward
3. Still get full scorer results, rationales, and experiment tracking

```python
import mlflow
import pandas as pd

# Load batch results (already computed via fast SQL)
results_df = spark.table("sandbox.hfsl_icf.translations_with_backtrans").toPandas()

# Build eval data from batch results
eval_data = [
    {
        "inputs": {
            "original_text": row["extracted_text"][:5000],
            "detected_language": row["detected_language"],
            "file_name": row["file_name"],
        },
        "outputs": {
            "english_translation": row["english_translation"][:5000],
            "back_translation": row["back_translation"][:5000],
        },
    }
    for _, row in results_df.iterrows()
]

# Run MLflow evaluation (scorers still work identically)
with mlflow.start_run(run_name="translation_quality_eval"):
    eval_result = mlflow.genai.evaluate(
        data=eval_data,
        scorers=[semantic_preservation, completeness_check, translation_guidelines],
    )
```

### Tracing Tradeoff

| Approach | Speed | Observability |
| --- | --- | --- |
| Row-by-row + `@mlflow.trace` | ~300s/doc | Per-call span tree (detect → translate → back-translate) |
| Batch SQL + post-hoc evaluate | Spark-parallel | Batch-level timing + per-row scorer results |

**Recommendation:** Use batch SQL for production. Use row-by-row tracing only
during development/debugging to diagnose individual document failures.

### `ai_query()` Usage Notes (Databricks)

- **No `responseFormat` for Claude models**: The `databricks-claude-sonnet-4`
  endpoint does not support `responseFormat`. Omit it — `ai_query()` returns
  STRING by default.
- **No `$` dollar-quoting in `spark.sql()`**: Dollar-quoting is only available
  in native SQL cells. From Python, escape single quotes with `replace("'", "''")`.
- **Batch is cheaper**: Spark's internal batching of `ai_query()` uses adaptive
  thread pools — a single SQL statement with N rows can issue N concurrent LLM
  calls (up to the endpoint's rate limit).
