"""Translation quality scorers for MLflow GenAI evaluation.

Usage:
    from scorers import semantic_preservation, completeness_check, translation_guidelines
"""

import json

from mlflow.genai.scorers import scorer, Guidelines
from mlflow.entities import Feedback


# Default endpoint — can be overridden by setting scorers.LLM_ENDPOINT before calling
LLM_ENDPOINT = "databricks-claude-sonnet-4"


@scorer
def semantic_preservation(inputs, outputs) -> Feedback:
    """Judge whether the back-translation preserves the meaning of the original.

    Uses the LLM to compare original vs back-translation (binary pass/fail).
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()

    original = inputs.get("original_text", "")
    back_trans = outputs.get("back_translation", "")
    lang = inputs.get("detected_language", "")

    if lang.strip() == "en":
        return Feedback(value=True, rationale="Document already in English, no translation needed.")

    judge_prompt = (
        "You are a translation quality judge. Compare the original text with its "
        "back-translation (original \u2192 English \u2192 original language).\n\n"
        "Determine if the back-translation preserves the MEANING of the original. "
        "Minor stylistic differences are acceptable. Focus on: semantic accuracy, "
        "completeness (no omissions), and factual correctness.\n\n"
        f"ORIGINAL ({lang}):\n{original[:3000]}\n\n"
        f"BACK-TRANSLATION ({lang}):\n{back_trans[:3000]}\n\n"
        'Respond with JSON: {"passed": true/false, "rationale": "..."}'
    )

    escaped_judge = judge_prompt.replace("'", "''")
    result = spark.sql(
        f"SELECT ai_query('{LLM_ENDPOINT}', '{escaped_judge}') AS judgment"
    ).collect()[0]["judgment"]

    # Strip any markdown fencing the LLM might add
    clean = result.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    judgment = json.loads(clean)
    return Feedback(
        value=judgment.get("passed", False),
        rationale=judgment.get("rationale", "No rationale provided"),
    )


@scorer
def completeness_check(inputs, outputs) -> Feedback:
    """Check that the translation doesn't omit significant content."""
    original = inputs.get("original_text", "")
    translation = outputs.get("english_translation", "")
    lang = inputs.get("detected_language", "")

    if lang.strip() == "en":
        return Feedback(value=True, rationale="Document already in English.")

    orig_len = len(original)
    trans_len = len(translation)
    ratio = trans_len / max(orig_len, 1)

    if ratio < 0.3:
        return Feedback(
            value=False,
            rationale=f"Translation is only {ratio:.0%} the length of original \u2014 likely content omitted.",
        )
    return Feedback(
        value=True,
        rationale=f"Translation length ratio: {ratio:.0%} \u2014 within acceptable range.",
    )


# --- Guidelines Scorer: Translation Quality Standards ---
translation_guidelines = Guidelines(
    name="translation_quality",
    guidelines=[
        "The translation should preserve all factual information from the original.",
        "Technical terminology should be translated accurately or kept in original form.",
        "The translation should maintain the document's logical structure and flow.",
        "No content should be added that was not present in the original.",
    ],
)
