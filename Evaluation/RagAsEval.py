import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from collections import defaultdict
import math

#os.environ["OPENAI_API_KEY"] = "" 

# Load JSON with raw string to avoid path escape errors
with open("Evaluation\qa_pairs.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Prepare dataset in RAGAS format
records = []
for item in qa_data:
    records.append({
        "question": item["question"],
        "answer": item["chatbot_answer"],
        "contexts": [item["standard_answer"]],
        "reference": item["standard_answer"],
        "ground_truths": [item["standard_answer"]],
    })

dataset = Dataset.from_list(records)

# Evaluate with RAGAS (returns EvaluationResult)
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]
)
# 🔁 Calculate total and count for each metric
totals = defaultdict(float)
counts = defaultdict(int)

for sample_scores in result.scores:
    for metric, value in sample_scores.items():
        if value is not None and not math.isnan(value):
            totals[metric] += value
            counts[metric] += 1

# ✅ Print average score per metric
print("\n📊 Average RAGAS Scores Across All Questions:")
for metric in totals:
    average = totals[metric] / counts[metric]
    print(f"{metric}: {average:.4f}")

# print("\n RAGAS Evaluation Results:")
# for i, score_dict in enumerate(result.scores):
#     print(f"\nSample {i+1}")
#     for metric, value in score_dict.items():
#         print(f"{metric}: {value}")




