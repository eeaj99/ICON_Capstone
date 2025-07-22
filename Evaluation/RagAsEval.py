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

os.environ["OPENAI_API_KEY"] = "sk......"  # Replace with your actual API key

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

print("\n RAGAS Evaluation Results:")
for i, score_dict in enumerate(result.scores):
    print(f"\nSample {i+1}")
    for metric, value in score_dict.items():
        print(f"{metric}: {value}")




