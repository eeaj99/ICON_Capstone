import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
# Load JSON data
with open("Evaluation\qa_pairs.json", "r") as f:
    data = json.load(f)

smooth = SmoothingFunction().method1
bleu_scores = []

# Loop through each QA pair
for entry in data:
    reference = word_tokenize(entry['standard_answer'].lower())
    hypothesis = word_tokenize(entry['chatbot_answer'].lower())

    score = sentence_bleu([reference], hypothesis, smoothing_function=smooth)
    bleu_scores.append(score)

# Average BLEU score
average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu:.4f}")
