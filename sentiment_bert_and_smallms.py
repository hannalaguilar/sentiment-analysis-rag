import pandas as pd
import numpy as np
import torch
from datasets import load_dataset

# Check CUDA
print(f'CUDA: {torch.cuda.is_available()}')

# Load IMDB dataset
dataset = load_dataset("imdb", download_mode="force_redownload")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Sample subset for quicker development
train_df = train_df.sample(n=5000, random_state=42)
test_df = test_df.sample(n=10, random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Display sample data
print("\nSample review:")
sample = train_df.iloc[0]
print(f"Text: {sample['text'][:200]}...")
print(f"Sentiment: {'Positive' if sample['label'] == 1 else 'Negative'}")

# EDA
print('\nLanguage: English\n')

# Positive/Negative distribution
train_labels_dist = train_df['label'].value_counts(normalize=True)
print(f'Train distribution: Negative sample: {train_labels_dist[0]*100}%, Positive:{train_labels_dist[1]*100}%')
test_labels_dist = test_df['label'].value_counts(normalize=True)
print(f'Test distribution: Negative sample: {test_labels_dist[0]*100}%, Positive:{test_labels_dist[1]*100}%')

print('')

# Average length text
train_length_text = train_df['text'].apply(lambda x:len(x)).tolist()
print(f'Average text length in the training set: {np.mean(train_length_text):.0f} +/- {np.std(train_length_text):.0f}')
test_length_text = test_df['text'].apply(lambda x:len(x)).tolist()
print(f'Average text length in the test set: {np.mean(test_length_text):.0f} +/- {np.std(test_length_text):.0f}')

# Task 1: Model Implementation
import time
TARGET_NAMES = ['negative', 'positive']

i = 1
review = test_df['text'].iloc[i]
gt_label = test_df['label'].iloc[i]


def run_example(inference_func, model_name: str):
    print(f'\nRunning example for {model_name}...')
    t0 = time.time()
    prediction = inference_func(review)
    elapsed_time = time.time() - t0
    print(f'Prediction: {prediction}')
    print(f'Inference time per sample: {elapsed_time:.4f} seconds')
    print(f'True label: {map_label2str(gt_label)}')


def map_label2int(label: str) -> int:
  return 1 if label == 'positive' else 0


def map_label2str(label: int) -> str:
  return 'positive' if label == 1 else 'negative'


# Approach 1
from typing import Dict, Union
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_distilbert = DistilBertTokenizer.from_pretrained(model_id)
model_distilbert = DistilBertForSequenceClassification.from_pretrained(model_id)


def classify_review_distilbert(review: str) -> Dict[str, Union[str, float]]:
    # Tokenizer
    inputs = tokenizer_distilbert(review, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model_distilbert(**inputs).logits

    # Probabilities
    probs = F.softmax(logits, dim=-1)

    # Preditected class and its confidence
    predicted_class_id = probs.argmax().item()

    return {
        'sentiment': model_distilbert.config.id2label[predicted_class_id].lower(),
        'predited_class': predicted_class_id,
        'confidence': probs[0, predicted_class_id].item(),
    }


# Example
# run_example(classify_review_distilbert, 'distilbert-base-uncased')

# Approach 2: Small LLMs (1.7B parameters) + Few-shot prompting


def build_prompt(review: str) -> str:
    return f"""You are a sentiment classifier.
Classify each movie review as either positive or negative.
Return only the label, without any additional text.

Review: "This movie was terrible, slow and boring."
Sentiment: negative

Review: "Absolutely amazing. One of the best films I've seen."
Sentiment: positive

Review: "{review}"
Sentiment:"""

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
model_smollm2 = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
tokenizer_smollm2 = AutoTokenizer.from_pretrained(model_id)

def classify_review_smollm2(review: str) -> str:
  prompt = build_prompt(review)
  inputs = tokenizer_smollm2(prompt, return_tensors='pt', truncation=True).to(model_smollm2.device)
  outputs = model_smollm2.generate(**inputs, max_new_tokens=3)
  output_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
  response_text = tokenizer_smollm2.decode(output_tokens, skip_special_tokens=True).strip()
  return response_text.lower()


# Example
# run_example(classify_review_smollm2, 'SmolLM2-1.7B-Instruct')

