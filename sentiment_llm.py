from typing import Dict, Union, List, Tuple
import os
import time
import pandas as pd
import numpy as np
import ollama
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from sklearn.metrics import classification_report


def map_label2str(label: int) -> str:
  return 'positive' if label == 1 else 'negative'


def map_label2int(label: str) -> int:
  return 1 if label == 'positive' else 0


# Generate embeddings RUN ONLY ONCE
# embedder = SentenceTransformer('intfloat/e5-large-v2')
# corpus_embeddings = embedder.encode(train_df['text'].tolist(), normalize_embeddings=True)
# index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
# index.add(corpus_embeddings)
# os.makedirs("rag_index", exist_ok=True)
# np.save("rag_index/embeddings.npy", corpus_embeddings)
# train_df.to_csv("rag_index/metadata.csv", index=False)
# faiss.write_index(index, "rag_index/faiss_index.index")


embedder = SentenceTransformer('intfloat/e5-large-v2')
corpus_embeddings = np.load("rag_index/embeddings.npy")
train_df = pd.read_csv("rag_index/metadata.csv")
index = faiss.read_index("rag_index/faiss_index.index")


def build_prompt_with_faiss(review: str, verbose: bool, k: int = 3, df: pd.DataFrame = train_df)  -> Tuple[str, List]:
    review_embedding = embedder.encode([review], normalize_embeddings=True)
    D, I = index.search(review_embedding, k)

    similar_reviews = []
    examples = ""
    for rank, (cosine_score, idx) in enumerate(zip(D[0], I[0]), start=1):
        row = df.iloc[idx]
        review = row["text"]
        sentiment = map_label2str(row["label"])
        examples += f'Review: "{review}"\nSentiment: {sentiment}\n\n'
        similar_reviews.append({'review': review, 'sentiment': sentiment, 'cosine_score': cosine_score.item()})

        if verbose:
            short_text = row["text"][:200].replace("\n", " ") + ("..." if len(row["text"]) > 200 else "")
            print(f"{rank}. Score: {cosine_score:.4f} | Sentiment: {sentiment}\n{short_text}\n")


    prompt = f"""You are a sentiment classifier.
    Your task is to classify each movie review as either positive or negative.
    Return only the label, without any additional text.

    Follow this reasoning process for each review:
    1. Analyze the tone and opinion expressed in the text.
    2. Identify words or phrases that indicate positive or negative sentiment.
    3. Decide the sentiment label based on the evidence and the examples.

    Use the following examples:

    {examples}
    Now classify the following review:

    Review: "{review}"
    Sentiment:"""
    return prompt, similar_reviews


def classify_rag_faiss(review: str, llm_model: str = 'gemma3', verbose: bool = True) -> Dict[str, Union[str, List]]:
    prompt, similar_reviews = build_prompt_with_faiss(review, verbose=verbose)
    response = ollama.chat(
                            model=llm_model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant for sentiment classification."},
                                {"role": "user", "content": prompt}

                            ],
                            options = {"temperature": 0}
                        )

    answer = response['message']['content'].strip().lower()
    return {"sentiment": answer,
            "similar_reviews": similar_reviews}


if __name__ == '__main__':
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    test_df = pd.DataFrame(dataset['test'])
    test_df = test_df.sample(n=10, random_state=42)

    i = 2
    REVIEW = test_df['text'].iloc[i]
    GT_LABEL = test_df['label'].iloc[i]

    MODEL_NAME = 'Rag-Faiss'
    print(f'\nRunning example for {MODEL_NAME}...')
    t0 = time.time()
    prediction = classify_rag_faiss(REVIEW)
    elapsed_time = time.time() - t0
    print(f'Prediction: {prediction}')
    print(f'Inference time per sample: {elapsed_time:.4f} seconds')
    print(f'True label: {map_label2str(GT_LABEL)}')



    # TARGET_NAMES = ['negative', 'positive']
    # print(f'Test results using {MODEL_NAME}\n')
    # t0 = time.time()
    # output = [classify_rag_faiss(review) for review in test_df['text']]
    # elapsed_time = time.time() - t0
    # average_time = elapsed_time / test_df.shape[0]
    # print(f'Average inference time per sample: {average_time:.4f} seconds\n')
    # print(classification_report(test_df['label'].values, [map_label2int(result['sentiment']) for result in output], target_names=TARGET_NAMES))