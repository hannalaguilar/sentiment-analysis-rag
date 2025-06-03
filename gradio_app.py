import re
import os
import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/classify")
print(f"Using API_URL: {API_URL}")


def strip_html(text: str):
    return re.sub(r'<[^>]+>', '', text)


def gradio_predict(review_text: str):
    try:
        response = requests.post(API_URL, json={"review_text": review_text})
        response.raise_for_status()
        result = response.json()

        sentiment = result["sentiment"]
        similar = "\n\n".join(
            [f"{i+1}. ({r['cosine_score']:.2f}) {r['sentiment'].capitalize()}:\n{strip_html(r['review'])}"
             for i, r in enumerate(result["similar_reviews"])]
        )
        return sentiment.capitalize(), similar

    except Exception as e:
        return "Error", str(e)

with gr.Blocks() as demo:
    gr.Markdown("## Sentiment Classifier using Gemma3")
    with gr.Row():
        with gr.Column():
            review_input = gr.Textbox(label="Write your movie review", lines=6, placeholder="e.g. I loved the film...")
            submit_btn = gr.Button("Classify")
        with gr.Column():
            sentiment_output = gr.Textbox(label="Predicted Sentiment")
            similar_output = gr.Textbox(label="Similar Reviews", lines=10)

    submit_btn.click(fn=gradio_predict, inputs=[review_input], outputs=[sentiment_output, similar_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

