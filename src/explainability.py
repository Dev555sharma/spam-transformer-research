import shap
import torch
from transformers import BertTokenizer, BertForSequenceClassification


def load_model():
    model = BertForSequenceClassification.from_pretrained("models/saved_model")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


def predict(texts, model, tokenizer):
    # Ensure input is always a list of strings
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        texts = list(texts)

    texts = [str(t) for t in texts]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    return probs.detach().cpu().numpy()


def run_explanation():
    model, tokenizer = load_model()

    explainer = shap.Explainer(
        lambda x: predict(x, model, tokenizer),
        masker=shap.maskers.Text(tokenizer)
    )

    texts = [
        "Congratulations! You won a free prize",
        "Hey, are we meeting tomorrow?"
    ]

    shap_values = explainer(texts)

    shap.plots.text(shap_values)


if __name__ == "__main__":
    run_explanation()
