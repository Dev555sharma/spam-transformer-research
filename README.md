# Robust Spam Detection using Transformer Models

## Overview
This project explores the robustness of transformer-based models for spam detection under adversarial perturbations. Unlike traditional approaches, this work evaluates how model performance degrades when input text is intentionally modified.

## Objectives
- Build a spam classifier using BERT
- Evaluate performance on clean vs noisy data
- Analyze robustness under adversarial attacks
- Interpret model predictions using SHAP

## Methodology
- Dataset: SMS Spam Collection
- Model: BERT (bert-base-uncased)
- Tokenization using HuggingFace Transformers
- Custom PyTorch Dataset pipeline

## Adversarial Strategy
We simulate real-world spam obfuscation:
- Character replacement (e → 3, o → 0)
- Random symbol insertion

## 📊 Results

| Metric            | Clean Data | Noisy Data |
|------------------|-----------|------------|
| Accuracy         | 98.9%     | 35.9%      |
| F1 Score         | 98.1%     | 46.4%      |
| Precision        | 97.6%     | 30.2%      |
| Recall           | 98.5%     | 100%       |

## 🚨 Key Finding
Model performance drops by **~63% under adversarial noise**, showing that transformer models are highly sensitive to input perturbations.

## 🧠 Insight
Under noisy conditions, the model becomes biased toward predicting spam, leading to high recall but very low precision.

## Key Insight
Transformer models perform well on clean data but show measurable degradation under adversarial perturbations, highlighting robustness limitations in real-world deployment.

## Explainability
We use SHAP to:
- Identify influential words
- Understand model decision patterns

## Project Structure
- src/ → core pipeline
- data/ → dataset
- models/ → trained model
- results/ → outputs

## Future Work
- Adversarial training
- Multi-lingual spam detection
- LLM-based classification comparison
