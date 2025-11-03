# create_sample_dataset.py

import os
import json

def create_sample_dataset(save_path="data/sample_rubric_pairs.jsonl"):
    """
    Creates a sample JSONL dataset for demo LoRA training, including examples 
    with high, medium, and low scores.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    examples = [
        # HIGH SCORE EXAMPLE (Score 9)
        {
            "question": "Explain the difference between precision and recall.",
            "answer": "Precision = TP/(TP+FP). Recall = TP/(TP+FN). Precision measures correctness of positive predictions; recall measures coverage of actual positives.",
            "score": 9,
            "feedback": "Good definitions and formulas. Could add a short example."
        },
        # MEDIUM SCORE EXAMPLE (Score 8)
        {
            "question": "What is overfitting and how do you prevent it?",
            "answer": "Overfitting is when a model performs well on training but poorly on unseen data. Prevent via regularization, validation, dropout, and simpler model or more data.",
            "score": 8,
            "feedback": "Clear; could mention cross-validation and early stopping."
        },
        # MEDIUM-LOW SCORE EXAMPLE (Score 4)
        {
            "question": "Describe the concept of an activation function in neural networks.",
            "answer": "An activation function is something that activates a neuron, like a switch. It just makes the network work and is required in every layer.",
            "score": 4,
            "feedback": "Partially correct, but lacks technical depth and precision. Does not mention non-linearity or popular types like ReLU."
        },
        # LOW SCORE EXAMPLE (Score 2)
        {
            "question": "How do you handle missing values in a dataset?",
            "answer": "I would just delete the rows that have missing values, or maybe replace them with zero. It depends on the size of the dataset.",
            "score": 2,
            "feedback": "Deleting rows is often a poor choice. Replacing with zero or mean/median are better methods, but the answer shows a lack of awareness of imputation techniques."
        },
        # HIGH SCORE EXAMPLE (Score 10)
        {
            "question": "Explain the bias-variance trade-off.",
            "answer": "Bias is the error from overly simple assumptions; high bias means underfitting. Variance is the error from excessive model sensitivity to training data; high variance means overfitting. The trade-off means reducing one usually increases the other.",
            "score": 10,
            "feedback": "Perfectly articulated, covering all key components clearly and concisely."
        },
    ]
    
    # Repeat to make a larger dataset (~50 lines now, for better diversity in the demo)
    # The original was 20 lines (2 * 10). Let's aim for 50 lines (5 * 10).
    dataset = examples * 10
    
    with open(save_path, "w", encoding="utf-8") as f:
        for item in dataset:
            # Use json.dumps to ensure the output is a valid JSON string per line
            f.write(json.dumps(item) + "\n")
            
    print(f"✅ Saved {len(dataset)} examples to {save_path}")

if __name__ == "__main__":
    create_sample_dataset()
