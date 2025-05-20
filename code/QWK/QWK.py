import pandas as pd
import numpy as np

# Fill in the MLLM output data path
data = pd.read_csv(r'')

# List of traits to evaluate
traits = [
    "argument_clarity", "justifying_persuasiveness", "organizational_structure", "coherence", "essay_length",
    "grammatical_accuracy", "grammatical_diversity", "lexical_accuracy", "lexical_diversity", "punctuation_accuracy"
]

# Define a function to calculate QWK
def quadratic_weighted_kappa_float(y_true, y_pred, min_rating=0, max_rating=5, step=0.5):
    # Calculate the number of rating steps (e.g., 0 to 5 with step 0.5 gives 11 levels)
    scale_factor = int(1 / step)  # e.g., step 0.5 -> multiply scores by 2

    # Scale scores to integers
    y_true_scaled = np.round(np.array(y_true) / step).astype(int)
    y_pred_scaled = np.round(np.array(y_pred) / step).astype(int)

    # Ensure scores are within [0, 10]
    y_true_scaled = np.clip(y_true_scaled, 0, 10)
    y_pred_scaled = np.clip(y_pred_scaled, 0, 10)

    # Create histograms for 11 rating values (0 to 10)
    num_ratings = 11
    hist_true = np.zeros(num_ratings)
    hist_pred = np.zeros(num_ratings)

    # Count occurrences of each score
    for a in y_true_scaled:
        hist_true[a] += 1
    for a in y_pred_scaled:
        hist_pred[a] += 1

    # Calculate the weight matrix
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i][j] = float((i - j) ** 2) / (num_ratings - 1) ** 2

    # Build confusion matrix
    conf_matrix = np.zeros((num_ratings, num_ratings))
    for a, b in zip(y_true_scaled, y_pred_scaled):
        conf_matrix[a][b] += 1

    # Calculate expected matrix
    expected = np.outer(hist_true, hist_pred) / len(y_true)

    # Calculate QWK
    kappa = 1 - (np.sum(weights * conf_matrix) / np.sum(weights * expected))
    return kappa


# Compute QWK for each trait
results_float = {}
for trait in traits:
    human_col = f"{trait}(Qwen/Qwen2.5-VL-32B-Instruct)" # Fill in the MLLM name
    gold_col = f"{trait}(ground_truth)"

    if human_col in data.columns and gold_col in data.columns:
        human_scores = data[human_col].dropna()
        gold_scores = data[gold_col].dropna()

        # Ensure indices are aligned between prediction and ground truth
        common_index = human_scores.index.intersection(gold_scores.index)
        human_scores = human_scores.loc[common_index]
        gold_scores = gold_scores.loc[common_index]

        if not human_scores.empty and not gold_scores.empty:
            qwk = quadratic_weighted_kappa_float(gold_scores.tolist(), human_scores.tolist())
            results_float[trait] = qwk

# Convert results to DataFrame for output
results_float_df = pd.DataFrame(list(results_float.items()), columns=['Trait', 'QWK'])
results_float_df.to_csv('', index=False)
