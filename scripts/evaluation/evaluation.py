import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import spearmanr
import json


# evaluate against gold standard
def evaluate_chatgpt(df_path, output_csv="evaluation_results.csv"):
    # load results df
    df = pd.read_csv(df_path, encoding="utf-8")

    # count ties between 1 (in favour) and 0 (against)
    parsed_cols = [f"parsed_response{i}" for i in range(1, 12)]
    def count_1_vs_0_ties(row):
        counts = row[parsed_cols].value_counts()
        count_1 = counts.get(1, 0)
        count_0 = counts.get(0, 0)
        return count_1 == count_0 and count_1 > 0

    num_1_0_ties = df.apply(count_1_vs_0_ties, axis=1).sum()
    print(f"Number of tweets with a tie between 1 and 0: {num_1_0_ties}")
    
    # exclude uncertain majority votes
    eval_df = df.dropna(subset=["majority_vote"]).copy()
    
    # overall accuracy
    accuracy = accuracy_score(eval_df["label"], eval_df["majority_vote"])
    print(f"Overall accuracy (excluding uncertain tweets): {accuracy:.2f}")
    
    # per-class metrics (F1, precision, recall)
    print("\nPer-class classification report:")
    print(classification_report(
        eval_df["label"],
        eval_df["majority_vote"],
        target_names=["Against", "In Favour", "Neutral"]
    ))
    
    # error indicator: 1 if prediction != GS, else 0
    eval_df["error"] = (eval_df["majority_vote"] != eval_df["label"]).astype(int)
    
    # Spearman correlation between entropy and errors
    corr, p_value = spearmanr(eval_df["entropy"], eval_df["error"])
    print(f"\nSpearman correlation between entropy and errors: {corr:.2f}, p-value: {p_value:.3f}")
    
    # boxplot of entropy for correct vs incorrect predictions
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="error", y="entropy", data=eval_df)
    plt.xticks([0,1], ["Correct", "Incorrect"])
    plt.xlabel("Prediction Error")
    plt.ylabel("Entropy")
    plt.title("Entropy vs Annotation Error")
    plt.tight_layout()
    plt.show()
    
    # save eval results 
    df["error"] = df["majority_vote"] != df["label"]
    df.to_csv(output_csv, index=False)

    # text report
    txt_path = output_csv.replace(".csv", "_summary.txt")

    # 
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Overall accuracy (excluding uncertain tweets): {accuracy:.2f}\n\n")
        
        f.write("Per-class classification report:\n")
        report_str = classification_report(
            eval_df["label"],
            eval_df["majority_vote"],
            target_names=["Against", "In Favour", "Neutral"]
        )
        f.write(report_str + "\n")
        
        f.write(f"\nSpearman correlation between entropy and errors: {corr:.2f}, p-value: {p_value:.3f}\n")
        f.write(f"Number of uncertain tweets: {df['majority_vote'].isna().sum()}\n")
        f.write(f"Total evaluated tweets: {len(eval_df)}\n")

    print(f"Evaluation summary saved to: {txt_path}")