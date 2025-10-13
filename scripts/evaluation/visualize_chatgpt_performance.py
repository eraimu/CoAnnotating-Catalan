import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# per-class F1 scores
def plot_f1_per_class(df, lang_name):
    
    eval_df = df.dropna(subset=["majority_vote"]).copy()
    
    classes = ["Against", "In Favour", "Neutral"]

    # f1_score with average=None returns F1 for each class in order of labels
    f1_scores = f1_score(eval_df["label"], eval_df["majority_vote"], labels=[0,1,2], average=None)
    
    return classes, f1_scores

# boxplot of entropy for correct vs incorrect predictions
def plot_entropy_boxplot(df, lang_name):
 
    df["error"] = (df["majority_vote"] != df["label"]).astype(int)
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="error", y="entropy", data=df)
    plt.xticks([0,1], ["Correct", "Incorrect"])
    plt.xlabel("Prediction Error")
    plt.ylabel("Entropy")
    plt.title(f"Entropy vs Annotation Error ({lang_name})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # load Spanish and Catalan results
    df_ca = pd.read_csv(path, encoding="utf-8")
    df_es = pd.read_csv(path, encoding="utf-8")
    
    # F1 per class 
    classes_es, f1_es = plot_f1_per_class(df_es, "Spanish")
    classes_ca, f1_ca = plot_f1_per_class(df_ca, "Catalan")
    
    # combined barplot
    import numpy as np
    x = np.arange(len(classes_es))
    width = 0.35

    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, f1_es, width, label="Spanish")
    plt.bar(x + width/2, f1_ca, width, label="Catalan")
    plt.xticks(x, classes_es)
    plt.ylabel("F1 Score")
    plt.ylim(0,1)
    plt.title("Per-Class F1 Score Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # entropy boxplots 
    plot_entropy_boxplot(df_es, "Spanish")
    plot_entropy_boxplot(df_ca, "Catalan")

