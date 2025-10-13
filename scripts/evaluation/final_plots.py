import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


input_csv = path  
output_dir = path     
os.makedirs(output_dir, exist_ok=True)

# load data
df = pd.read_csv(input_csv, encoding="utf-8")

# if language column does not exist, infer from ID
if 'language' not in df.columns:
    df['language'] = df['id'].apply(lambda x: 'Catalan' if x.startswith('CAT') else 'Spanish')


# PLOTS
# entropy vs error (boxplot)
plt.figure(figsize=(6,4))
sns.boxplot(x="error", y="entropy", data=df)
plt.xticks([0,1], ["Correct", "Incorrect"])
plt.ylabel("Entropy")
plt.xlabel("Prediction Outcome")
plt.title("Entropy vs Error")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "entropy_vs_error_boxplot.png"))
plt.close()

# entropy per class per language (barplot)
entropy_stats = df.groupby(["language", "label"])["entropy"].mean().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(x="label", y="entropy", hue="language", data=entropy_stats)
plt.xlabel("Class Label")
plt.ylabel("Mean Entropy")
plt.title("Mean Entropy per Class per Language")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "entropy_per_class_language.png"))
plt.close()

# entropy vs mean consensus confidence, colored by error (scatter)
plt.figure(figsize=(6,4))
sns.scatterplot(
    x="mean_consensus_confidence", 
    y="entropy", 
    hue="error", 
    data=df, 
    palette={0:"green",1:"red"}
)
plt.xlabel("Mean Consensus Confidence")
plt.ylabel("Entropy")
plt.title("Entropy vs Mean Consensus Confidence")
plt.legend(title="Error", labels=["Correct","Incorrect"])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "entropy_vs_mean_confidence_scatter.png"))
plt.close()


# report
report_path = os.path.join(output_dir, "error_entropy_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== ERROR AND ENTROPY REPORT ===\n\n")
    
    # stats
    f.write("Overall statistics:\n")
    f.write(f"Total tweets: {len(df)}\n")
    f.write(f"Total errors: {df['error'].sum()} ({df['error'].mean()*100:.2f}%)\n")
    f.write(f"Mean entropy: {df['entropy'].mean():.3f}\n")
    f.write(f"Mean consensus confidence: {df['mean_consensus_confidence'].mean():.3f}\n\n")
    
    # error per class per language
    f.write("Error rate per class per language:\n")
    class_lang_stats = df.groupby(["language","label"])["error"].agg(["mean","count"]).reset_index()
    for _, row in class_lang_stats.iterrows():
        f.write(f"{row['language']} - Class {row['label']}: Error rate = {row['mean']:.2f}, N={row['count']}\n")
    f.write("\n")
    
    # entropy per class per language
    f.write("Mean entropy per class per language:\n")
    entropy_stats = df.groupby(["language","label"])["entropy"].mean().reset_index()
    for _, row in entropy_stats.iterrows():
        f.write(f"{row['language']} - Class {row['label']}: Mean entropy = {row['entropy']:.3f}\n")
    f.write("\n")
    
    # high entropy examples
    f.write("=== High-Entropy Tweets (top 10) ===\n")
    high_entropy = df.sort_values("entropy", ascending=False).head(10)
    low_entropy = df.sort_values("entropy").head(10) 

    
    # list of top 10 high-entropy tweet IDs 
    top10_ids = high_entropy["id"].tolist()
    f.write("\nList of Top 10 High-Entropy Tweet IDs:\n")
    f.write("List = [" + ", ".join(f'"{id_}"' for id_ in top10_ids) + "]\n\n")

    # list of top 10 high-entropy tweet IDs 
    top10_ids = low_entropy["id"].tolist()
    f.write("\nList of Top 10 Low-Entropy Tweet IDs:\n")
    f.write("List = [" + ", ".join(f'"{id_}"' for id_ in top10_ids) + "]\n\n")

    for _, row in high_entropy.iterrows():
        f.write(f"ID: {row['id']}\n")
        f.write(f"Language: {row['language']}\n")
        f.write(f"Label: {row['label']}, Majority: {row['majority_vote']}, Error: {row['error']}\n")
        f.write(f"Entropy: {row['entropy']:.3f}, Mean Confidence: {row['mean_consensus_confidence']:.3f}\n")
        f.write(f"Text: {row.get('text','N/A')}\n") 
        f.write("-"*40 + "\n")

print(f"Plots and report saved to {output_dir}")
