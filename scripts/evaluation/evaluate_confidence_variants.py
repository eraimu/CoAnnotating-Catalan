import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def compute_consensus_confidence_variants(df):

    confidence_cols = [f"confidence_response{i}" for i in range(1, 12)]
    parsed_cols = [f"parsed_response{i}" for i in range(1, 12)]

    # convert parsed responses to integers (ignoring NaNs!!)
    df[parsed_cols] = df[parsed_cols].apply(pd.to_numeric, errors='coerce').astype('Int64')

    def consensus_stats(row):
        valid_conf = []
        for c, p in zip(confidence_cols, parsed_cols):
            if pd.notna(row[p]) and pd.notna(row["majority_vote"]):
                if int(row[p]) == int(row["majority_vote"]):
                    valid_conf.append(row[c])

        if valid_conf:
            return pd.Series({
                "mean_consensus_confidence": np.mean(valid_conf),
                "max_consensus_confidence": np.max(valid_conf),
                "std_consensus_confidence": np.std(valid_conf)
            })
        else:
            # leave as NaN 
            return pd.Series({
                "mean_consensus_confidence": np.nan,
                "max_consensus_confidence": np.nan,
                "std_consensus_confidence": np.nan
            })

    df_stats = df.apply(consensus_stats, axis=1)
    df = pd.concat([df, df_stats], axis=1)
    return df

def evaluate_confidence_metrics(df, metrics=["mean_consensus_confidence"], plot=True):
  
    df["error"] = (df["majority_vote"] != df["label"]).astype(int)
    
    for metric in metrics:
        # only keep rows with valid consensus confidence
        df_valid = df.dropna(subset=[metric])
        num_valid = len(df_valid)
        num_ignored = df[metric].isna().sum()
        print(f"Number of tweets ignored for {metric}: {num_ignored}")
        print(f"\nEvaluating {metric} on {num_valid} valid rows (ignoring rows with no matching prompts).")
        
        if num_valid == 0:
            print(f"No valid rows for {metric}, skipping correlation and plotting.")
            continue

        corr, pval = spearmanr(df_valid[metric], df_valid["error"])
        print(f"Spearman correlation between {metric} and errors: {corr:.2f}, p-value={pval:.3f}")
        
        if plot:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x="error", y=metric, data=df_valid)
            plt.xticks([0,1], ["Correct", "Incorrect"])
            plt.ylabel(metric.replace("_", " ").title())
            plt.xlabel("Prediction Error")
            plt.title(f"{metric.replace('_', ' ').title()} vs Annotation Error")
            plt.tight_layout()
            plt.show()

            # bin confidence values
            df_valid["conf_bin"] = pd.cut(df_valid["mean_consensus_confidence"], bins=10)

            # compute mean error rate per bin
            bin_stats = df_valid.groupby("conf_bin")["error"].mean().reset_index()

            # plot
            plt.figure(figsize=(7, 4))
            sns.lineplot(x=bin_stats["conf_bin"].astype(str), y=bin_stats["error"])
            plt.xticks(rotation=45)
            plt.ylabel("Average Error Rate")
            plt.xlabel("Binned Mean Consensus Confidence")
            plt.title("Error Rate vs Confidence")
            plt.tight_layout()
            plt.show()

            
    return df

if __name__ == "__main__":
    df = pd.read_csv(path, encoding="utf-8")
    
    # compute consensus confidence 
    df = compute_consensus_confidence_variants(df)

    # evaluate correlations with errors, ignoring rows with no matching prompts
    df = evaluate_confidence_metrics(df, metrics=["mean_consensus_confidence", "max_consensus_confidence", "std_consensus_confidence"])
    
    # save df
    df.to_csv("parsed_responses_with_confidence_variants_filtered.csv", index=False)