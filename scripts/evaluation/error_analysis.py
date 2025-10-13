import pandas as pd


# read dataset
df = pd.read_csv(path)

prompt_cols = [f"parsed_response{i}" for i in range(1, 12)]
accuracy_per_prompt = {}

for col in prompt_cols:
    acc = (df[col] == df["label"]).mean()
    accuracy_per_prompt[col] = acc

# convert accuracy to df 
accuracy_df = pd.DataFrame.from_dict(accuracy_per_prompt, orient="index", columns=["accuracy"])
accuracy_df = accuracy_df.sort_values("accuracy", ascending=False)
print(accuracy_df)

most_accurate = accuracy_df.idxmax()[0]
most_error_prone = accuracy_df.idxmin()[0]

print(f"Most accurate prompt: {most_accurate} (accuracy = {accuracy_df.loc[most_accurate, 'accuracy']:.2f})")
print(f"Most error-prone prompt: {most_error_prone} (accuracy = {accuracy_df.loc[most_error_prone, 'accuracy']:.2f})")