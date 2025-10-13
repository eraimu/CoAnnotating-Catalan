import pandas as pd
import re
import numpy as np
from scipy.stats import entropy

# 0 --> AGAINST, 1 --> IN FAVOUR, 2 --> NEUTRAL

def parse_annotations(results_file):
    df = pd.read_csv(results_file, encoding="utf-8")

    for index, row in df.iterrows():
        tweet = row["text"]

        # parse Instruction 
        resp = str(row["response1"]).lower()

        if "favor" in resp:
            df.at[index, "parsed_response1"] = 1
        elif "neutral" in resp:
            df.at[index, "parsed_response1"] = 2
        elif "contra" in resp:
            df.at[index, "parsed_response1"] = 0
        else:
            df.at[index, "parsed_response1"] = None 
            print(f"Error! Could not parse label in tweet: {tweet} \n at response1: {resp} ") 

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response1"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response1"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response1: {resp} ")
             
        # parse Sequence swapping
        resp = str(row["response2"]).lower()

        if "favor" in resp:
            df.at[index, "parsed_response2"] = 1
        elif "neutral" in resp:
            df.at[index, "parsed_response2"] = 2
        elif "contra" in resp:
            df.at[index, "parsed_response2"] = 0
        else:
            df.at[index, "parsed_response2"] = None 
            print(f"Error! Could not parse label in tweet: {tweet} \n at response2: {resp} ") 

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response2"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response2"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response2: {resp} ")

        # parse Paraphrasing
        resp = str(row["response3"]).lower()

        if "favor" in resp:
            df.at[index, "parsed_response3"] = 1
        elif "neutral" in resp:
            df.at[index, "parsed_response3"] = 2
        elif "contra" in resp:
            df.at[index, "parsed_response3"] = 0
        else:
            df.at[index, "parsed_response3"] = None 
            print(f"Error! Could not parse label in tweet: {tweet} \n at response3: {resp} ") 

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response3"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response3"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response3: {resp} ")


        # parse True/False (3 prompts)
        resp = str(row["response4"]).lower()   # is it in favour? 
        
        if "cert" in resp or "verdadero" in resp:
            df.at[index, "parsed_response4"] = 1
        else:
            if "neutral" in resp:
                df.at[index, "parsed_response4"] = 2
            elif "contra" in resp:
                df.at[index, "parsed_response4"] = 0

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response4"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response4"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response4: {resp} ")


        resp = str(row["response5"]).lower()   # is it neutral? 
        
        if "cert" in resp or "verdadero" in resp:
            df.at[index, "parsed_response5"] = 2
        else:
            if "favor" in resp:
                df.at[index, "parsed_response5"] = 1
            elif "contra" in resp:
                df.at[index, "parsed_response5"] = 0

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response5"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response5"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response5: {resp} ")


        resp = str(row["response6"]).lower()   # is it against? 
        
        if "cert" in resp or "verdadero" in resp:
            df.at[index, "parsed_response6"] = 0
        else:
            if "favor" in resp:
                df.at[index, "parsed_response6"] = 1
            elif "neutral" in resp:
                df.at[index, "parsed_response6"] = 2

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response6"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response6"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response6: {resp} ")


        # parse Question Answering
        resp = str(row["response7"]).lower()

        if "favor" in resp:
            df.at[index, "parsed_response7"] = 1
        elif "neutral" in resp:
            df.at[index, "parsed_response7"] = 2
        elif "contra" in resp:
            df.at[index, "parsed_response7"] = 0
        else:
            df.at[index, "parsed_response7"] = None 
            print(f"Error! Could not parse label in tweet: {tweet} \n at response7: {resp} ") 

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response7"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response7"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response7: {resp} ")


        # parse Multiple Choice Question
        resp = str(row["response8"])

        if "A" in resp:
            df.at[index, "parsed_response8"] = 1
        elif "B" in resp:
            df.at[index, "parsed_response8"] = 2
        elif "C" in resp:
            df.at[index, "parsed_response8"] = 0
        else:
            df.at[index, "parsed_response8"] = None 
            print(f"Error! Could not parse label in tweet: {tweet} \n at response8: {resp} ") 

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response8"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response8"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response8: {resp} ")

        # parse Confirmation Bias (3 prompts)

        resp = str(row["response9"]).lower()   # is it in favour? 
        
        if "sí" in resp:
            df.at[index, "parsed_response9"] = 1
        else:
            if "neutral" in resp:
                df.at[index, "parsed_response9"] = 2
            elif "contra" in resp:
                df.at[index, "parsed_response9"] = 0

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response9"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response9"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response9: {resp} ")


        resp = str(row["response10"]).lower()   # is it neutral? 
        
        if "sí" in resp:
            df.at[index, "parsed_response10"] = 2
        else:
            if "favor" in resp:
                df.at[index, "parsed_response10"] = 1
            elif "contra" in resp:
                df.at[index, "parsed_response10"] = 0

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response10"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response10"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response10: {resp} ")


        resp = str(row["response11"]).lower()   # is it against? 
        
        if "sí" in resp:
            df.at[index, "parsed_response11"] = 0
        else:
            if "favor" in resp:
                df.at[index, "parsed_response11"] = 1
            elif "neutral" in resp:
                df.at[index, "parsed_response11"] = 2

        conf_match = re.search(r"\b(0\.\d+|1(?:\.0+)?)\b", resp)
        if conf_match:
            df.at[index, "confidence_response11"] = float(conf_match.group())
        else:
            df.at[index, "confidence_response11"] = None
            print(f"Error! Could not parse confidence in tweet: {tweet} \n at response11: {resp} ")

    df = df.drop(columns=["prompt1", "prompt2", "prompt3", "prompt4", "prompt5", "prompt6", "prompt7", "prompt8", "prompt9", "prompt10", "prompt11"])

    return df

# calculate majority vote 
def majority_vote(parsed_df):
    parsed_cols = [f"parsed_response{i}" for i in range(1, 12)]

    def vote(row):
        responses = row[parsed_cols].dropna().astype(int)
        if len(responses) == 0:
            return None 

        counts = responses.value_counts()
        max_count = counts.max()
        majority_labels = counts[counts == max_count].index.tolist()

        # tie-breaking rules
        if len(majority_labels) == 1:
            return majority_labels[0]  
        elif 2 in majority_labels and (0 in majority_labels or 1 in majority_labels):
            # tie between neutral and clear stance --> label as clear stance
            majority_labels = [label for label in majority_labels if label != 2]
            return majority_labels[0]
        elif 0 in majority_labels and 1 in majority_labels:
            # tie between against and in favour --> uncertain (None)
            return None
        else:
            return min(majority_labels)

    # apply to every row
    parsed_df["majority_vote"] = parsed_df.apply(vote, axis=1)

    # how many uncertain tweets
    num_uncertain = parsed_df["majority_vote"].isna().sum()
    print(f"Number of uncertain tweets: {num_uncertain}")

    return parsed_df



def add_entropy(parsed_df):
    parsed_cols = [f"parsed_response{i}" for i in range(1, 12)]

    def row_entropy(row):
        # get responses
        responses = row[parsed_cols].dropna().astype(int)
        if len(responses) == 0:
            return None 

        # count occurrences of each label and compute entropy
        counts = np.array([np.sum(responses == label) for label in [0, 1, 2]], dtype=float)
        probs = counts / counts.sum()
        return entropy(probs, base=2)

    # apply to every row
    parsed_df["entropy"] = parsed_df.apply(row_entropy, axis=1)

    return parsed_df


final_responses = parse_annotations(path) 
final_responses = majority_vote(final_responses)
final_responses = add_entropy(final_responses)
final_responses.to_csv("parsed_responses_with_entropy.csv", index=False)