import os
import pandas as pd
from typing import List
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI

# setup
os.environ["OPENAI_API_KEY"] = "" 
MODEL_NAME = "gpt-4o-mini"   
TEMPERATURE = 0.7
N_SAMPLES = 1                
MAX_TOKENS = 70
TOP_P = 0.95
FREQ_PENALTY = 0
PRES_PENALTY = 0


class OpenAILLM:
    def __init__(self, model: str):
        self.model_source = model
        self.model = OpenAI()

    def generate(self, prompts: List[List[dict]], **kwargs):
        responses = []
        for prompt in tqdm(prompts):
            output = self.completions_with_backoff(
                model=self.model_source,
                messages=prompt,
                **kwargs
            )
            response = [x.message.content for x in output.choices]
            responses.extend(response)
        return responses

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def completions_with_backoff(self, **kwargs):
        return self.model.chat.completions.create(**kwargs)

# read dataset
df = pd.read_csv("", encoding="utf-8") 

# run prompts
model = OpenAILLM(model=MODEL_NAME)

# loop over 11 prompt columns
for i in range(1, 12):  
    col = f"prompt{i}"
    print(f"Processing {col} ...")

    # convert each prompt into OpenAI format
    formatted_prompts = [
        [{"role": "user", "content": text}] for text in df[col].tolist()
    ]

    # generate responses
    responses = model.generate(
        prompts=formatted_prompts,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p = TOP_P,
        frequency_penalty=FREQ_PENALTY,
        presence_penalty=PRES_PENALTY,
        n=N_SAMPLES
    )

    # save result to a new column
    df[f"response{i}"] = responses


df.to_csv("results.csv", index=False)