import os
import requests
import json
import pandas as pd
from tqdm import tqdm
import time

file_path = r'' #Fill in the data path
data = pd.read_csv(file_path)

output_file_path = r'' # fill in the output path
url = "" # Your MLLM url
api_key = ''  #

headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

def LAscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the lexical accuracy in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - Vocabulary is accurately chosen, with correct meanings and spelling, and minimal errors; words are used precisely to convey the intended meaning. 
                    4 - Vocabulary is generally accurate, with occasional slight meaning errors or minor spelling mistakes, but they do not affect overall understanding; words are fairly precise. 
                    3 - Vocabulary is mostly correct, but frequent minor errors or spelling mistakes affect some expressions; word choice is not fully precise. 
                    2 - Vocabulary is inaccurate, with significant meaning errors and frequent spelling mistakes, affecting understanding. 
                    1 - Vocabulary is severely incorrect, with unclear meanings and noticeable spelling errors, making comprehension difficult. 
                    0 - Vocabulary choice and spelling are completely incorrect, and the intended meaning is unclear or impossible to understand.
                    Please output only the number of the score (e.g. 5)：
    """

    payload = {
        "model": "llama-3.2-11b-vision-instruct",
        "stream": False,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 1500
    }

    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            if not score.isdigit():
                attempt += 1
                time.sleep(3)
                continue
            return score
        except requests.exceptions.RequestException:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None

        return None

score_col = 'lexical accuracy(llama-3.2-11b-vision-instruct)'
if score_col not in data.columns:
    data[score_col] = None

processed_records = set()
if os.path.exists(output_file_path):
    existing_data = pd.read_csv(output_file_path)

    for _, row in existing_data.iterrows():
        if not pd.isna(row[score_col]):
            processed_records.add((row['graph'], row['Question'], row['Essay']))

f_mode = 'a' if os.path.exists(output_file_path) else 'w'
with open(output_file_path, mode=f_mode, newline='', encoding='utf-8') as f:
    if f_mode == 'w':
        data.to_csv(f, index=False)
        f.flush()

with open(output_file_path, mode='a', newline='', encoding='utf-8') as f:
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        record_key = (row['graph'], row['Question'], row['Essay'])
        if record_key in processed_records:
            continue

        if pd.isna(row[score_col]) or row[score_col] is None:
            score = LAscore(row['graph'], row['Question'], row['Essay'])
            if score is not None:
                row[score_col] = score
                pd.DataFrame([row]).to_csv(f, index=False, header=False)
                f.flush()
                processed_records.add(record_key)

        time.sleep(1)

print(f"Results have been saved to {output_file_path}")
