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

def PAscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the punctuation accuracy in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - Punctuation is used correctly throughout, adhering to standard rules with no errors. 
                    4 - Punctuation is mostly correct, with occasional minor errors that do not affect understanding. 
                    3 - Punctuation is generally correct, but there are some noticeable errors that slightly affect understanding. 
                    2 - There are frequent punctuation errors, some of which affect understanding. 
                    1 - Punctuation errors are severe, significantly affecting comprehension. 
                    0 - Punctuation is completely incorrect or barely used, severely hindering understanding.
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

score_col = 'punctuation_accuracy(llama-3.2-11b-vision-instruct)'
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
            score = PAscore(row['graph'], row['Question'], row['Essay'])
            if score is not None:
                row[score_col] = score
                pd.DataFrame([row]).to_csv(f, index=False, header=False)
                f.flush()
                processed_records.add(record_key)

        time.sleep(1)

print(f"Results have been saved to {output_file_path}")
