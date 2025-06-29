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

def CHscore(image_url, question, essay):
    prompt = f"""
            Assume you are an IELTS examiner. You need to score the coherence in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0–5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]:
                    5 - Transitions between sentences are natural, and logical connections flow smoothly; appropriate use of linking words and transitional phrases. 
                    4 - Sentences are generally coherent, with some transitions slightly awkward; linking words are used sparingly but are generally appropriate. 
                    3 - The logical connection between sentences is not smooth, with some sentences jumping or lacking flow; linking words are used insufficiently or inappropriately. 
                    2 - Logical connections are weak, sentence connections are awkward, and linking words are either used too little or excessively. 
                    1 - There is almost no logical connection between sentences, transitions are unnatural, and linking words are very limited or incorrect. 
                    0 - No coherence at all, with logical confusion between sentences.
            Below is the reference content:
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Please output only the number of the score (e.g. 5):
    """

    payload = {
        "model": "llama-3.2-11b-vision-instruct",  # Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url.strip()}},
                    {"type": "text", "text": prompt}
                ]
            }
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

score_col = 'coherence(llama-3.2-11b-vision-instruct)'
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
            score = CHscore(row['graph'], row['Question'], row['Essay'])
            if score is not None:
                row[score_col] = score
                pd.DataFrame([row]).to_csv(f, index=False, header=False)
                f.flush()
                processed_records.add(record_key)

        time.sleep(1)

print(f"Results have been saved to {output_file_path}")
