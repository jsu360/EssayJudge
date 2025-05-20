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

def JPscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the persuasiveness of the justifying in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]:
                    5 - Fully addresses and accurately analyzes all important information in the image and prompt (e.g., data turning points, trends); argumentation is in-depth and logically sound. 
                    4 - Addresses most of the important information in the image and prompt, with reasonable analysis but slight shortcomings; argumentation is generally logical. 
                    3 - Addresses some important information in the image and prompt, but analysis is insufficient; argumentation is somewhat weak. 
                    2 - Mentions a small amount of important information in the image and prompt, with simple or incorrect analysis; there are significant logical issues in the argumentation. 
                    1 - Only briefly mentions important information in the image and prompt or makes clear analytical errors, lacking reasonable reasoning. 
                    0 - Fails to mention key information from the image and prompt, lacks any argumentation, and is logically incoherent.
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

score_col = 'justifying_persuasiveness(llama-3.2-11b-vision-instruct)'
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
            score = JPscore(row['graph'], row['Question'], row['Essay'])
            if score is not None:
                row[score_col] = score
                pd.DataFrame([row]).to_csv(f, index=False, header=False)
                f.flush()
                processed_records.add(record_key)

        time.sleep(1)

print(f"Results have been saved to {output_file_path}")
