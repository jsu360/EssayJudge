import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModel, AutoTokenizer

# Fill in the output path
OUTPUT_PATH = ''

# Load the model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
                                  attn_implementation='sdpa',
                                  torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

generation_config = {
    "max_new_tokens": 1500,
    "do_sample": False,
    "temperature": 0.1
}

def is_valid_output(output):
    try:
        float(output)
        return True
    except ValueError:
        return False

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image

df = pd.read_csv('') # Fill in the data path

try:
    output_df = pd.read_csv(OUTPUT_PATH)
    processed_graphs = output_df['graph'].tolist()
    df = df[~df['graph'].isin(processed_graphs)]
except FileNotFoundError:
    output_df = pd.DataFrame(columns=['graph', 'question', 'essay', 'score'])


def save_results(df, output_filename=OUTPUT_PATH):
    df.to_csv(output_filename, index=False)

def process_data(row, max_retries=10):
    graph = row['graph']
    question = row['Question']
    essay = row['Essay']

    success = False
    attempts = 0
    score = None

    while not success and attempts < max_retries:
        try:
            image = load_image_from_url(graph)

            prompt = f"""
            Assume you are an IELTS examiner. You need to score the organizational structure in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - The essay has a well-organized structure, with clear paragraph divisions, each focused on a single theme. There are clear topic sentences and concluding sentences, and transitions between paragraphs are natural.
                    4 - The structure is generally reasonable, with fairly clear paragraph divisions, though transitions may be somewhat awkward and some paragraphs may lack clear topic sentences. 
                    3 - The structure is somewhat disorganized, with unclear paragraph divisions, a lack of topic sentences, or weak logical flow. 
                    2 - The structure is unclear, with improper paragraph divisions and poor logical coherence. 
                    1 - The paragraph structure is chaotic, with most paragraphs lacking clear topic sentences and disorganized content. 
                    0 - No paragraph structure, content is jumbled, and there is a complete lack of logical connections.
            Below is the reference content:
            image: "{graph}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Please output only the number of the score (e.g. 5)：
            """

            response = model.chat(
                image=None,
                msgs=[{"role": "user", "content": [image, prompt]}],
                tokenizer=tokenizer,
                **generation_config
            )
            score = response.strip()

            if score.isdigit() and 0 <= int(score) <= 5:
                return score
            else:
                raise ValueError("Invalid score")
        except Exception as e:
            print(f"Error at row {row.name}: {e}")
            attempts += 1

    return score

processed_count = 0
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Data"):
    score = process_data(row)

    if score is not None:
        new_row = pd.DataFrame(
            {'graph': [row['graph']], 'question': [row['Question']], 'essay': [row['Essay']], 'score': [score]})
        output_df = pd.concat([output_df, new_row], ignore_index=True)

        if processed_count < 5:
            print({score})
            processed_count += 1

save_results(output_df, OUTPUT_PATH)
