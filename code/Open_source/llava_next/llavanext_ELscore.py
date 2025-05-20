from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import pandas as pd
from tqdm import tqdm
import os

# Load the processor and model
processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16,
                                                          device_map="auto")

df = pd.read_csv('')  # Fill in the data path

scores = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing", ncols=100):
    url = row['graph']
    question = row['Question']
    essay = row['Essay']

    try:
        image = Image.open(requests.get(url, stream=True).raw)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f"""
                            Essay title: "{question}"
                            Student's essay: "{essay}"
                            Assume you are an IELTS examiner. You need to score the essay length in the student's essay.
                            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
                            [Rubric]: 
                                    5 - Word count is 150 words or more, with the content being substantial and without obvious excess or brevity. 
                                    4 - Word count is around 150 words, but slightly off (within 10 words), and the content is complete. 
                                    3 - Word count is noticeably too short or too long, and the content is not sufficiently substantial or is somewhat lengthy. 
                                    2 - Word count deviates significantly, failing to fully cover the requirements of the prompt. 
                                    1 - Word count is far below the requirement, and the content is incomplete. 
                                    0 - Word count is severely insufficient or excessive, making it impossible to meet the requirements of the prompt.
                                    Please output only the number of the score (e.g. 5)：
                        """},
                    {"type": "image"},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=100)

        output_text = processor.decode(output[0], skip_special_tokens=True)
        score = output_text.split()[-1]
        scores.append(score)

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        scores.append("Error")

df['ELscore'] = scores

output_file_path = ''  # Fill in the output path
df.to_csv(output_file_path, index=False)

print(f"File saved at: {os.path.abspath(output_file_path)}")
