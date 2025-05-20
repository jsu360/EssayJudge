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

df['LAscore'] = scores

output_file_path = ''  # Fill in the output path
df.to_csv(output_file_path, index=False)

print(f"File saved at: {os.path.abspath(output_file_path)}")
