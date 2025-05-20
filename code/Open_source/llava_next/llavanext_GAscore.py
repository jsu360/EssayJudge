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
                            Assume you are an IELTS examiner. You need to score the grammatical accuracy in the student's essay.
                            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
                            [Rubric]: 
                                    5 - Sentence structure is accurate with no grammatical errors; both simple and complex sentences are error-free. 
                                    4 - Sentence structure is generally accurate, with occasional minor errors that do not affect understanding; some errors in complex sentence structures. 
                                    3 - Few grammatical errors, but more noticeable errors that affect understanding; simple sentences are accurate, but complex sentences frequently contain errors. 
                                    2 - Numerous grammatical errors, with sentence structure affecting understanding; simple sentences are occasionally correct, but complex sentences have frequent errors. 
                                    1 - A large number of grammatical errors, with sentence structure severely affecting understanding; sentence structure is unstable, and even simple sentences contain mistakes. 
                                    0 - Sentence structure is completely incorrect, nonsensical, and difficult to understand.
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

df['GAscore'] = scores

output_file_path = ''# Fill in the output path
df.to_csv(output_file_path, index=False)

print(f"File saved at: {os.path.abspath(output_file_path)}")
