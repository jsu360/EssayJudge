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
                        Assume you are an IELTS examiner. You need to score the persuasiveness of the justifying in the student's essay.
                        Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0–5) according to the criteria in the rubric. The output should be only the score.
                        [Rubric]:
                                5 - Fully addresses and accurately analyzes all important information in the image and prompt (e.g., data turning points, trends); argumentation is in-depth and logically sound. 
                                4 - Addresses most of the important information in the image and prompt, with reasonable analysis but slight shortcomings; argumentation is generally logical. 
                                3 - Addresses some important information in the image and prompt, but analysis is insufficient; argumentation is somewhat weak. 
                                2 - Mentions a small amount of important information in the image and prompt, with simple or incorrect analysis; there are significant logical issues in the argumentation. 
                                1 - Only briefly mentions important information in the image and prompt or makes clear analytical errors, lacking reasonable reasoning. 
                                0 - Fails to mention key information from the image and prompt, lacks any argumentation, and is logically incoherent.
                        Below is the reference content:
                        image: "{url}"
                        Essay title: "{question}"
                        Student's essay: "{essay}"
                        Please output only the number of the score (e.g. 5):
                        """},
                    {"type": "image", "image": image},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=1500)

        output_text = processor.decode(output[0], skip_special_tokens=True)
        score = output_text.split()[-1]
        scores.append(score)

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        scores.append("Error")

df['JPscore'] = scores

output_file_path = ''  # Fill in the output path
df.to_csv(output_file_path, index=False)

print(f"File saved at: {os.path.abspath(output_file_path)}")