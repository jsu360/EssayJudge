import os
import torch
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM
import re  # 用于正则表达式匹配
from tqdm import tqdm  # 引入tqdm

# Load model
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# Load your CSV data
df = pd.read_csv('')

# Define the output directory and create it if it doesn't exist
output_dir = ''
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Prepare to store the results
results = []

# Process each row in the dataframe using tqdm to show a progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing", unit="row"):
    # Extract image URL, question, and essay
    image_url = row['graph']
    question = row['Question']
    essay = row['Essay']

    # Load image from URL
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error loading image from URL {image_url}: {e}")
        results.append({'Index': index, 'Score': 'Error'})
        continue

    # Create the text prompt
    text = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the grammatical diversity in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - Uses a variety of sentence structures, including both simple and complex sentences, with flexible use of clauses and compound sentences, demonstrating rich sentence variation. 
                    4 - Generally uses a variety of sentence structures, with appropriate use of common clauses and compound sentences. Sentence structures vary, though some sentence types lack flexibility. 
                    3 - Uses a variety of sentence structures, but with limited use of complex sentences, which often contain errors. Sentence variation is somewhat restricted. 
                    2 - Sentence structures are simple, primarily relying on simple sentences, with occasional attempts at complex sentences, though errors occur frequently. 
                    1 - Sentence structures are very basic, with almost no complex sentences, and even simple sentences contain errors. 
                    0 - Only uses simple, repetitive sentences with no complex sentences, resulting in rigid sentence structures.
                    Please output only the number of the score (e.g. 5)：
    """
    query = f'<image>\n{text}'

    # Preprocess inputs
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

    # Generate output
    try:
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = \
            model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

            # Extract the score using regex, expecting a number between 0 and 5
            match = re.search(r'\b([0-5])\b', output)  # Match a number between 0 and 5
            if match:
                score = int(match.group(1))  # Get the matched score
            else:
                score = 'Error'  # In case no valid score was found

            results.append({'Index': index, "Essay":essay, 'Score': score})
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        results.append({'Index': index,"Essay":essay, 'Score': 'Error'})

# Save results to the specified output directory
output_path = os.path.join(output_dir, 'ovis1_6_gemma2_9b_GDscores.csv')
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
