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
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
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
                max_new_tokens=1500,
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
output_path = os.path.join(output_dir, 'ovis1_6_gemma2_9b_OSscores.csv')
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
