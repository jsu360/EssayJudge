import torch
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from tqdm import tqdm  # Import tqdm for progress bar

# Helper function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image

model_path = "deepseek-ai/deepseek-vl-7b-chat"# Fill in the path to the MLLM
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


# Function to process each row and get the score
def process_row(row):
    # Load the graph image from URL
    image_url = row['graph']
    pil_image = load_image_from_url(image_url)

    # Prepare the conversation input
    conversation = [
        {
            "role": "User",
            "content": f"""
                Assume you are an IELTS examiner. You need to score the grammatical diversity in the student's essay.
                Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0–5) according to the criteria in the rubric. The output should be only the score.
                [Rubric]: 
                    5 - Uses a variety of sentence structures, including both simple and complex sentences, with flexible use of clauses and compound sentences, demonstrating rich sentence variation. 
                    4 - Generally uses a variety of sentence structures, with appropriate use of common clauses and compound sentences. Sentence structures vary, though some sentence types lack flexibility. 
                    3 - Uses a variety of sentence structures, but with limited use of complex sentences, which often contain errors. Sentence variation is somewhat restricted. 
                    2 - Sentence structures are simple, primarily relying on simple sentences, with occasional attempts at complex sentences, though errors occur frequently. 
                    1 - Sentence structures are very basic, with almost no complex sentences, and even simple sentences contain errors. 
                    0 - Only uses simple, repetitive sentences with no complex sentences, resulting in rigid sentence structures.
                Below is the reference content:
                image: "{image_url}"
                Essay title: "{row['Question']}"
                Student's essay: "{row['Essay']}"
                Please output only the number of the score (e.g. 5):
            """,
            "images": [image_url]
        }
    ]

    # Load image and prepare inputs
    pil_images = [pil_image]
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # Run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1500,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    # Extract the score (digits only)
    score = ''.join([char for char in answer if char.isdigit()])
    return score


# Function to process the dataset and save the results
def process_dataset(input_csv, output_csv):
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Add tqdm progress bar to the apply function
    tqdm.pandas(desc="Processing rows")

    # Apply the process_row function to each row and create a new column 'ACscore'
    df['GDscore'] = df.progress_apply(process_row, axis=1)

    # Save the result to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Example usage: Process a dataset and save results
input_csv = ''  # Input CSV file path
output_csv = ''  # Output CSV file path

process_dataset(input_csv, output_csv)

