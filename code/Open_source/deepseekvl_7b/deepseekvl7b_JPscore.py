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
    df['JPscore'] = df.progress_apply(process_row, axis=1)

    # Save the result to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Example usage: Process a dataset and save results
input_csv = ''  # Input CSV file path
output_csv = ''  # Output CSV file path

process_dataset(input_csv, output_csv)


