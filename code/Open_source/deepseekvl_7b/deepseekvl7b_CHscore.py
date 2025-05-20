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
                Essay title: "{row['Question']}"
                Student's essay: "{row['Essay']}"
                Assume you are an IELTS examiner. You need to score the coherence in the student's essay.
                Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
                [Rubric]:
                    5 - Transitions between sentences are natural, and logical connections flow smoothly; appropriate use of linking words and transitional phrases. 
                    4 - Sentences are generally coherent, with some transitions slightly awkward; linking words are used sparingly but are generally appropriate. 
                    3 - The logical connection between sentences is not smooth, with some sentences jumping or lacking flow; linking words are used insufficiently or inappropriately. 
                    2 - Logical connections are weak, sentence connections are awkward, and linking words are either used too little or excessively. 
                    1 - There is almost no logical connection between sentences, transitions are unnatural, and linking words are very limited or incorrect. 
                    0 - No coherence at all, with logical confusion between sentences.
                    Please output only the number of the score (e.g. 5)：
            """,
            "images": [image_url],  # Use the image URL here
        },
        {"role": "Assistant", "content": ""},
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
        max_new_tokens=512,
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
    df['CHscore'] = df.progress_apply(process_row, axis=1)

    # Save the result to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Example usage: Process a dataset and save results
input_csv = ''  # Input CSV file path
output_csv = ''  # Output CSV file path

process_dataset(input_csv, output_csv)

