import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

torch.cuda.empty_cache()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def LDscore(graph_url, question, essay):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": graph_url},
                {
                    "type": "text",
                    "text": f"""
                    Assume you are an IELTS examiner. You need to score the lexical diversity in the student's essay.
                    Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
                    [Rubric]: 
                            5 - Vocabulary is rich and diverse, with a wide range of words used flexibly, avoiding repetition. 
                            4 - Vocabulary diversity is good, with a broad range of word choices, occasional repetition, but overall flexible expression. 
                            3 - Vocabulary diversity is average, with some variety in word choice but limited, with frequent repetition. 
                            2 - Vocabulary is fairly limited, with a lot of repetition and restricted word choice. 
                            1 - Vocabulary is very limited, with frequent repetition and an extremely narrow range of words. 
                            0 - Vocabulary is monotonous, with almost no variation, failing to demonstrate vocabulary diversity.
                    Below is the reference content:
                    image: "{graph_url}"
                    Essay title: "{question}"
                    Student's essay: "{essay}"
                    Please output only the number of the score (e.g. 5):
                    """
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    cleaned_text = generated_text[0].split("\n")[-1].strip()

    return cleaned_text


file_path = ""  # Fill in the data path
data = pd.read_csv(file_path)

output_path = ""  # Fill in the output path

output_columns = list(data.columns) + ["lexical_diversity(qwen7b)"]
pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)

print_count = 0

for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    try:
        score = LDscore(row["graph"], row["Question"], row["Essay"])
        row["lexical_diversity(qwen7b)"] = score

        row.to_frame().T.to_csv(output_path, mode='a', index=False, header=False)

        if print_count < 5:
            print_count += 1

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception as e:
        print(f"error in row {index + 1}: {e}")
        continue

print(f"Results have been saved to：{output_path}")