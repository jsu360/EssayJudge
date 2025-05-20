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

def GDscore(graph_url, question, essay):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": graph_url},
                {
                    "type": "text",
                    "text": f"""
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

output_columns = list(data.columns) + ["grammatical_diversity(qwen7b)"]
pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)

print_count = 0

for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    try:
        score = GDscore(row["graph"], row["Question"], row["Essay"])
        row["grammatical_diversity(qwen7b)"] = score

        row.to_frame().T.to_csv(output_path, mode='a', index=False, header=False)

        if print_count < 5:
            print_count += 1

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception as e:
        print(f"error in row {index + 1}: {e}")
        continue

print(f"Results have been saved to：{output_path}")