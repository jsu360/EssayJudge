import os
import requests
import pandas as pd
from io import BytesIO
import torch
from tqdm import tqdm  # 导入 tqdm
from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    else:
        raise ValueError(f"Failed to fetch image from URL: {url}")

def single_infer(model_path, image_url, question, conv_mode="mm_default", temperature=0.2, top_p=None, num_beams=1):
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)

    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    image = load_image_from_url(image_url)

    if getattr(model.config, "image_aspect_ratio", None) == "pad":
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean)
        )
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    model = model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            stopping_criteria=[stopping_criteria],
            max_new_tokens=1500,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0].strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs

def batch_process(model_path, input_csv, output_csv):
    data = pd.read_csv(input_csv)

    data["output"] = ""

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        try:
            image_url = row["graph"]
            question = row["Question"]
            essay = row["Essay"]
            full_question = f"""
            Assume you are an IELTS examiner. You need to score the lexical accuracy in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - Vocabulary is accurately chosen, with correct meanings and spelling, and minimal errors; words are used precisely to convey the intended meaning. 
                    4 - Vocabulary is generally accurate, with occasional slight meaning errors or minor spelling mistakes, but they do not affect overall understanding; words are fairly precise. 
                    3 - Vocabulary is mostly correct, but frequent minor errors or spelling mistakes affect some expressions; word choice is not fully precise. 
                    2 - Vocabulary is inaccurate, with significant meaning errors and frequent spelling mistakes, affecting understanding. 
                    1 - Vocabulary is severely incorrect, with unclear meanings and noticeable spelling errors, making comprehension difficult. 
                    0 - Vocabulary choice and spelling are completely incorrect, and the intended meaning is unclear or impossible to understand.
            Below is the reference content:
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Please output only the number of the score (e.g. 5):
            """

            output = single_infer(
                model_path=model_path,
                image_url=image_url,
                question=full_question,
                conv_mode="mm_default",
                temperature=0.2,
                top_p=None,
                num_beams=1,
            )
            data.at[index, "output"] = output
        except Exception as e:
            print(f"Error processing row {index + 1}: {e}")
            data.at[index, "output"] = "ERROR"

    data.to_csv(output_csv, index=False)
    print(f"Batch processing completed. Results saved to {output_csv}")


if __name__ == "__main__":
    model_path = "" # Fill in the MLLM path
    input_csv = "" # Fill in the data path
    output_csv = "" # Fill in the output path

    batch_process(model_path=model_path, input_csv=input_csv, output_csv=output_csv)

