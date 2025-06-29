import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import requests
from io import BytesIO
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# 必要的模型加载部分
path = 'OpenGVLab/InternVL2-8B' # or 'OpenGVLab/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# 预处理函数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
generation_config = dict(max_new_tokens=1500, do_sample=False)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_from_url(image_url, input_size=448, max_num=12):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

data_feishu = pd.read_csv('')  #Fill in the data file path

results_df = pd.DataFrame(columns=['graph', 'question', 'essay', 'score'])

for idx, row in tqdm(data_feishu.iterrows(), total=len(data_feishu), desc="Processing Data"):
    graph = row['graph']
    question = row['Question']
    essay = row['Essay']

    success = False
    attempts = 0
    while not success and attempts < 10:
        try:
            pixel_values = load_image_from_url(graph, max_num=12).to(torch.bfloat16).cuda()

            prompt = f"""
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
            image: "{graph}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Please output only the number of the score (e.g. 5):
            """

            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            score = response.strip()

            if score.isdigit() and 0 <= int(score) <= 5:
                new_row = pd.DataFrame({'graph': [graph], 'question': [question], 'essay': [essay], 'score': [score]})
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                success = True

            else:
                raise ValueError("Invalid score")
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            attempts += 1

results_df.to_csv('', index=False) # Fill in the output file path
