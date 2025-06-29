import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModel, AutoTokenizer

# 定义输出文件的路径
OUTPUT_PATH = '/hpc2hdd/home/jsu360/project/AES/MiniCPM_V_2_6/MiniCPM_V_2_6_ACscore.csv'

# Load the model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
                                  attn_implementation='sdpa',
                                  torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

# 定义生成配置
generation_config = {
    "max_new_tokens": 1500,  # 最大生成的 token 数
    "do_sample": False,    # 是否使用采样
    "temperature": 0.1     # 温度值（控制输出的多样性）
}

# 判断输出是否为数字的辅助函数
def is_valid_output(output):
    try:
        float(output)
        return True
    except ValueError:
        return False

# 从URL加载图片的函数
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image

# 读取CSV文件
df = pd.read_csv('/hpc2hdd/home/jsu360/project/AES/data_feishu.csv')

# 如果已有部分数据处理成功，我们只需处理未成功的部分
try:
    output_df = pd.read_csv(OUTPUT_PATH)
    processed_graphs = output_df['graph'].tolist()  # 获取已处理过的图像URL
    df = df[~df['graph'].isin(processed_graphs)]  # 只处理未处理过的数据
except FileNotFoundError:
    output_df = pd.DataFrame(columns=['graph', 'question', 'essay', 'score'])  # 如果没有已保存的输出，初始化空的结果 DataFrame

# 保存结果的函数
def save_results(df, output_filename=OUTPUT_PATH):
    df.to_csv(output_filename, index=False)

# 处理每一行数据的函数
def process_data(row, max_retries=10):
    graph = row['graph']
    question = row['Question']
    essay = row['Essay']

    # 设置初始变量
    success = False
    attempts = 0
    score = None

    # 尝试获取有效的输出，最多尝试max_retries次
    while not success and attempts < max_retries:
        try:
            # 加载图像数据
            image = load_image_from_url(graph)

            # 定义prompt
            prompt = f"""
            Assume you are an IELTS examiner. You need to score the clarity of the argument in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0–5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]:
                    5 - The central argument is clear, and the first paragraph clearly outlines the topic of the image and question, providing guidance with no ambiguity.
                    4 - The central argument is clear, and the first paragraph mentions the topic of the image and question, but the guidance is slightly lacking or the expression is somewhat vague.
                    3 - The argument is generally clear, but the expression is vague, and it doesn't adequately guide the rest of the essay.
                    2 - The argument is unclear, the description is vague or incomplete, and it doesn't guide the essay.
                    1 - The argument is vague, and the first paragraph fails to effectively summarize the topic of the image or question.
                    0 - No central argument is presented, or the essay completely deviates from the topic and image.
            Below is the reference content:
            image: "{graph}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Please output only the number of the score (e.g. 5):
            """

            # 生成图像描述并评分
            response = model.chat(
                image=None,
                msgs=[{"role": "user", "content": [image, prompt]}],
                tokenizer=tokenizer,
                **generation_config
            )
            score = response.strip()  # 清除多余空白字符

            # 检查输出是否为数字且符合范围
            if score.isdigit() and 0 <= int(score) <= 5:
                return score
            else:
                raise ValueError("Invalid score")
        except Exception as e:
            print(f"Error at row {row.name}: {e}")
            attempts += 1

    return score  # 如果尝试超过最大次数，则返回None

# 逐行处理数据并输出前五行结果
processed_count = 0
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Data"):
    score = process_data(row)

    if score is not None:
        # 保存结果
        new_row = pd.DataFrame(
            {'graph': [row['graph']], 'question': [row['Question']], 'essay': [row['Essay']], 'score': [score]})
        output_df = pd.concat([output_df, new_row], ignore_index=True)

        # 仅输出前五行
        if processed_count < 5:
            print({score})
            processed_count += 1

# 保存最终的结果
save_results(output_df, OUTPUT_PATH)
