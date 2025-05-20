import requests
import pandas as pd
from tqdm import tqdm
import time

file_path = ''# Fill in the data url
data = pd.read_csv(file_path)
url = "" # Fill in the MLLM url
api_key = '' # Fill in the API key
headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

def ACscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the clarity of the argument in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]:
                    5 - The central argument is clear, and the first paragraph clearly outlines the topic of the image and question, providing guidance with no ambiguity.
                    4 - The central argument is clear, and the first paragraph mentions the topic of the image and question, but the guidance is slightly lacking or the expression is somewhat vague.
                    3 - The argument is generally clear, but the expression is vague, and it doesn't adequately guide the rest of the essay.
                    2 - The argument is unclear, the description is vague or incomplete, and it doesn't guide the essay.
                    1 - The argument is vague, and the first paragraph fails to effectively summarize the topic of the image or question.
                    0 - No central argument is presented, or the essay completely deviates from the topic and image.
                    Please output only the number of the score (e.g. 5)：
    """

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score
        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
    print("None is returned after multiple failed API requests")
    return None
ACscores = []

def CHscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
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
    """

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score
        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
    print("None is returned after multiple failed API requests")
    return None
CHscores = []

def ELscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the essay length in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - Word count is 150 words or more, with the content being substantial and without obvious excess or brevity. 
                    4 - Word count is around 150 words, but slightly off (within 10 words), and the content is complete. 
                    3 - Word count is noticeably too short or too long, and the content is not sufficiently substantial or is somewhat lengthy. 
                    2 - Word count deviates significantly, failing to fully cover the requirements of the prompt. 
                    1 - Word count is far below the requirement, and the content is incomplete. 
                    0 - Word count is severely insufficient or excessive, making it impossible to meet the requirements of the prompt.
                    Please output only the number of the score (e.g. 5)：
    """

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score

        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
    print("None is returned after multiple failed API requests")
    return None
ELscores = []

def GAscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the grammatical accuracy in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - Sentence structure is accurate with no grammatical errors; both simple and complex sentences are error-free. 
                    4 - Sentence structure is generally accurate, with occasional minor errors that do not affect understanding; some errors in complex sentence structures. 
                    3 - Few grammatical errors, but more noticeable errors that affect understanding; simple sentences are accurate, but complex sentences frequently contain errors. 
                    2 - Numerous grammatical errors, with sentence structure affecting understanding; simple sentences are occasionally correct, but complex sentences have frequent errors. 
                    1 - A large number of grammatical errors, with sentence structure severely affecting understanding; sentence structure is unstable, and even simple sentences contain mistakes. 
                    0 - Sentence structure is completely incorrect, nonsensical, and difficult to understand.
                    Please output only the number of the score (e.g. 5)：
    """

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score

        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
        print("None is returned after multiple failed API requests")
    return None
GAscores = []

def GDscore(image_url, question, essay):
    prompt = f"""
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

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score

        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
    print("None is returned after multiple failed API requests")
    return None
GDscores = []

def JPscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the persuasiveness of the justifying in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]:
                    5 - Fully addresses and accurately analyzes all important information in the image and prompt (e.g., data turning points, trends); argumentation is in-depth and logically sound. 
                    4 - Addresses most of the important information in the image and prompt, with reasonable analysis but slight shortcomings; argumentation is generally logical. 
                    3 - Addresses some important information in the image and prompt, but analysis is insufficient; argumentation is somewhat weak. 
                    2 - Mentions a small amount of important information in the image and prompt, with simple or incorrect analysis; there are significant logical issues in the argumentation. 
                    1 - Only briefly mentions important information in the image and prompt or makes clear analytical errors, lacking reasonable reasoning. 
                    0 - Fails to mention key information from the image and prompt, lacks any argumentation, and is logically incoherent.
                    Please output only the number of the score (e.g. 5)：
    """

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score
        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
    print("None is returned after multiple failed API requests")
    return None
JPscores = []

def LAscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the lexical accuracy in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - Vocabulary is accurately chosen, with correct meanings and spelling, and minimal errors; words are used precisely to convey the intended meaning. 
                    4 - Vocabulary is generally accurate, with occasional slight meaning errors or minor spelling mistakes, but they do not affect overall understanding; words are fairly precise. 
                    3 - Vocabulary is mostly correct, but frequent minor errors or spelling mistakes affect some expressions; word choice is not fully precise. 
                    2 - Vocabulary is inaccurate, with significant meaning errors and frequent spelling mistakes, affecting understanding. 
                    1 - Vocabulary is severely incorrect, with unclear meanings and noticeable spelling errors, making comprehension difficult. 
                    0 - Vocabulary choice and spelling are completely incorrect, and the intended meaning is unclear or impossible to understand.
                    Please output only the number of the score (e.g. 5)：
    """

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score
        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
    print("None is returned after multiple failed API requests")
    return None
LAscores = []

def LDscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the lexical diversity in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - Vocabulary is rich and diverse, with a wide range of words used flexibly, avoiding repetition. 
                    4 - Vocabulary diversity is good, with a broad range of word choices, occasional repetition, but overall flexible expression. 
                    3 - Vocabulary diversity is average, with some variety in word choice but limited, with frequent repetition. 
                    2 - Vocabulary is fairly limited, with a lot of repetition and restricted word choice. 
                    1 - Vocabulary is very limited, with frequent repetition and an extremely narrow range of words. 
                    0 - Vocabulary is monotonous, with almost no variation, failing to demonstrate vocabulary diversity.
                    Please output only the number of the score (e.g. 5)：
    """

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score
        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
    print("None is returned after multiple failed API requests")
    return None
LDscores = []

def OSscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the organizational structure in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - The essay has a well-organized structure, with clear paragraph divisions, each focused on a single theme. There are clear topic sentences and concluding sentences, and transitions between paragraphs are natural.
                    4 - The structure is generally reasonable, with fairly clear paragraph divisions, though transitions may be somewhat awkward and some paragraphs may lack clear topic sentences. 
                    3 - The structure is somewhat disorganized, with unclear paragraph divisions, a lack of topic sentences, or weak logical flow. 
                    2 - The structure is unclear, with improper paragraph divisions and poor logical coherence. 
                    1 - The paragraph structure is chaotic, with most paragraphs lacking clear topic sentences and disorganized content. 
                    0 - No paragraph structure, content is jumbled, and there is a complete lack of logical connections.
                    Please output only the number of the score (e.g. 5)：
    """

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score
        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
    print("None is returned after multiple failed API requests")
    return None
OSscores = []

def PAscore(image_url, question, essay):
    prompt = f"""
            image: "{image_url}"
            Essay title: "{question}"
            Student's essay: "{essay}"
            Assume you are an IELTS examiner. You need to score the punctuation accuracy in the student's essay.
            Based on the IELTS Writing Task 1 text prompt and image prompt, as well as the student's essay, please assign a score (0-5) according to the criteria in the rubric. The output should be only the score.
            [Rubric]: 
                    5 - Punctuation is used correctly throughout, adhering to standard rules with no errors. 
                    4 - Punctuation is mostly correct, with occasional minor errors that do not affect understanding. 
                    3 - Punctuation is generally correct, but there are some noticeable errors that slightly affect understanding. 
                    2 - There are frequent punctuation errors, some of which affect understanding. 
                    1 - Punctuation errors are severe, significantly affecting comprehension. 
                    0 - Punctuation is completely incorrect or barely used, severely hindering understanding.
                    Please output only the number of the score (e.g. 5)：
    """

    payload = {
        "model": "",# Fill in the MLLM name
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1500
    }
    attempt = 0
    while attempt < 20:
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result['choices'][0]['message']['content'].strip()
            return score
        except requests.exceptions.RequestException as e:
            attempt += 1
            time.sleep(3)
        except KeyError:
            print("error")
            return None
    print("None is returned after multiple failed API requests")
    return None
PAscores = []


for _, row in tqdm(data.iterrows(), total=len(data)):
    AC_score = ACscore(row['graph'], row['Question'], row['Essay'])
    ACscores.append(AC_score)

    CH_score = CHscore(row['graph'], row['Question'], row['Essay'])
    CHscores.append(CH_score)

    EL_score = ELscore(row['graph'], row['Question'], row['Essay'])
    ELscores.append(EL_score)

    GA_score = GAscore(row['graph'], row['Question'], row['Essay'])
    GAscores.append(GA_score)

    GD_score = GDscore(row['graph'], row['Question'], row['Essay'])
    GDscores.append(GD_score)

    JP_score = JPscore(row['graph'], row['Question'], row['Essay'])
    JPscores.append(JP_score)

    LA_score = LAscore(row['graph'], row['Question'], row['Essay'])
    LAscores.append(LA_score)

    LD_score = LDscore(row['graph'], row['Question'], row['Essay'])
    LDscores.append(LD_score)

    OS_score = OSscore(row['graph'], row['Question'], row['Essay'])
    OSscores.append(OS_score)

    PA_score = PAscore(row['graph'], row['Question'], row['Essay'])
    PAscores.append(PA_score)

# You can replace the '(claude-3-5-sonnet-20241022)' below with the name of the evaluated MLLM.
data['argument_clarity(claude-3-5-sonnet-20241022)'] = ACscores
data['coherence(claude-3-5-sonnet-20241022)'] = CHscores
data['essay_length(claude-3-5-sonnet-20241022)'] = ELscores
data['grammatical_accuracy(claude-3-5-sonnet-20241022)'] = GAscores
data['grammatical_diversity(claude-3-5-sonnet-20241022)'] = GDscores
data['justifying_persuasiveness(claude-3-5-sonnet-20241022)'] = JPscores
data['lexical_accuracydata(claude-3-5-sonnet-20241022)'] = LAscores
data['lexical_diversity(claude-3-5-sonnet-20241022)'] = LDscores
data['organizational_structure(claude-3-5-sonnet-20241022)'] = OSscores
data['punctuation_accuracy(claude-3-5-sonnet-20241022)'] = PAscores

output_file_path = ''# Fill in the path to your own output result file.
with open(output_file_path, mode='w', newline='', encoding='utf-8') as f:
    data.columns.to_list().insert(-1, 'argument_clarity(claude-3-5-sonnet-20241022)')
    data.to_csv(f, index=False)
    for _, row in tqdm(data.iterrows(), total=len(data)):
        AC_score = ACscore(row['graph'], row['Question'], row['Essay'])
        if AC_score is not None:
            row['argument_clarity(claude-3-5-sonnet-20241022)'] = AC_score
            row.to_csv(f, index=False, header=False)
        time.sleep(1)

print(f"Results have been saved to {output_file_path}")