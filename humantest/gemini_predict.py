import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from google import genai
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

# 从环境变量中获取 API key
api_key = os.getenv("GEMINI_API")


client = genai.Client(api_key=api_key)

# 模板
template = """
Your task is to evaluate the emotional alignment between the provided audio and the given emotional description. The evaluation should focus primarily on how well the audio expresses the emotional tone described in the instruction. The evaluation should consider both the clarity of the emotional expression and the effectiveness of the audio's delivery in conveying the intended emotion.
Rate the audio on a scale of 1 to 5 based on the following criteria:
1 point: The audio completely fails to convey the emotional tone described. It may sound indifferent, flat, or contradictory to the given emotional description.
2 points: The audio somewhat conveys the described emotion but lacks depth or consistency. The emotional tone may be faint or unclear, with noticeable mismatches in certain parts.
3 points: The audio generally matches the described emotion but with minor inconsistencies. There may be parts of the audio where the emotional delivery is not as strong or is slightly off, but the overall tone aligns with the description.
4 points: The audio effectively conveys the described emotion, with only minor imperfections or slight deviations in emotional expression. The overall delivery is strong, and the emotional tone is clear and appropriate.
5 points: The audio perfectly matches the emotional description. The delivery is emotionally rich, engaging, and flawlessly expresses the intended emotion with clarity and depth.

Please note: The audio you are evaluating has been randomly selected from a pool of audio files that includes all score ranges (1 to 5) rated by humans. Your evaluation should be independent and strictly based on the provided description and the audio's emotional alignment.
Please provide your score based on the emotional expression's alignment and delivery in the audio with respect to the provided description.
Below is the description of the intended emotion:
### [Emotion Description]
{question}

After evaluating, please output ONLY the final calculated score (a number between 1.0 and 5.0, rounded to the nearest 0.5) without anything else.
Please strictly follow the standards and avoid leniency in your evaluation. Ensure that the score reflects the exact alignment between the audio and the emotional description, without overestimating or underestimating the quality.
"""

# 文件路径
test_set_csv = "./test_set.csv"  # 替换为你的输入 CSV 文件路径
output_csv = "audio_scores_gemini2.5_pro.csv"  # 替换为输出 CSV 文件路径

# 读取测试集 CSV 文件
df = pd.read_csv(test_set_csv)  # 假设 CSV 包含列：'Filename', 'Description'

# 初始化结果列表
results = []

# 遍历每一行数据
for _, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['Filename']
    description = row['instructions']

    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"Audio file not found: {filename}")
        continue

    scores = []  # 存储 5 次评分
    for i in range(5):  # 每个文件评分 5 次
        try:
            # 上传音频文件到 Gemini
            print(f"Processing {filename}, attempt {i + 1}")
            myfile = client.files.upload(file=filename)
            prompt = template.replace("{question}", description)

            # 调用 Gemini API 生成评分
            response = client.models.generate_content(
                # model='gemini-2.5-flash-preview-05-20',
                model='gemini-2.5-pro-preview-05-06',
                contents=[
                    prompt,
                    myfile,
                ]
            )

            # 获取返回的评分
            score = response.text.strip()
            print(f"Score for {filename}, attempt {i + 1}: {score}")

            # 检查 score 是否有效
            try:
                score = float(score)
                scores.append(score)
            except (ValueError, TypeError):
                print(f"Invalid score received for {filename}, attempt {i + 1}: {score}")
                continue

        except Exception as e:
            print(f"Error processing file {filename}, attempt {i + 1}: {e}")
            continue

    # 计算均值和方差
    if scores:
        mean_score = np.mean(scores)
        variance_score = np.var(scores)
        results.append({
            "Filename": filename,
            "MeanScore": round(mean_score, 2),
            "Variance": round(variance_score, 2)
        })
    else:
        print(f"No valid scores for {filename}")

# 保存结果到 CSV 文件
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"评分完成，结果已保存到 {output_csv}")