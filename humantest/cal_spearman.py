import pandas as pd
from scipy.stats import spearmanr

# 文件路径
audio_scores_file = "./audio_scores_gemini2.5_flash.csv"  # 替换为你的 audio_scores 文件路径
test_set_file = "./test_set.csv"  # 替换为你的 test_set 文件路径

# 读取 CSV 文件
audio_scores_df = pd.read_csv(audio_scores_file)  # 包含 Filename, MeanScore, Variance
test_set_df = pd.read_csv(test_set_file)  # 包含 Filename, AverageScore 等

# 按 Filename 匹配数据
merged_df = pd.merge(audio_scores_df, test_set_df, on="Filename", how="inner")

# 提取模型评分和人类评分
model_scores = merged_df["MeanScore"]
human_scores = merged_df["AverageScore"]

# 计算斯皮尔曼等级相关系数
spearman_corr, p_value = spearmanr(model_scores, human_scores)

# 打印结果
print(f"Spearman Rank Correlation Coefficient: {spearman_corr:.4f}")
print(f"P-value: {p_value:.4f}")

# 保存匹配结果到新的 CSV 文件（可选）
output_file = "matched_scores.csv"  # 替换为输出文件路径
merged_df.to_csv(output_file, index=False, encoding="utf-8")
print(f"匹配结果已保存到 {output_file}")