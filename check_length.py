import os
import torch
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
# 指向你的数据文件夹 (请根据实际路径修改)
# 假设你的清洗后数据在 'data/cleaned' 目录下，且都是 .txt 文件
DATA_DIR = r"E:\taiyang-bert\data\train\input"
# 如果你的数据还没清洗，可以指向原始目录，但建议先运行 replace_newlines.py

# 指向你的 BERT 模型路径 (必须与 train_val.py 中一致)
BERT_PATH = r"E:\taiyang-bert\model"


# ===========================================

def get_all_texts(directory):
    texts = []
    if not os.path.exists(directory):
        print(f"❌ 错误：找不到目录 {directory}")
        return texts

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
            except Exception as e:
                print(f"读取文件 {filename} 失败: {e}")
    return texts


def main():
    print("🔄 正在加载分词器...")
    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    except Exception as e:
        print(f"❌ 加载分词器失败，请检查路径：{BERT_PATH}\n错误信息：{e}")
        return

    print("📂 正在读取文本数据...")
    texts = get_all_texts(DATA_DIR)

    if not texts:
        print("❌ 未读取到任何文本，请检查 DATA_DIR 路径。")
        return

    print(f"✅ 共读取到 {len(texts)} 条样本。")

    # 统计长度
    char_lengths = [len(t) for t in texts]
    token_lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]

    # 基础统计信息
    def print_stats(name, lengths):
        arr = np.array(lengths)
        print(f"\n--- {name} 统计 ---")
        print(f"最小值: {arr.min()}")
        print(f"最大值: {arr.max()}")
        print(f"平均值: {arr.mean():.2f}")
        print(f"中位数: {np.median(arr)}")
        print(f"90% 分位数 (90%的数据小于此值): {np.percentile(arr, 90)}")
        print(f"95% 分位数 (95%的数据小于此值): {np.percentile(arr, 95)}")
        return np.percentile(arr, 95)

    p95_char = print_stats("字符数 (Characters)", char_lengths)
    p95_token = print_stats("BERT Token 数", token_lengths)

    # 建议
    print("\n💡 优化建议:")
    recommended_length = int(p95_token) + 10  # 留一点余量
    print(f"建议将 train_val.py 中的 MAX_LENGTH 设置为: {recommended_length}")
    print(f"(当前覆盖 95% 的样本，若设为 512 可能会引入过多 PAD，影响小样本训练效果)")

    # 绘制直方图 (可选，需要安装 matplotlib: pip install matplotlib)
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(token_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(recommended_length, color='red', linestyle='--', label=f'建议长度: {recommended_length}')
        plt.axvline(512, color='green', linestyle=':', label='当前长度: 512')
        plt.title('BERT Token Length Distribution')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('length_distribution.png')
        print("\n📊 长度分布图已保存为 'length_distribution.png'")
        # 如果在服务器或无界面环境，注释掉下面这行
        # plt.show()
    except Exception as e:
        print(f"\n⚠️ 无法绘制图表 (可能未安装 matplotlib): {e}")


if __name__ == "__main__":
    main()