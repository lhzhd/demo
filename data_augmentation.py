import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import shutil
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ================= 配置区域 =================
DATA_ROOT = r"E:\taiyang-bert\data"

# 原始数据路径
SRC_INPUT_DIR = os.path.join(DATA_ROOT, "train", "input")
SRC_LABEL_DIR = os.path.join(DATA_ROOT, "train", "label")

# 目标输出路径 (最终只读这个文件夹)
DEST_ROOT = os.path.join(DATA_ROOT, "augmented_train")
DEST_INPUT_DIR = os.path.join(DEST_ROOT, "input")
DEST_LABEL_DIR = os.path.join(DEST_ROOT, "label")

# 增强倍数 (每条原始数据生成多少条新数据)
AUGMENT_FACTOR = 2

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {DEVICE}")

# ================= 1. 准备目录 & 复制原始数据 =================
print("📂 正在准备目标目录并复制原始数据...")

# 如果目标目录已存在，先清空（防止重复叠加）
if os.path.exists(DEST_ROOT):
    print(f"⚠️ 发现已存在的文件夹: {DEST_ROOT}，正在清空...")
    shutil.rmtree(DEST_ROOT)

os.makedirs(DEST_INPUT_DIR, exist_ok=True)
os.makedirs(DEST_LABEL_DIR, exist_ok=True)


# 复制原始 input 和 label 文件到目标目录
def copy_files(src_dir, dst_dir):
    files = [f for f in os.listdir(src_dir) if f.endswith('.txt')]
    for f in tqdm(files, desc=f"复制 {os.path.basename(src_dir)}"):
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))


copy_files(SRC_INPUT_DIR, DEST_INPUT_DIR)
copy_files(SRC_LABEL_DIR, DEST_LABEL_DIR)

original_count = len([f for f in os.listdir(DEST_INPUT_DIR) if f.endswith('.txt')])
print(f"✅ 原始数据复制完成：{original_count} 条")

# ================= 2. 加载翻译模型 =================
print("\n🤖 正在加载翻译模型 (首次运行需下载)...")
MODEL_NAME_ZH_TO_EN = "Helsinki-NLP/opus-mt-zh-en"
MODEL_NAME_EN_TO_ZH = "Helsinki-NLP/opus-mt-en-zh"

tokenizer_zh2en = AutoTokenizer.from_pretrained(MODEL_NAME_ZH_TO_EN)
model_zh2en = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_ZH_TO_EN).to(DEVICE)

tokenizer_en2zh = AutoTokenizer.from_pretrained(MODEL_NAME_EN_TO_ZH)
model_en2zh = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_EN_TO_ZH).to(DEVICE)
model_zh2en.eval()
model_en2zh.eval()


def translate_zh_to_en(text):
    inputs = tokenizer_zh2en.prepare_seq2seq_batch([text], return_tensors="pt").to(DEVICE)
    outputs = model_zh2en.generate(**inputs, max_length=512)
    return tokenizer_zh2en.decode(outputs[0], skip_special_tokens=True)


def translate_en_to_zh(text):
    inputs = tokenizer_en2zh.prepare_seq2seq_batch([text], return_tensors="pt").to(DEVICE)
    outputs = model_en2zh.generate(**inputs, max_length=512)
    return tokenizer_en2zh.decode(outputs[0], skip_special_tokens=True)


def augment_text(text):
    """回译增强"""
    try:
        en_text = translate_zh_to_en(text)
        time.sleep(0.05)
        zh_back = translate_en_to_zh(en_text)
        return zh_back
    except Exception as e:
        print(f"⚠️ 翻译出错: {e}")
        return text


# ================= 3. 生成增强数据并保存 =================
print(f"\n🔄 开始生成增强数据 (倍数: {AUGMENT_FACTOR})...")

# 重新读取刚刚复制过去的原始数据（确保路径统一）
input_files = sorted([f for f in os.listdir(DEST_INPUT_DIR) if f.endswith('.txt')])

aug_count = 0
for fname in tqdm(input_files, desc="数据增强中"):
    # 读取文本
    with open(os.path.join(DEST_INPUT_DIR, fname), 'r', encoding='utf-8') as f:
        text = f.read().strip()
    # 读取标签 (同名文件)
    with open(os.path.join(DEST_LABEL_DIR, fname), 'r', encoding='utf-8') as f:
        label = f.read().strip()

    # 生成增强数据
    for i in range(AUGMENT_FACTOR):
        new_text = augment_text(text)

        # 生成新文件名：原文件名_au_i.txt (例如: 100.txt -> 100_au_0.txt)
        name_no_ext = fname.replace('.txt', '')
        new_fname = f"{name_no_ext}_au_{i}.txt"

        # 保存 Input
        with open(os.path.join(DEST_INPUT_DIR, new_fname), 'w', encoding='utf-8') as f:
            f.write(new_text)

        # 保存 Label (标签不变)
        with open(os.path.join(DEST_LABEL_DIR, new_fname), 'w', encoding='utf-8') as f:
            f.write(label)

        aug_count += 1

# ================= 4. 总结 =================
final_count = len([f for f in os.listdir(DEST_INPUT_DIR) if f.endswith('.txt')])

print("\n" + "=" * 40)
print("✅ 数据增强及合并完成！")
print("=" * 40)
print(f"📂 最终数据路径: {DEST_ROOT}")
print(f"📊 数据统计:")
print(f"   - 原始数据: {original_count} 条")
print(f"   - 新增增强: {aug_count} 条")
print(f"   - 总数据量: {final_count} 条")
print(f"\n🚀 下一步操作:")
print(f"   请修改你的训练代码 (MyData.py / train_val.py)，将训练集路径直接指向:")
print(f"   INPUT:  {DEST_INPUT_DIR}")
print(f"   LABEL:  {DEST_LABEL_DIR}")
print("=" * 40)