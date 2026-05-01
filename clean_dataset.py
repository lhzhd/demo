import os
import re
import shutil
from datetime import datetime

# ================= 配置区域 =================
DATA_ROOT = r"E:\taiyang-bert\data"
DO_BACKUP = True


# ================= 清洗逻辑 =================
def clean_text(text):
    """
    终极清洗函数 (v5.0 - 新增：删除首个标点前少于等于3字的短前缀)
    """
    if not text:
        return ""

    # --- 定义需要移除的标签词库 ---
    labels = [
        "初诊", "复诊", "二诊", "三诊", "首诊", "末诊",
        "主诉", "现病史", "既往史", "个人史", "家族史", "过敏史",
        "查体", "体格检查", "专科检查", "辅助检查", "实验室检查",
        "诊断", "中医诊断", "西医诊断", "辨证", "治法", "方药", "处方", "用药",
        "辰下", "刻下", "症见", "症状", "体征", "按语", "医嘱"
    ]

    label_pattern_str = r'^(' + '|'.join(labels) + r')[\s\.,:：,;.!！？]*'

    # --- 第 1 步：循环移除开头的特定标签 ---
    while True:
        match = re.match(label_pattern_str, text)
        if match:
            text = text[match.end():]
        else:
            break

    # --- 第 2 步：去除个人信息噪声 (姓名/日期/年龄/职业) ---

    # A. 去除开头的纯年龄模式 ("65 岁", "65 岁，")
    text = re.sub(r'^\d{1,3}岁[\s\.,:：,;]*', '', text)

    # B. 去除常见身份/职业词 (防止年龄删完后露出"退休教师")
    identities = ["退休教师", "教师", "工人", "农民", "干部", "职员", "学生", "儿童", "婴儿", "患儿"]
    id_pattern_str = r'^(' + '|'.join(identities) + r')[\s\.,:：,;]*'
    while True:
        match = re.match(id_pattern_str, text)
        if match:
            text = text[match.end():]
        else:
            break

    # C. 去除姓名
    text = re.sub(r'^[一 - 龟]{1,4}某[\s\.,:：,;]*', '', text)
    text = re.sub(r'^[一 - 龟]{2,5}[\s\.,:：,;]*', '', text)
    text = re.sub(r'^患者[一 - 龟]{1,4}[\s\.,:：,;]*', '', text)

    # D. 去除日期
    text = re.sub(r'\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}[日号]?[\s\.,:：,;]*', '', text)

    # E. 去除带性别的年龄
    text = re.sub(r'^[男女][\s\.,:：,;]*\d{1,3}岁', '', text)
    text = re.sub(r'^\d{1,3}岁[男女]', '', text)
    text = re.sub(r'^[男女][\s\.,:：,;]*', '', text)

    # ==========================================
    # 【新增核心逻辑】第 3 步：删除首个标点前的短前缀 (≤3字)
    # ==========================================
    # 定义标点符号集合
    punctuation_set = set(' ,.，。、；：！？()（）""''\'\":;.!?')

    while True:
        first_punct_index = -1
        # 寻找第一个标点符号的位置
        for i, char in enumerate(text):
            if char in punctuation_set:
                first_punct_index = i
                break

        if first_punct_index != -1:
            # 计算标点前的字符数
            prefix_len = first_punct_index
            # 如果长度 <= 3，则删除这部分（包括标点）
            if prefix_len <= 3:
                text = text[first_punct_index + 1:]
                # 继续循环，检查新的开头是否还是短前缀 (例如："患者，男，" -> 删"患者，" -> 剩"男，" -> 删"男，")
                continue
            else:
                # 如果前缀够长，说明是有效内容的开始，停止删除
                break
        else:
            # 如果没有标点，说明整句都没有标点，直接跳出
            break

    # --- 第 4 步：深度清理标点和格式 ---

    # A. 暴力去除开头的所有非有效字符 (以防万一)
    def is_valid_start_char(char):
        return char.isalnum() or '\u4e00' <= char <= '\u9fff'

    while len(text) > 0 and not is_valid_start_char(text[0]):
        text = text[1:]

    # B. 去除结尾的无效字符
    while len(text) > 0 and not is_valid_start_char(text[-1]):
        text = text[:-1]

    # C. 将中间的连续标点替换为单个逗号
    text = re.sub(r'[,.，.。、；;：:] {2,}', ',', text)

    # D. 将多个连续空格/换行替换为一个空格
    text = re.sub(r'\s+', ' ', text)

    # E. 最后再次 strip
    text = text.strip(' ,.，。、；：！？()（）""''\'\"')

    return text


def process_folder(split_name):
    split_dir = os.path.join(DATA_ROOT, split_name)
    input_dir = os.path.join(split_dir, "input")

    if not os.path.exists(input_dir):
        print(f"⚠️ 跳过 {split_name}: 目录不存在")
        return

    files = os.listdir(input_dir)
    txt_files = [f for f in files if f.endswith('.txt') or '.' not in f]

    if not txt_files:
        print(f"⚠️ {split_name} 中无文件")
        return

    print(f"\n🚀 开始深度清洗 [{split_name}] ({len(txt_files)} 个文件)...")

    count_cleaned = 0
    backup_dir = None
    if DO_BACKUP:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(DATA_ROOT, f"_backup_{split_name}_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)

    for filename in txt_files:
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except Exception as e:
                continue

        original = content.strip()
        cleaned = clean_text(original)

        if not cleaned:
            print(f"⚠️ {filename} 清洗后为空，跳过")
            continue

        if cleaned != original:
            if DO_BACKUP:
                shutil.copy2(file_path, os.path.join(backup_dir, filename))
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            count_cleaned += 1

    print(f"✅ [{split_name}] 完成! 更新: {count_cleaned} 个文件")
    if DO_BACKUP and count_cleaned > 0:
        print(f"   💾 备份于: {backup_dir}")


if __name__ == '__main__':
    print("🔍 数据根目录:", DATA_ROOT)
    if not os.path.exists(DATA_ROOT):
        print("❌ 路径错误，请修改 DATA_ROOT")
    else:
        # 测试案例 1: 年龄 + 职业
        test1 = "65 岁，退休教师，经常感到眩晕伴上肢麻木 3 年"
        # 测试案例 2: 短前缀 (≤3 字)
        test2 = "患者，男，因头痛..."
        test3 = "男，45 岁，头晕..."
        test4 = "诊，头痛..."
        test5 = "因，发热..."
        # 测试案例 3: 正常长句 (不应被删)
        test6 = "反复皮肤红斑 3 年，加重 1 周。"

        print(f"🧪 测试 1 (年龄职业):\n原: {test1}\n新: {clean_text(test1)}\n")
        print(f"🧪 测试 2 (患者，男):\n原: {test2}\n新: {clean_text(test2)}\n")
        print(f"🧪 测试 3 (男，45 岁):\n原: {test3}\n新: {clean_text(test3)}\n")
        print(f"🧪 测试 4 (诊，):\n原: {test4}\n新: {clean_text(test4)}\n")
        print(f"🧪 测试 5 (因，):\n原: {test5}\n新: {clean_text(test5)}\n")
        print(f"🧪 测试 6 (正常长句):\n原: {test6}\n新: {clean_text(test6)}\n")

        for split in ['train', 'val', 'test']:
            process_folder(split)
        print("\n🎉 所有数据已深度清洗！(已修复短前缀问题)")