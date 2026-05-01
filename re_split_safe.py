import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

# ================= 配置区域 =================
DATA_ROOT = r"E:\taiyang-bert\data"
SPLITS = ['train', 'val', 'test']
INPUT_DIR_NAME = 'input'
LABEL_DIR_NAME = 'label'

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# 【重要】建议开启此选项，防止不同集合间同名文件互相覆盖
# 开启后，文件名会变为 _train_xxx.txt, _val_xxx.txt，绝对安全
USE_PREFIX = True


def get_label_class(label_path):
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if '表虚' in content or content == '0':
            return 'BiaoXu'
        elif '表实' in content or content == '1':
            return 'BiaoShi'
        else:
            return 'Unknown'
    except:
        return 'Unknown'


def main():
    print("🔍 正在扫描现有数据...")
    all_samples = []

    for split_name in SPLITS:
        input_dir = os.path.join(DATA_ROOT, split_name, INPUT_DIR_NAME)
        label_dir = os.path.join(DATA_ROOT, split_name, LABEL_DIR_NAME)

        if not os.path.exists(input_dir): continue

        files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        for fname in files:
            label_path = os.path.join(label_dir, fname)
            if not os.path.exists(label_path): continue

            all_samples.append({
                'filename': fname,
                'source_split': split_name,
                'label': get_label_class(label_path),
                'src_input': os.path.join(input_dir, fname),
                'src_label': os.path.join(label_dir, fname)
            })

    if len(all_samples) == 0:
        print("❌ 无数据")
        return

    print(f"✅ 加载 {len(all_samples)} 个样本。分布：{Counter([s['label'] for s in all_samples])}")

    # --- 抽样 ---
    labels = [s['label'] for s in all_samples]
    indices = list(range(len(all_samples)))

    tv_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=42, stratify=labels)
    test_samples = [all_samples[i] for i in test_idx]

    rem_samples = [all_samples[i] for i in tv_idx]
    rem_labels = [labels[i] for i in tv_idx]
    val_r = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

    tr_idx, val_idx = train_test_split(list(range(len(rem_samples))), test_size=val_r, random_state=42,
                                       stratify=rem_labels)
    train_samples = [rem_samples[i] for i in tr_idx]
    val_samples = [rem_samples[i] for i in val_idx]

    print(f"\n🎯 划分结果: Train:{len(train_samples)}, Val:{len(val_samples)}, Test:{len(test_samples)}")

    # --- 核心逻辑：先复制，后清理 ---

    # 1. 记录所有“应该存在”的新文件名集合 (用于后续清理)
    expected_files = {
        'train': set(),
        'val': set(),
        'test': set()
    }

    def copy_samples(samples, target_split):
        target_in = os.path.join(DATA_ROOT, target_split, INPUT_DIR_NAME)
        target_lab = os.path.join(DATA_ROOT, target_split, LABEL_DIR_NAME)
        os.makedirs(target_in, exist_ok=True)
        os.makedirs(target_lab, exist_ok=True)

        count = 0
        for item in samples:
            # 生成目标文件名
            final_name = item['filename']
            if USE_PREFIX:
                final_name = f"_{item['source_split']}_{item['filename']}"

            expected_files[target_split].add(final_name)

            dst_in = os.path.join(target_in, final_name)
            dst_lab = os.path.join(target_lab, final_name)

            src_in = item['src_input']
            src_lab = item['src_label']

            # 跳过自身
            if os.path.abspath(src_in) == os.path.abspath(dst_in):
                count += 1
                continue

            # 复制
            shutil.copy2(src_in, dst_in)
            shutil.copy2(src_lab, dst_lab)
            count += 1
        print(f"   📝 {target_split}: 已写入 {count} 对文件")

    print("\n📦 第一阶段：复制所有新文件（不清理旧文件，防止误删源文件）...")
    copy_samples(train_samples, 'train')
    copy_samples(val_samples, 'val')
    copy_samples(test_samples, 'test')

    # 2. 清理阶段：删除那些不在 expected_files 列表中的旧文件
    print("\n🧹 第二阶段：清理多余旧文件...")

    for split_name in SPLITS:
        target_in = os.path.join(DATA_ROOT, split_name, INPUT_DIR_NAME)
        target_lab = os.path.join(DATA_ROOT, split_name, LABEL_DIR_NAME)

        if not os.path.exists(target_in): continue

        # 清理 Input
        current_files = [f for f in os.listdir(target_in) if f.endswith('.txt')]
        to_delete = [f for f in current_files if f not in expected_files[split_name]]
        for f in to_delete:
            os.remove(os.path.join(target_in, f))
            # 同步删除对应的 label
            lab_path = os.path.join(target_lab, f)
            if os.path.exists(lab_path):
                os.remove(lab_path)

        if to_delete:
            print(f"   🗑️ {split_name}: 删除了 {len(to_delete)} 个不再需要的旧文件")
        else:
            print(f"   ✅ {split_name}: 无需清理")

    print("\n🎉 完成！数据已安全重划，无文件丢失风险。")


if __name__ == '__main__':
    main()