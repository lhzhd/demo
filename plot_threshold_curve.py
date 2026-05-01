import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer

# ================= 配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_THRESHOLD = 0.425


# =======================================

def get_threshold_data(model, test_loader, device):
    """获取不同阈值下的指标数据"""
    model.eval()
    all_probs = []
    all_labels = []

    print("🔄 正在获取预测概率...")
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 定义搜索范围 (更细一点的步长，让曲线更平滑)
    thresholds = np.arange(0.30, 0.66, 0.01)

    data = {
        'thresholds': [],
        'f1_weighted': [],
        'f1_macro': [],
        'precision': [],
        'recall': [],
        'accuracy': []
    }

    for thresh in thresholds:
        preds = (all_probs > thresh).astype(int)
        try:
            data['thresholds'].append(thresh)
            data['f1_weighted'].append(f1_score(all_labels, preds, average='weighted'))
            data['f1_macro'].append(f1_score(all_labels, preds, average='macro'))
            data['precision'].append(precision_score(all_labels, preds, average='weighted'))
            data['recall'].append(recall_score(all_labels, preds, average='weighted'))
            data['accuracy'].append(accuracy_score(all_labels, preds))
        except:
            continue

    return data


def plot_threshold_curve(data, save_path="threshold_search_curve.png"):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    thresholds = data['thresholds']

    # --- 左 Y 轴：F1 分数 ---
    color_f1 = 'tab:blue'
    ax1.set_xlabel('分类阈值 (Threshold)', fontsize=12)
    ax1.set_ylabel('F1 分数 (F1 Score)', color=color_f1, fontsize=12)

    # 绘制 Weighted F1 (主指标)
    line_w, = ax1.plot(thresholds, data['f1_weighted'], color=color_f1, linewidth=2, label='Weighted F1')
    # 绘制 Macro F1
    line_m, = ax1.plot(thresholds, data['f1_macro'], color='tab:cyan', linestyle='--', linewidth=2, label='Macro F1')

    ax1.tick_params(axis='y', labelcolor=color_f1)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 右 Y 轴：精确率 & 召回率 ---
    ax2 = ax1.twinx()
    color_pr = 'tab:red'
    color_rec = 'tab:green'
    ax2.set_ylabel('精确率 / 召回率 (Precision / Recall)', fontsize=12)

    # 绘制 Precision 和 Recall
    line_p, = ax2.plot(thresholds, data['precision'], color=color_pr, linestyle='-.', linewidth=2, label='Precision')
    line_r, = ax2.plot(thresholds, data['recall'], color=color_rec, linestyle='-.', linewidth=2, label='Recall')

    ax2.tick_params(axis='y', labelcolor='black')  # 右边轴标签用黑色区分

    # --- 标注最佳点 (0.4) ---
    # 找到最接近 0.4 的索引
    idx_best = np.argmin(np.abs(np.array(data['thresholds']) - BEST_THRESHOLD))
    best_x = thresholds[idx_best]
    best_y_f1 = data['f1_weighted'][idx_best]
    best_y_rec = data['recall'][idx_best]
    best_y_pre = data['precision'][idx_best]

    # 画垂直虚线
    ax1.axvline(best_x, color='gold', linestyle='--', linewidth=2, label=f'最佳阈值 ({best_x})')

    # 画散点标记
    ax1.scatter(best_x, best_y_f1, color='gold', s=100, zorder=5, edgecolors='black')
    ax2.scatter(best_x, best_y_rec, color='gold', s=80, zorder=5, edgecolors='black')
    ax2.scatter(best_x, best_y_pre, color='gold', s=80, zorder=5, edgecolors='black')

    # 添加文字注释
    ax1.annotate(f'F1 峰值\n({best_x}, {best_y_f1:.3f})',
                 xy=(best_x, best_y_f1), xytext=(best_x + 0.02, best_y_f1 + 0.01),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=10, fontweight='bold')

    # --- 合并图例 ---
    lines = [line_w, line_m, line_p, line_r]
    labels = [l.get_label() for l in lines]
    # 添加垂直线的图例
    lines.append(plt.Line2D([0], [0], color='gold', linestyle='--', linewidth=2, label=f'最佳阈值={best_x}'))

    ax1.legend(handles=lines, loc='upper right', fontsize=10)

    plt.title(f'阈值搜索曲线分析 (最佳阈值={best_x})', fontsize=14, pad=20)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 阈值搜索曲线图已保存至：{save_path}")
    plt.show()


if __name__ == '__main__':
    # 1. 准备数据 (复用你之前的加载逻辑)
    print("📂 加载数据中...")
    tokenizer_path = r"E:\taiyang-bert\model"
    token = BertTokenizer.from_pretrained(tokenizer_path)


    def collate_fn(data):
        sents = [i[0] for i in data]
        label = [i[1] for i in data]
        data_enc = token.batch_encode_plus(sents, truncation=True, max_length=512, padding="max_length",
                                           return_tensors="pt")
        return data_enc["input_ids"], data_enc["attention_mask"], data_enc["token_type_ids"], torch.LongTensor(label)


    test_dataset = MyDataset("test")
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, collate_fn=collate_fn)

    # 2. 加载模型
    model = Model().to(DEVICE)
    checkpoint = torch.load("params/best_bert.pth", map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. 获取数据并绘图
    data = get_threshold_data(model, test_loader, DEVICE)
    plot_threshold_curve(data)