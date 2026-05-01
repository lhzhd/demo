# test.py - 模型评估测试模块 (包含动态阈值寻优与绘图)
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置 Matplotlib 中文显示 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ===============================================================

# 定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载字典和分词器
# 注意：这里假设你本地有 "bert-base-chinese" 或者指定路径
token = BertTokenizer.from_pretrained("bert-base-chinese")


# 数据整理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]

    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    label = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, label


def get_probs_and_labels(model, loader, device):
    """辅助函数：获取模型输出的概率和真实标签"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(outputs, dim=1)
            prob_class_1 = probs[:, 1].cpu().numpy()

            all_probs.extend(prob_class_1)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


def find_optimal_threshold_and_plot(y_true, y_probs, save_path="threshold_optimization_curve.png"):
    """
    寻找最佳阈值并绘制详细曲线图
    新增功能：绘制模型输出概率分布直方图
    限制逻辑：仅在 [0.3, 0.7] 区间内搜索最佳阈值
    """

    # ================= 新增功能 1：绘制概率分布直方图 =================
    plt.figure(figsize=(10, 6))

    # 分离“表虚”(0) 和 “表实”(1) 的预测概率
    probs_class_0 = y_probs[y_true == 0]
    probs_class_1 = y_probs[y_true == 1]

    # 绘制直方图
    plt.hist(probs_class_0, bins=20, alpha=0.7, label='真实标签: 表虚 (0)', color='skyblue', edgecolor='black')
    plt.hist(probs_class_1, bins=20, alpha=0.7, label='真实标签: 表实 (1)', color='lightcoral', edgecolor='black')

    plt.title('【诊断】模型输出概率分布直方图', fontsize=16, pad=20)
    plt.xlabel('预测为正类 (表实) 的概率', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 保存并显示直方图
    hist_path = save_path.replace(".png", "_hist.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"✅ 概率分布诊断图已生成: {hist_path}")
    plt.show()
    # =========================================================

    # 1. 【修正】定义阈值搜索范围 (限定在 0.3 到 0.7 之间)
    # 步长 0.001，确保精度
    thresholds = np.arange(0.3, 0.7, 0.001)

    f1_macro_scores = []
    f1_weighted_scores = []
    precision_scores = []
    recall_scores = []

    # 2. 遍历每个阈值，计算各项指标
    for threshold in thresholds:
        preds = (y_probs > threshold).astype(int)
        f1_macro = f1_score(y_true, preds, average='macro')
        f1_weighted = f1_score(y_true, preds, average='weighted')
        precision = precision_score(y_true, preds, average='binary')
        recall = recall_score(y_true, preds, average='binary')

        f1_macro_scores.append(f1_macro)
        f1_weighted_scores.append(f1_weighted)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # 3. 【修正】寻找平台期的中位数 (仅在限定范围内寻找)
    # 找到 F1 分数达到最大值的所有点的索引 (在当前的 thresholds 列表中)
    max_f1_macro = max(f1_macro_scores)

    # 找出所有等于最大值的索引
    best_indices_in_limited = [i for i, score in enumerate(f1_macro_scores) if score == max_f1_macro]

    # 获取平台期的起始和结束阈值 (这些值必然在 0.3-0.7 之间)
    platform_start = thresholds[best_indices_in_limited[0]]
    platform_end = thresholds[best_indices_in_limited[-1]]

    # 计算平台期的中位数
    optimal_threshold = float(np.median(thresholds[best_indices_in_limited]))

    print(f"🔍 在限定范围 [0.3, 0.7] 内搜索完成。")
    print(f"🎯 建议使用的最佳阈值: {optimal_threshold:.4f}")
    print(f"📈 对应的 Macro F1: {max_f1_macro:.4f}")

    # 4. 绘制指标曲线图 (保持不变)
    plt.figure(figsize=(12, 8))

    # 主轴 (左侧 Y 轴): F1 Score
    ax1 = plt.gca()
    color_f1_w = '#1f77b4'  # 蓝色
    color_f1_m = '#ff7f0e'  # 橙色

    ax1.plot(thresholds, f1_weighted_scores, label='Weighted F1', color=color_f1_w, linewidth=2)
    ax1.plot(thresholds, f1_macro_scores, label='Macro F1', color=color_f1_m, linestyle='--', linewidth=2)

    ax1.set_xlabel('分类阈值 (Threshold)', fontsize=12)
    ax1.set_ylabel('F1 分数 (F1 Score)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 次轴 (右侧 Y 轴): Precision / Recall
    ax2 = ax1.twinx()
    color_precision = '#2ca02c'  # 绿色
    color_recall = '#d62728'  # 红色

    ax2.plot(thresholds, precision_scores, label='Precision', color=color_precision, linestyle='-.', linewidth=2)
    ax2.plot(thresholds, recall_scores, label='Recall', color=color_recall, linestyle=':', linewidth=2)

    ax2.set_ylabel('精确率 / 召回率 (Precision / Recall)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 5. 标记最佳阈值
    # 绘制黄色虚线
    ax1.axvline(x=optimal_threshold, color='yellow', linestyle='--', linewidth=2, alpha=0.8)
    # 在线上方标注文字
    ax1.text(optimal_threshold + 0.01, max_f1_macro + 0.01,
             f'最佳阈值={optimal_threshold:.4f}',
             color='yellow', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='yellow', facecolor='black', alpha=0.8))

    # 6. 图例设置 (合并两个轴的图例)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)

    plt.title(f'阈值搜索曲线分析 (最佳阈值={optimal_threshold:.4f})', fontsize=14, pad=20)
    plt.tight_layout()

    # 保存并显示
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 曲线图已保存至: {save_path}")
    plt.show()

    return optimal_threshold



def evaluate_model_fixed_threshold(model, test_loader, device, threshold):
    """
    使用固定阈值评估模型性能
    """
    all_probs, all_labels = get_probs_and_labels(model, test_loader, device)

    # 使用固定阈值生成预测结果
    all_preds = (all_probs > threshold).astype(int)

    # 计算评估指标
    try:
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'classification_report': classification_report(all_labels, all_preds,
                                                           target_names=['表虚 (0)', '表实 (1)'], digits=4)
        }
    except Exception as e:
        print(f"❌ 计算指标时出错: {e}")
        metrics = {'error': str(e)}

    return metrics


def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'混淆矩阵 (阈值={BEST_THRESHOLD:.3f})')  # 动态显示阈值

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ 混淆矩阵已保存至：{save_path}")
    plt.show()


def save_metrics_to_file(metrics, threshold, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("模型评估报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"⚙️ 使用分类阈值：{threshold:.4f}\n\n")

        if 'error' in metrics:
            f.write(f"发生错误：{metrics['error']}\n")
            return

        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n\n")

        f.write("宏平均指标 (Macro-average):\n")
        f.write(f"  精确率 (Precision): {metrics['precision_macro']:.4f}\n")
        f.write(f"  召回率 (Recall): {metrics['recall_macro']:.4f}\n")
        f.write(f"  F1分数 (F1 Score): {metrics['f1_macro']:.4f}\n\n")

        f.write("加权平均指标 (Weighted-average):\n")
        f.write(f"  精确率 (Precision): {metrics['precision_weighted']:.4f}\n")
        f.write(f"  召回率 (Recall): {metrics['recall_weighted']:.4f}\n")
        f.write(f"  F1分数 (F1 Score): {metrics['f1_weighted']:.4f}\n\n")

        f.write("分类报告 (Classification Report):\n")
        f.write(metrics['classification_report'])

        f.write("\n\n混淆矩阵 (Confusion Matrix):\n")
        np.savetxt(f, metrics['confusion_matrix'], fmt='%d')

    print(f"✅ 评估报告已保存至：{save_path}")


if __name__ == '__main__':
    # ==========================================
    # 1. 加载数据 (验证集用于寻优，测试集用于最终评估)
    # ==========================================
    print("📂 正在加载数据集...")

    # 加载验证集 (用于寻找最佳阈值)
    val_dataset = MyDataset("val")
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    print(f"✅ VAL 数据加载完成！样本数：{len(val_dataset)}")

    # 加载测试集 (用于最终评估)
    test_dataset = MyDataset("test")
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, collate_fn=collate_fn)
    print(f"✅ TEST 数据加载完成！样本数：{len(test_dataset)}")

    # ==========================================
    # 2. 加载模型
    # ==========================================
    print(f"\n💻 使用设备：{DEVICE}")
    model = Model().to(DEVICE)
    model_path = "params/best_bert.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型参数文件不存在：{model_path}")

    print(f"📥 正在加载模型权重：{model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✅ 模型权重加载成功!")

    # ==========================================
    # 3. 动态阈值寻优 (在验证集上)
    # ==========================================
    print("\n" + "=" * 50)
    print("🔍 阶段一：在验证集上搜索最佳阈值")
    print("=" * 50)

    # 获取验证集的概率
    val_probs, val_labels = get_probs_and_labels(model, val_loader, DEVICE)

    # 寻找最佳阈值并画图
    BEST_THRESHOLD = find_optimal_threshold_and_plot(val_labels, val_probs, "threshold_search_curve.png")

    # ==========================================
    # 4. 最终评估 (在测试集上，使用最佳阈值)
    # ==========================================
    print("\n" + "=" * 50)
    print(f"🔍 阶段二：在测试集上评估 (阈值 = {BEST_THRESHOLD:.4f})")
    print("=" * 50)

    metrics = evaluate_model_fixed_threshold(model, test_loader, DEVICE, BEST_THRESHOLD)

    # 5. 输出最终结果
    if 'error' not in metrics:
        print("\n" + "🎉" * 15)
        print(f"📈 准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"📈 加权 F1 分数：{metrics['f1_weighted']:.4f}")
        print(f"📈 宏平均 F1 分数：{metrics['f1_macro']:.4f}")
        print("🎉" * 15)

        print("\n📝 详细分类报告:")
        print(metrics['classification_report'])

        # 6. 可视化与保存
        class_names = ["表虚 (0)", "表实 (1)"]
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, "final_test_confusion_matrix.png")
        save_metrics_to_file(metrics, BEST_THRESHOLD, "final_evaluation_report.txt")

        print("\n✅ 所有评估任务完成！请查看生成的图片和报告文件。")
    else:
        print("❌ 评估过程中发生错误，请检查日志。")