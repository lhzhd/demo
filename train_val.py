import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer
from torch.optim import AdamW
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




# ================= 配置区域 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 100
BATCH_SIZE = 4
LEARNING_RATE = 2e-5  # 建议显式指定学习率

# 【关键修改 1】设置类别权重
# 你的数据分布大约是 Label0:117, Label1:88 (训练集)
# 为了让模型重视少数类 (Label1)，我们给 Label1 更高的权重
# 权重计算公式参考：总样本数 / (类别数 * 该类样本数) 或者简单粗暴给 1.5 ~ 2.0 倍
# 这里设置 [1.0, 1.8] 表示 Label1 的错误惩罚是 Label0 的 1.8 倍
CLASS_WEIGHTS = torch.tensor([1.0, 1.33]).to(DEVICE)

print(f"🚀 使用设备: {DEVICE}")

# 加载分词器
token = BertTokenizer.from_pretrained(r"E:\taiyang-bert\model")

def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=300,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    label = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, label


# 创建数据集
print("📂 正在加载数据集...")
train_dataset = MyDataset("train")
val_dataset = MyDataset("val")
test_dataset = MyDataset("test")  # 加载测试集用于最终评估

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                          collate_fn=collate_fn)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                        collate_fn=collate_fn)  # 验证集通常不打乱，不舍弃
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                         collate_fn=collate_fn)


def evaluate(model, loader, mode="验证集"):
    """评估函数：计算损失、准确率、并返回预测结果用于画混淆矩阵"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, label in loader:
            input_ids, attention_mask, token_type_ids, label = \
                input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)

            out = model(input_ids, attention_mask, token_type_ids)

            # 【关键修改 2】计算 Loss 时传入权重 (仅在训练/验证时有效，测试时也可用来看加权loss)
            # 注意：如果是纯测试看指标，有时会用无权重Loss，但这里为了统一逻辑先保留
            loss_batch = loss_func(out, label)
            total_loss += loss_batch.item()

            preds = out.argmax(dim=1)

            # 收集所有预测和真实标签用于后续分析
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            total_correct += (preds == label).sum().item()
            total_samples += label.size(0)

    acc = total_correct / total_samples
    avg_loss = total_loss / len(loader)

    print(f"--- {mode} 结果 ---")
    print(f"Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    return acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['label0 (表虚)', 'label1 (表实)'],
                yticklabels=['label0 (表虚)', 'label1 (表实)'])
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    save_path = f"params/{title.replace(' ', '_')}.png"
    plt.savefig(save_path)
    print(f"💾 混淆矩阵已保存至: {save_path}")
    # plt.show() # 如果在服务器运行可注释掉，本地运行可打开


if __name__ == '__main__':

    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_func = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    # 【关键修改】增加训练轮次并引入早停机制
    MAX_EPOCH = 50  # 原来只有 20，现在增加到 50，给模型更多时间学习
    PATIENCE = 8  # 如果验证集准确率连续 8 轮不提升，则提前停止
    best_val_acc = 0.0
    patience_counter = 0

    save_dir = "params"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"🔥 开始训练 (最大轮次: {MAX_EPOCH}, 早停容忍度: {PATIENCE})...")

    for epoch in range(MAX_EPOCH):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # 遍历训练集
        for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, label = \
                input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)

            out = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加当前 epoch 的统计信息
            epoch_loss += loss.item()
            preds = out.argmax(dim=1)
            epoch_correct += (preds == label).sum().item()
            epoch_total += label.size(0)

            # 只在每个 epoch 的第 1 个和最后 1 个 batch 打印细节，避免刷屏
            if i == 0 or i == len(train_loader) - 1:
                batch_acc = (preds == label).float().mean().item()
                print(f"  [Epoch {epoch}] Batch {i}: Loss={loss.item():.4f}, Batch_Acc={batch_acc:.4f}")

        # 计算本 Epoch 的平均指标
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_acc = epoch_correct / epoch_total

        print(f"\n✅ [Epoch {epoch} 完成] 训练集 -> Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")

        # === 验证阶段 ===
        val_acc, val_preds, val_labels = evaluate(model, val_loader, mode=f"Epoch {epoch} 验证集")

        # === 早停逻辑与模型保存 ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # 重置计数器

            # 保存最佳模型状态 (包含 optimizer 状态以便后续可能的微调)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }
            torch.save(checkpoint, f"{save_dir}/best_bert.pth")

            # 保存最佳时刻的混淆矩阵
            plot_confusion_matrix(val_labels, val_preds, title=f"Best_Val_Epoch{epoch}_CM")

            print(f"✨ [EPOCH {epoch}] 🎉 发现新纪录! Val_Acc: {val_acc:.4f} -> 已保存最佳模型")
        else:
            patience_counter += 1
            print(
                f"⏳ [EPOCH {epoch}] 验证集未提升 ({val_acc:.4f} < {best_val_acc:.4f}), 耐心计数: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"🛑 [触发早停] 验证集连续 {PATIENCE} 轮未提升，训练终止！")
                break

        # 每轮保存最新模型 (以防中途断开)
        torch.save(model.state_dict(), f"{save_dir}/last_bert.pth")

    # ================= 第三步：最终测试 =================
    print("\n" + "=" * 50)
    print("🧪 正在加载最佳模型并在测试集上进行最终评估...")
    print("=" * 50)

    # 加载最佳模型权重
    if os.path.exists(f"{save_dir}/best_bert.pth"):
        checkpoint = torch.load(f"{save_dir}/best_bert.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 已加载 Epoch {checkpoint['epoch']} 的最佳模型 (Val_Acc: {checkpoint['val_acc']:.4f})")
    else:
        print("⚠️ 未找到最佳模型文件，使用最后一轮模型进行评估。")
        model.load_state_dict(torch.load(f"{save_dir}/last_bert.pth"))

    model.eval()
    test_acc, test_preds, test_labels = evaluate(model, test_loader, mode="最终测试集")

    # 打印详细分类报告
    print("\n📝 最终测试集详细分类报告:")
    target_names = ['表虚 (Label 0)', '表实 (Label 1)']
    report = classification_report(test_labels, test_preds, target_names=target_names, digits=4)
    print(report)

    # 绘制最终混淆矩阵
    plot_confusion_matrix(test_labels, test_preds, title="Final_Test_Confusion_Matrix")

    print("\n🏁 全部流程结束！请查看 params 文件夹中的模型和图表。")
