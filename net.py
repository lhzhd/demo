import torch
import torch.nn as nn
from transformers import BertModel, BertConfig  # 注意：需要导入 BertConfig

# 定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, bert_path=r"E:\taiyang-bert\model"):
        super().__init__()

        # --- 1. 修改配置并加载预训练模型 (关键：Dropout 0.3) ---
        # 加载 BERT 的配置文件
        config = BertConfig.from_pretrained(bert_path)
        # 修改配置：将 Dropout 从默认的 0.1 改为 0.3
        # 这里修改两个参数，确保全连接层和注意力层都生效
        config.hidden_dropout_prob = 0.1
        config.attention_probs_dropout_prob = 0.1

        # 使用修改后的配置加载模型
        self.bert = BertModel.from_pretrained(bert_path, config=config)

        # --- 2. 【关键修改】分层冻结策略 ---
        # 策略：冻结底层 (通用语言特征)，解冻顶部 4 层 (领域语义特征)
        # BERT base 一共 12 层 (Layer 0 - Layer 11)
        # 冻结 Layer 0 到 Layer 7 (共 8 层)，解冻 Layer 8-11
        total_layers = len(self.bert.encoder.layer)
        print(f"📊 BERT 总层数: {total_layers}")

        for i in range(total_layers):
            # 如果层数索引小于 8 (即 0-7)，则冻结
            if i < 8:
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
                print(f" -> 冻结编码器第 {i} 层 (通用特征)")
            else:
                # 保留原来的打印，或者标记为解冻
                print(f" -> ✅ 解冻编码器第 {i} 层 (领域微调)")

        # --- 3. 增加分类头 ---
        # 原来的代码中有一个 self.dropout=0.4，为了防止过拟合太严重，
        # 既然 BERT 内部已经是 0.3，我们可以将这里的 Dropout 适当降低，比如 0.2
        # 或者保持 0.4 也可以（取决于你的实验效果，这里建议改为 0.3 保持一致）
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 2)

        # --- 4. 统计参数 ---
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Model 初始化完成。")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数量: {trainable_params:,} (占比 {trainable_params / total_params:.2%})")

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 提取 [CLS] 向量
        cls_vector = out.last_hidden_state[:, 0]
        cls_vector = self.dropout(cls_vector)
        logits = self.fc(cls_vector)
        return logits


# 测试代码
if __name__ == '__main__':
    print(f"当前设备: {DEVICE}")
    try:
        model = Model().to(DEVICE)
        dummy_input = torch.randint(0, 1000, (2, 192)).to(DEVICE)
        dummy_mask = torch.ones_like(dummy_input).to(DEVICE)
        dummy_type = torch.zeros_like(dummy_input).to(DEVICE)

        model.train()
        output = model(dummy_input, dummy_mask, dummy_type)

        print(f"\n🧪 测试通过:")
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   模型输出形状: {output.shape} (期望: [2, 2])")

    except Exception as e:
        print(f"❌ 模型初始化或测试失败: {e}")
        print("请检查 bert_path 路径是否正确，以及 transformers 库是否安装。")