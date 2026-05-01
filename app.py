import streamlit as st
import torch
from transformers import BertTokenizer
import os

# 假设你的 Model 类定义在 net.py 中，或者你可以直接把 Model 类的定义放在这里
# 这里我们假设你有一个 Model 类
from net import Model

# ================== 1. 页面配置 ==================
st.set_page_config(
    page_title="中医太阳表虚与表实判别器",
    page_icon="🤖",
    layout="centered"
)

st.title("中医太阳表虚与表实分类系统展示")
st.write("这是一个基于 BERT 的太阳表虚与表实判别器")

# ================== 2. 加载模型 (使用缓存) ==================
# Streamlit 的 @st.cache_resource 会确保模型只加载一次，而不是每次刷新页面都加载
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "./params/best_bert.pth"
    tokenizer_path = "./model"

    # 检查文件是否存在
    if not os.path.exists(model_path):
        st.error(f"错误：未找到模型文件 {model_path}，请检查路径。")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # 初始化模型结构并加载权重
    model = Model(bert_path="./model")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # 设置为评估模式

    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# 如果模型加载失败，停止运行
if model is None:
    st.stop()

# ================== 3. 用户输入界面 ==================
st.markdown("---")
text_input = st.text_area("请输入要分析的医案：", height=150)
# 这里填入你之前计算出来的最佳阈值（中位数）
best_threshold = 0.40

# 预测按钮
if st.button("进行预测"):
    if not text_input.strip():
        st.warning("请输入医案")
    else:
        spinner = st.spinner("正在分析...")
        spinner.__enter__()  # type: ignore[attr-defined]
        try:

            # ================== 4. 推理逻辑 ==================
            # 1. 文本预处理
            inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=128)

            # 2. 模型推理
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)

            # 3. 获取概率 (假设输出是 logits)
            probs = torch.softmax(outputs, dim=1)
            prob_positive = probs[0][1].item() # 表实概率
            prob_negative = 1 - prob_positive   # 表虚概率
            pred_label = 1 if prob_positive >= best_threshold else 0
            confidence = prob_positive if pred_label == 1 else prob_negative

            # ================== 5. 结果展示 ==================
            st.markdown("---")
            st.subheader("分析结果")

            # 使用卡片式布局展示
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("预测类别", "表实" if pred_label == 1 else "表虚")

            with col2:
                st.metric("置信度", f"{confidence:.2%}")

            with col3:
                st.metric("判定阈值", f"{best_threshold:.3f}")

            # 显示概率条
            st.progress(confidence)
            st.write(f"表实概率: {prob_positive:.4f}")
            st.write(f"表虚概率: {prob_negative:.4f}")

        finally:
            spinner.__exit__(None, None, None)  # type: ignore[attr-defined]

# ================== 6. 侧边栏信息 (可选) ==================
st.sidebar.title("项目信息")
st.sidebar.info("这是基于 BERT 模型的中医太阳表虚与表实判别项目。")
st.sidebar.write("模型已加载完毕。")