from transformers import BertTokenizer

# 加载你的本地分词器
tokenizer = BertTokenizer.from_pretrained(r"E:\taiyang-bert\model")

# 测试句子
text = "太阳表虚证"

# 执行分词
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"原始文本: {text}")
print(f"分词结果: {tokens}")
print(f"对应 ID: {ids}")
print(f"词汇表大小: {len(tokenizer)}")
# 如果是 bert-base-chinese，词汇表大小通常是 21128