#模型使用接口（主观评估）

import torch
from net import Model
from transformers import BertTokenizer

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载字典和分词器
token = BertTokenizer.from_pretrained(r"E:\taiyang-bert\model")
model = Model().to(DEVICE)
names = ["表实","表虚"]

#将传入的字符串进行编码
def collate_fn(data):
    sents = []
    sents.append(data)
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        # 当句子长度大于max_length(上限是model_max_length)时，截断
        truncation=True,
        max_length=512,
        # 一律补0到max_length
        padding="max_length",
        # 可取值为tf,pt,np,默认为list
        return_tensors="pt",
        # 返回序列长度
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    return input_ids,attention_mask,token_type_ids

def test():
    #加载模型训练参数
    checkpoint = torch.load("params/best_bert.pth",
    map_location=torch.device('cpu'))  # 加上 map_location 防止 GPU/CPU 环境不一致报错

    # 2. 提取出真正的模型权重部分
    model_weights = checkpoint['model_state_dict']

    # 3. 加载权重到模型
    model.load_state_dict(model_weights)
    #开启测试模型
    model.eval()

    while True:
        data = input("请输入测试数据（输入‘q’退出）：")
        if data=='q':
            print("测试结束")
            break
        input_ids,attention_mask,token_type_ids = collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE)

        #将数据输入到模型，得到输出
        with torch.no_grad():
            out = model(input_ids,attention_mask,token_type_ids)
            out = out.argmax(dim=1)
            print("模型判定：",names[out],"\n")

if __name__ == '__main__':
    test()