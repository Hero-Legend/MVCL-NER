import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, BertTokenizer
from datasets import load_metric
from seqeval.metrics import classification_report
import json
import pandas as pd
import torch.nn.functional as F
from torch import nn

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

#############################################读取数据################################################################
def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 读取 JSON 文件

    sentences = []
    labels = []

    for item in data:
        text = item["context"]  # 句子文本
        entity_list = item.get("entities", [])  # 获取实体列表，若不存在则设为空

        # 初始化全 O 标签
        label_seq = ["O"] * len(text)

        for entity in entity_list:
            entity_type = entity["label"]
            for span in entity["span"]:
                start, end = span
                label_seq[start] = f"B-{entity_type}"  # 开头标记
                for i in range(start + 1, end):
                    label_seq[i] = f"I-{entity_type}"  # 内部标记

        sentences.append(list(text))  # 按字符拆分
        labels.append(label_seq)

    return sentences, labels


DATA_PATH = "./data/formatted_data_fixed.json"
sentences, labels = load_data(DATA_PATH)


# 生成标签字典
unique_labels = set(tag for doc in labels for tag in doc)
label2id = {tag: i for i, tag in enumerate(sorted(unique_labels))}
id2label = {i: tag for tag, i in label2id.items()}

#print("标签映射:", label2id)

# 加载 BERT tokenizer
# 指定本地模型路径
model_path = './bert-chinese/'

# 加载分词器
tokenizer = BertTokenizerFast.from_pretrained(model_path)


# 加载模型
model = BertForTokenClassification.from_pretrained(model_path, num_labels=10)

# 数据预处理
class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, label2id, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

        # 确保 tokenizer 是 Fast 版本
        if not isinstance(self.tokenizer, BertTokenizerFast):
            raise ValueError("Error: tokenizer must be BertTokenizerFast, but got BertTokenizer!")

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]

        encoding = self.tokenizer(words, is_split_into_words=True, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        labels = [self.label2id[tag] for tag in tags]
        label_ids = []
        word_ids = encoding.word_ids(batch_index=0)  # 这里要求 Fast Tokenizer

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        encoding["labels"] = torch.tensor(label_ids)

        return {key: val.squeeze(0) for key, val in encoding.items()}
    
    def __len__(self):
        return len(self.sentences)  # ✅ 这里应该返回数据集大小



# 创建数据集
dataset = NERDataset(sentences, labels, tokenizer, label2id)

# 数据集划分 (80% 训练, 10% 验证, 10% 测试)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

print(f"Dataset size: {len(dataset)}")

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#################################################################################
class SupConLoss(nn.Module):
    """
    监督对比损失 (Supervised Contrastive Loss)
    参考: https://arxiv.org/abs/2004.11362
    """

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, mask=None):
        """
        参数:
        - features: 形状为 (batch_size, seq_len, hidden_size) 的嵌入向量
        - labels: 形状为 (batch_size, seq_len) 的标签
        - mask: 形状为 (batch_size, seq_len)，用于屏蔽 padding 位置
        """
        batch_size, seq_len, hidden_size = features.shape

        # 展平所有 token
        features = features.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)
        labels = labels.view(-1)  # (batch_size * seq_len)

        # 计算余弦相似度
        features = F.normalize(features, dim=-1)  # 归一化特征
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 生成标签 mask
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        if mask is not None:
            mask = mask.view(-1).unsqueeze(0) * mask.view(-1).unsqueeze(1)  # 仅对有效 token 计算
            label_mask *= mask

        # 计算对比损失
        pos_samples = similarity_matrix[label_mask].mean()
        neg_samples = similarity_matrix[~label_mask].mean()

        loss = -torch.log(torch.exp(pos_samples) / (torch.exp(pos_samples) + torch.exp(neg_samples)))

        return loss


class ContrastiveNERTrainer(Trainer):
    def __init__(self, contrastive_weight=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = SupConLoss()
        self.contrastive_weight = contrastive_weight  # 对比损失的权重

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算损失 (标准 NER 损失 + 对比损失)
        """
        labels = inputs.pop("labels")  # 取出标签
        outputs = model(**inputs, output_hidden_states=True)  # 获取 BERT 输出
        logits = outputs.logits  # (batch_size, seq_len, num_labels)
        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

        # 计算 NER 标准交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略 padding
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        # 计算对比损失
        contrastive_loss = self.contrastive_loss_fn(hidden_states, labels)

        # 总损失
        total_loss = loss + self.contrastive_weight * contrastive_loss
        return (total_loss, outputs) if return_outputs else total_loss




# 加载模型
model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id)

# 训练参数
training_args = TrainingArguments(
    output_dir="./bert_ner_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# 评估指标
metric = load_metric("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # 1️⃣ 确保 logits 维度一致
    if isinstance(logits, tuple):  # Trainer 可能返回 tuple
        logits = logits[0]

    logits = np.array(logits)  # 强制转换成 NumPy 数组
    labels = np.array(labels)

    print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

    # 2️⃣ 确保 logits 形状为 (batch_size, seq_len, num_labels)
    if logits.ndim == 2:  # 可能缺少 num_labels 维度
        logits = np.expand_dims(logits, axis=-1)

    # 3️⃣ 计算预测结果
    predictions = np.argmax(logits, axis=-1)

    # 4️⃣ 过滤掉 -100 位置，避免 seqeval 出错
    true_labels = [
        [id2label[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [id2label[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    # 5️⃣ 计算 NER 评估指标
    results = classification_report(true_labels, true_predictions, output_dict=True)

    return {
        "precision": results["macro avg"]["precision"],
        "recall": results["macro avg"]["recall"],
        "f1": results["macro avg"]["f1-score"],
        "accuracy": results["macro avg"]["precision"],  # 这里 accuracy 不是 seqeval 默认提供的，我们用 precision 代替
    }


# 训练器
trainer = ContrastiveNERTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    contrastive_weight=0.1,  # 控制对比损失的影响程度
)

# 训练模型
trainer.train()

# 评估
print("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(test_results)

# 预测函数
def predict(text):
    tokens = tokenizer(text, is_split_into_words=True, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    word_ids = tokens.word_ids(batch_index=0)
    predicted_labels = []
    previous_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        predicted_labels.append(id2label[predictions[idx]])
        previous_word_idx = word_idx

    return list(zip(text, predicted_labels))

# 示例预测
sample_text = ["我", "是", "北", "京", "人"]
print("预测结果:", predict(sample_text))
