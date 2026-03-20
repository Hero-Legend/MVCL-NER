import json
import os
import re
import torch
import random
from torch.utils.data import random_split
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm

# ==========================================
# 1. 大模型 API 配置 (保留您的 Key)
# ==========================================
LLM_CONFIGS = {
    "DeepSeek": {
        "api_key": "sk-8ea7a8e3f24742fc9ea5d558c4fe6776",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    },
    "Qwen": {
        "api_key": "sk-dc179c017a194a5589ad7eab522197ca",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max"
    },
    "Kimi": {
        "api_key": "sk-FdqsmrpVK0Nzz6SWmVHKQr7dQYR2igKPUWK7wfLRMmYcIIyE",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k"
    }
}

# 🚀 增加 CAIL2021 官方类别语义映射字典
LABEL_MAPPING = {
    "受害人": "NHVI",
    "嫌疑人或被告人": "NHCS",
    "公检法机关": "NCSP",
    "鉴定或医疗机构": "NCSM",
    "政府行政机关": "NCGV",
    "时间": "NT",
    "地点": "NS",
    "赃款赃物": "NASI",
    "作案工具": "NATS",
    "其他涉案物品": "NO"
}

# ==========================================
# 2. 复现您的测试集
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

def load_test_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    valid_data = []
    for item in data:
        text = item.get("context") or item.get("text") or ""
        if not text: continue
        valid_data.append(item)
            
    set_seed(42)
    indices = list(range(len(valid_data)))
    train_size = int(0.8 * len(indices))
    val_size = int(0.1 * len(indices))
    test_size = len(indices) - train_size - val_size
    _, _, test_indices = random_split(indices, [train_size, val_size, test_size])
    
    test_data = [valid_data[i] for i in test_indices]
    return test_data

# ==========================================
# 3. 大模型请求与解析逻辑 (结合语义映射)
# ==========================================
def call_llm(text, config):
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    
    # 告诉大模型用它听得懂的自然语言来提取
    prompt = f"""你是一个专业的中国司法信息抽取专家。请从给定的司法文本中提取命名实体。

【实体类别要求】你只能提取以下 10 种类型的实体：
1. "受害人": 受到犯罪侵害的人。
2. "嫌疑人或被告人": 实施犯罪的人。
3. "公检法机关": 公安局、检察院、法院等。
4. "鉴定或医疗机构": 医院、物价局、鉴定中心等。
5. "政府行政机关": 除公检法外的其他政府部门。
6. "时间": 案发时间。
7. "地点": 案发地点。
8. "赃款赃物": 涉案的赃款或被盗被抢的赃物。
9. "作案工具": 作案时使用的工具。
10. "其他涉案物品": 与案件相关的其他关键物品。

【输出格式】
严格以 JSON 数组格式输出，不要包含任何多余文字或 Markdown 标记。如果没有实体，输出 []。
示例：[{{"entity": "张三", "type": "受害人"}}, {{"entity": "一把菜刀", "type": "作案工具"}}]

待提取文本: {text}
"""
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "你是一个只输出JSON的机器。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, 
            max_tokens=1024
        )
        res_text = response.choices[0].message.content
        
        match = re.search(r'\[.*\]', res_text, re.DOTALL)
        if match:
            res_text = match.group(0)
            
        entities = json.loads(res_text)
        
        extracted = set()
        if isinstance(entities, list):
            for ent in entities:
                if "entity" in ent and "type" in ent:
                    chinese_type = str(ent["type"])
                    # 将大模型输出的中文类型，映射回您的英文缩写标签
                    if chinese_type in LABEL_MAPPING:
                        extracted.add((str(ent["entity"]), LABEL_MAPPING[chinese_type]))
        return extracted
    except Exception as e:
        return set()

# ==========================================
# 4. 评估逻辑 (Strict F1)
# ==========================================
def evaluate_model(model_name, test_data):
    config = LLM_CONFIGS[model_name]
    print(f"\n🚀 开始评测大模型: {model_name} ({config['model']})")
    
    true_positives = 0
    pred_positives = 0
    actual_positives = 0
    debug_printed = False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for item in test_data:
            text = item.get("context") or item.get("text") or ""
            futures[executor.submit(call_llm, text, config)] = item
            
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(test_data), desc=model_name):
            item = futures[future]
            text = item.get("context") or item.get("text") or ""
            predicted_entities = future.result()
            
            true_entities = set()
            for ent in item.get("entities", []):
                entity_type = ent.get("label") or ent.get("type") or "UNKNOWN"
                raw_spans = ent.get("span", [])
                if len(raw_spans) > 0 and not isinstance(raw_spans[0], list):
                    raw_spans = [raw_spans]
                for span in raw_spans:
                    try:
                        if isinstance(span, (list, tuple)) and len(span) == 1 and isinstance(span[0], str):
                            parts = span[0].replace(';', ',').split(',')
                            start, end = int(parts[0]), int(parts[1])
                        elif isinstance(span, (list, tuple)) and len(span) >= 2:
                            start, end = int(span[0]), int(span[1])
                        elif isinstance(span, str):
                            parts = span.replace(';', ',').split(',')
                            start, end = int(parts[0]), int(parts[1])
                        else: continue
                        
                        entity_text = text[start:end]
                        true_entities.add((entity_text, entity_type))
                    except:
                        continue

            true_positives += len(predicted_entities.intersection(true_entities))
            pred_positives += len(predicted_entities)
            actual_positives += len(true_entities)

            # 遇到完全不匹配的情况打印一次对比
            if not debug_printed and len(predicted_entities) > 0 and len(predicted_entities.intersection(true_entities)) == 0:
                print(f"\n[DEBUG] (映射转换后)")
                print(f"模型预测: {predicted_entities}")
                print(f"标准答案: {true_entities}")
                debug_printed = True

    precision = true_positives / pred_positives if pred_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 {model_name} 最终成绩 - Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%")
    return precision, recall, f1

if __name__ == "__main__":
    DATA_PATH = "./data/formatted_data_fixed.json"
    test_data = load_test_data(DATA_PATH)
    print(f"✅ 成功加载测试集 {len(test_data)} 条数据。Prompt 语义映射已生效！")
    
    # 您可以把 Qwen 和 Kimi 一次性取消注释，三个模型一起跑
    evaluate_model("DeepSeek", test_data)
    evaluate_model("Qwen", test_data)
    evaluate_model("Kimi", test_data)