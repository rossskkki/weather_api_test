import pandas as pd
import requests
import time
import os
import ast
import re
import math
from dotenv import load_dotenv
import logging
import json 
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# 加载 .env (如果你仍然想用环境变量存部分key)
load_dotenv()

# 1. 配置日志 (放在代码最开头)
# 这样错误会被记录到 'api_errors.log' 文件中，不会弄乱控制台
logging.basicConfig(
    filename='api_errors.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8' # 防止中文乱码
)
# ================= 配置区域 (最重要部分) =================
INPUT_FILE = 'weather_data_150.csv'
PROMPT_FILE = 'prompt_template.txt'

# Excel 格式配置 
ITEMS_PER_SHEET = 50
BASE_ROW_HEIGHT = 18
EMPTY_ROW_HEIGHT = 30
COL_WIDTHS = {
    "A": 80,  # Input (原值150可能太宽，设为80配合自动换行更佳，可自行调整)
    "B": 15,  # Model
    "C": 80,  # Infer
    "D": 10,  # 打分
    "E": 50   # 纠错
}

# --- 在这里定义你的 4-5 个模型配置 ---
# 格式：{"name": "显示在Excel的名字", "url": "API地址", "key": "API密钥", "model_id": "传给API的模型参数名"}
MODELS_CONFIG = [
    {   "name": "chatgpt-4o-latest",
        "url": os.getenv("OPENAI_API_URL"), # 示例
        "key": os.getenv("OPENAI_API_KEY"), # 可以从env读，也可以直接写字符串 "sk-xxxx"
        "model_id": os.getenv("OPENAI_MODEL_NAME"),
        "params": {
            "temperature": 0.1,       # 翻译任务通常用低温度
            "max_tokens": 500,        # 限制输出长度
            "top_p": 0.9
        }
    },
    {
        "name": "deepseek-chat",
        "url": os.getenv("OPENAI_API_URL_DEEPSEEK"), # 示例
        "key": os.getenv("OPENAI_API_KEY_DEEPSEEK"), 
        "model_id": os.getenv("OPENAI_MODEL_NAME_DEEPSEEK"),
        "params": {
            "temperature": 1,       # 翻译任务通常用低温度
            "max_tokens": 500,        # 限制输出长度
            "top_p": 1
        }
    },
    {
        "name": "gemini-2.5-pro",
        "url": os.getenv("OPENRT_API_URL"), # 假设你用的是兼容OpenAI格式的中转
        "key": os.getenv("OPENRT_API_KEY"), 
        "model_id": os.getenv("OPENRT_MODEL_NAME_GEMINI"),
        "params": {
            "temperature": 0.2,       # 翻译任务通常用低温度
            "max_tokens": 500,        # 限制输出长度
            "top_p": 1
        }
    },
    {
        "name": "claude",
        "url": os.getenv("OPENRT_API_URL"), # 假设你用的是兼容OpenAI格式的中转
        "key": os.getenv("OPENRT_API_KEY"), 
        "model_id": os.getenv("OPENRT_MODEL_NAME_CLAUDE"),
        "params": {
            "temperature": 0.7,       # 翻译任务通常用低温度
            "max_tokens": 500,        # 限制输出长度
            "top_p": 1
        }
    },
    {
        "name": "qwen3-instruct",
        "url": os.getenv("OPENAI_API_URL_QWEN"), # 假设你用的是兼容OpenAI格式的中转
        "key": os.getenv("OPENAI_API_KEY_QWEN"), 
        "model_id": os.getenv("OPENAI_MODEL_NAME_QWEN"),
        "params": {
            "temperature": 0.7,       # 翻译任务通常用低温度
            "max_tokens": 500,        # 限制输出长度
            "top_p": 0.8
        }
    },
    # 你可以继续添加更多...
]

# 模拟模式 (True=不花钱测试流程, False=真实请求)
MOCK_MODE = False 

# # ================= 核心工具：按标点切分 =================
def split_text_by_punctuation(text):
    """
    将文本按标点符号切分，同时保留标点符号。
    例如输入: "你好，世界。" 
    输出: ['你好', '，', '世界', '。']
    """
    if not isinstance(text, str):
        text = str(text)
        
    # 定义标点符号的正则模式 (包含全角和半角常见标点)
    # 这里的 () 是捕获组，re.split 会保留捕获组内的内容作为单独的列表项
    pattern = r'([，,。\.？\?！!；;：:])'
    
    # 切分
    parts = re.split(pattern, text)
    
    # 去除空字符串 (re.split 有时会在首尾产生空串)
    return [p for p in parts if p.strip() != '']

def is_punctuation(text):
    """判断一个字符串是否纯粹是标点符号"""
    return re.match(r'^[，,。\.？\?！!；;：:]+$', text.strip()) is not None

# ================= 辅助函数：清洗数据 =================
def clean_gloss_text(text):
    """
    清洗 gloss 文本。
    【重要修改】：因为我们在外层逻辑手动保留了原文标点，
    这里我们要【彻底删除】模型可能生成的标点，防止双重标点。
    只保留：汉字、英文字母、数字、空格。
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    
    text_str = str(text).strip()
    text_str = re.sub(r'^```(json)?|```$', '', text_str, flags=re.IGNORECASE | re.MULTILINE).strip()
    
    def extract_from_data(data):
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return " ".join([str(x).strip() for x in value if str(x).strip()])
            for value in data.values():
                if isinstance(value, str):
                    return value.strip()
        elif isinstance(data, list):
            return " ".join([str(x).strip() for x in data if str(x).strip()])
        return None

    # 尝试解析 JSON/AST
    try:
        data = json.loads(text_str)
        res = extract_from_data(data)
        if res: text_str = res # 如果解析成功，更新 text_str 为提取出的内容
    except: pass

    try:
        data = ast.literal_eval(text_str)
        res = extract_from_data(data)
        if res: text_str = res
    except: pass

    # 暴力清洗：如果解析失败，尝试提取引号内容
    matches = re.findall(r'["\'](.*?)["\']', text_str)
    if matches:
        filtered = [m.strip() for m in matches if m.strip() not in ['gloss', 'json', '', ' ']]
        if filtered:
            text_str = " ".join(filtered)

    # --- 最终清洗 ---
    # 替换掉所有 非单词字符 (保留中文、英文、数字、下划线) 和 空格
    # 这一步会把逗号、句号等标点全部删掉，确保只留下 Gloss 词汇
    cleaned = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', text_str) 
    
    # 合并多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

# ================= 辅助函数：读取 Prompt =================
def load_prompt():
    """
    从 txt 文件中读取 system prompt。
    如果文件不存在，返回 None。
    """
    if not os.path.exists(PROMPT_FILE):
        print(f"❌ 错误: 找不到文件 {PROMPT_FILE}")
        return None
    
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print(f"⚠️ 警告: {PROMPT_FILE} 文件是空的")
                return None
            return content
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return None

# ================= 通用 API 调用函数 =================
# 这是一个装饰器，意思是：
# stop_after_attempt(3): 最多试 3 次
# wait_fixed(2): 每次失败后等待 2 秒
# 这样你就不用在主代码里写复杂的 for 循环了
# 修改你的装饰器配置
@retry(
    # 遇到任何错误都重试（也可以指定只在 RateLimitError 时重试）
    reraise=True, 
    # 最多重试 5 次
    stop=stop_after_attempt(5), 
    # 核心：指数退避。
    # 第1次失败等 4秒，第2次等 8秒，第3次等 16秒... 最大等 60秒
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def call_translation_api_generic(text, system_prompt, config):
    # 获取该模型的自定义参数，如果没有则为空字典
    custom_params = config.get('params', {})
    
    # 打印一下当前使用的参数（调试用，可注释掉）
    # print(f"   [DEBUG] {config['name']} params: {custom_params}")

    if MOCK_MODE:
        return {"hksl": f"模拟结果({config['name']} T={custom_params.get('temperature', 'default')}): {text[:5]}...", "status": "mock"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['key']}"
    }
    
    # 1. 构建基础 Payload
    payload = {
        "model": config['model_id'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    }

    # 2. 设置默认值 (如果 config 里没写，就用这个默认值)
    # 翻译任务建议默认 temperature 较低
    if "temperature" not in custom_params:
        payload["temperature"] = 0.1 

    # 3. 【关键步骤】将 config 里的 params 合并进 payload
    # 这会覆盖上面的默认值，并添加 max_tokens, top_p 等其他参数
    payload.update(custom_params)

    # 发起请求
    response = requests.post(config['url'], json=payload, headers=headers, timeout=30)
    
    # 如果状态码是 4xx/5xx，这里会抛出异常，触发 @retry
    response.raise_for_status() 
    
    data = response.json()
    
    if "choices" in data:
        content = data['choices'][0]['message']['content']
        return {"hksl": content, "status": "success"}
    else:
        # 这种是 API 通了但返回结构不对，通常不重试，直接报错
        return {"hksl": f"结构错误: {data}", "status": "error"}

def save_formatted_excel(collected_data, filename):
    """
    实现你的格式化逻辑：
    1. 分 Sheet (每50个Item一个Sheet)
    2. 设置列宽
    3. 设置行高 (普通行18，分隔空行30)
    """
    print(f"正在生成格式化 Excel: {filename} ...")
    
    # 必须使用 xlsxwriter 引擎才能设置样式
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        # 定义通用格式：自动换行，顶部对齐
        fmt_wrap = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        
        total_items = len(collected_data)
        num_sheets = math.ceil(total_items / ITEMS_PER_SHEET)
        
        for i in range(num_sheets):
            # 1. 切片数据
            start_idx = i * ITEMS_PER_SHEET
            end_idx = min((i + 1) * ITEMS_PER_SHEET, total_items)
            chunk = collected_data[start_idx:end_idx]
            
            sheet_name = f"Sheet{i+1}-{start_idx+1}_{end_idx}"
            
            # 2. 准备 DataFrame 数据和行高记录
            sheet_rows = []
            row_heights = {} # 记录: 行号(从0开始) -> 高度
            
            # ExcelWriter 写入时，header 占第0行，数据从第1行开始
            current_row_idx = 1 
            
            for item in chunk:
                # --- A. 原文行 ---
                sheet_rows.append({
                    "Input": item['input'],
                    "Model": "", "Infer": "", "打分": "", "纠错": ""
                })
                row_heights[current_row_idx] = BASE_ROW_HEIGHT
                current_row_idx += 1
                
                # --- B. 模型结果行 ---
                for res in item['results']:
                    sheet_rows.append({
                        "Input": "",
                        "Model": res['model_name'],
                        "Infer": res['infer_text'],
                        "打分": "", "纠错": ""
                    })
                    row_heights[current_row_idx] = BASE_ROW_HEIGHT
                    current_row_idx += 1
                
                # --- C. 空行分隔 ---
                sheet_rows.append({}) # 空字典产生空行
                row_heights[current_row_idx] = EMPTY_ROW_HEIGHT
                current_row_idx += 1
            
            # 3. 写入数据到 Sheet
            df_sheet = pd.DataFrame(sheet_rows)
            # 确保列顺序
            cols = ["Input", "Model", "Infer", "打分", "纠错"]
            # 防止空数据报错
            if not df_sheet.empty:
                df_sheet = df_sheet[cols]
            
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 4. 应用格式 (列宽 & 行高)
            worksheet = writer.sheets[sheet_name]
            
            # 设置列宽
            worksheet.set_column('A:A', COL_WIDTHS['A'], fmt_wrap)
            worksheet.set_column('B:B', COL_WIDTHS['B'], fmt_wrap)
            worksheet.set_column('C:C', COL_WIDTHS['C'], fmt_wrap)
            worksheet.set_column('D:D', COL_WIDTHS['D'], fmt_wrap)
            worksheet.set_column('E:E', COL_WIDTHS['E'], fmt_wrap)
            
            # 设置行高
            for r_idx, height in row_heights.items():
                # set_row 的第一个参数是行号 (0-indexed)
                # 因为 header 是第0行，我们的 current_row_idx 也是配合 excel 逻辑的
                worksheet.set_row(r_idx, height)
                
    print("✅ Excel 生成完毕！")


# ================= 主程序逻辑 =================
def main():
    system_prompt = load_prompt()
    if system_prompt is None:
        print("⛔️ 程序终止：必须提供 prompt 模板文件。")
        return 

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件 {INPUT_FILE}")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        df = pd.read_csv(INPUT_FILE, encoding='gbk')

    target_column = 'input_text'
    if target_column not in df.columns:
        print(f"❌ CSV 中找不到列名 '{target_column}'。现有列名: {list(df.columns)}")
        return

    # 2. 读取数据 (测试时只取前几行，正式跑请去掉 .head(rows))
    rows = 2 # 测试用
    df_subset = df.head(rows).copy()
    # df_subset = df.copy() # 正式跑用这行

    # --- 1. 数据收集阶段 ---
    collected_data = [] 

    print(f"✅ 开始处理 {len(df_subset)} 条数据 (分段翻译模式)...")

    for index, row in df_subset.iterrows():
        source_text = row[target_column]
        if pd.isna(source_text) or str(source_text).strip() == "":
            continue

        print(f"\nProcessing [{index+1}/{len(df_subset)}]: {source_text[:15]}...")
        
        # 1. 切分原文：['直至下午5時', '，', '錄得氣溫30度', '，', '相對濕度百分之85']
        segments = split_text_by_punctuation(source_text)

        # 单条数据的结构
        item_data = {
            "input": source_text,
            "results": []
        }

        for config in MODELS_CONFIG:
            print(f"  -> {config['name']}... ", end="", flush=True)

            final_parts = [] # 存储这一轮模型的翻译片段
            sentence_has_error = False # 标记整句中是否有片段出错

            # 2. 遍历切分后的片段
            for seg in segments:
                # A. 如果是标点，直接原样保留，不发给模型
                if is_punctuation(seg):
                    final_parts.append(seg) 
                    continue
                
                # B. 如果是纯空格，保留或跳过
                if not seg.strip():
                    final_parts.append(seg)
                    continue

                # C. 如果是文本，调用 API (植入详细的错误处理)
                try:
                    # 【尝试执行】
                    # 这里调用函数，如果失败会自动重试 (由 tenacity 控制)
                    res = call_translation_api_generic(str(seg), system_prompt, config)

                    if res['status'] == 'success' or res['status'] == 'mock':
                        # 成功拿到结果，清洗数据
                        clean_seg_text = clean_gloss_text(res['hksl'])
                        final_parts.append(clean_seg_text)
                    else:
                        # API 通了，但返回了业务错误 (如 JSON 结构不对)
                        error_text = f"[Logic Error: {res['hksl']}]"
                        final_parts.append(error_text)
                        sentence_has_error = True
                        logging.error(f"模型: {config['name']} | 片段: {seg} | 逻辑错误: {res['hksl']}")
                
                except Exception as e:
                    # --- 彻底失败 (重试耗尽 或 网络中断) ---
                    import tenacity 
                    
                    real_error = e
                    # 尝试拆开“快递盒子”，取出真正的错误原因
                    if isinstance(e, tenacity.RetryError):
                        real_error = e.last_attempt.exception()
                    
                    error_msg_str = str(real_error)
                    
                    # 记录错误信息以便写入 Excel (用方括号包起来，方便后续筛选)
                    final_parts.append(f"[API Fail: {error_msg_str}]")
                    sentence_has_error = True
                    
                    # 记录详细日志
                    logging.error(f"模型: {config['name']} | 片段: {seg} | 异常详情: {error_msg_str}")

            # 3. 智能拼接结果 (处理空格)
            full_translation = ""
            for i, part in enumerate(final_parts):
                # 如果当前部分是标点
                if is_punctuation(part):
                    full_translation += part
                else:
                    # 如果是文本
                    # 只有当前一个部分也是文本(且不是空)时，才加空格分隔
                    # 逻辑：Gloss 词之间要有空格，但 Gloss 和标点之间通常不需要(或视情况而定)
                    if i > 0 and not is_punctuation(final_parts[i-1]) and final_parts[i-1].strip():
                        full_translation += " " + part
                    else:
                        full_translation += part

            # 4. 打印该模型本句的状态
            if sentence_has_error:
                print("⚠️ (Partial Error)") # 有部分片段失败
            else:
                print("✅") # 完美

            # 5. 存储结果
            item_data["results"].append({
                "model_name": config['name'],
                "infer_text": full_translation
            })
            
            if not MOCK_MODE: 
                time.sleep(0.2) # 避免速率限制

        collected_data.append(item_data)

    # --- 2. Excel 生成阶段 ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"formatted_result_{timestamp}.xlsx"
    
    save_formatted_excel(collected_data, output_filename)
    # (可选) 强制设置某一列的格式，确保生效
    # worksheet.set_column('C:C', COL_WIDTHS['C'], fmt_wrap) 
    # 通常你现在的代码是可以工作的，但如果不行，需要改用 workbook.add_format 然后在 write 时指定

if __name__ == "__main__":
    main()