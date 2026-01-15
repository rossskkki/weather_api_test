import pandas as pd
import requests
import os
import ast
import re
from dotenv import load_dotenv
import logging
import json 
import mysql.connector
from opencc import OpenCC
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from flask import Flask, request, jsonify
load_dotenv()

# 1. 配置日志 (放在代码最开头)
# 这样错误会被记录到 'api_errors.log' 文件中，不会弄乱控制台
logging.basicConfig(
    filename='api_errors.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8' # 防止中文乱码
)
# 初始化 Flask 应用 
app = Flask(__name__)
# ================= 配置区域 (最重要部分) =================
PROMPT_FILE = 'prompt_template.txt'
# 模拟模式 (True=不花钱测试流程, False=真实请求)
MOCK_MODE = False 
OPENCC_T2S = OpenCC("t2s")

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

# ================= 核心工具函数 =================
# ... (你的 MODELS_CONFIG 定义代码保持不变) ...

def get_model_config(target_name):
    """
    根据名字获取配置，没传名字默认用第一个
    """
    # 1. 如果前端没传名字，或者传的是空，默认使用列表里的第一个
    if not target_name:
        return MODELS_CONFIG[0]

    # 2. 遍历列表查找匹配的名字
    for config in MODELS_CONFIG:
        if config["name"] == target_name:
            return config
            
    # 3. 如果找不到，返回 None (后面会处理报错)
    return None

def load_prompt():
    """
    从 txt 文件中读取 system prompt。
    """
    if not os.path.exists(PROMPT_FILE):
        print(f"❌ 错误: 找不到文件 {PROMPT_FILE}")
        return ""  # <--- 改为返回空字符串，而不是 None
    
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print(f"⚠️ 警告: {PROMPT_FILE} 文件是空的")
                return "" # <--- 改为返回空字符串
            return content
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return "" # <--- 改为返回空字符串
        
# ================= 核心工具：按标点切分 =================
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

# ================= 数据库 (Mock Database) =================
#连接 MySQL/PostgreSQL/MongoDB
GLOSS_DATABASE = os.getenv("GLOSS_DATABASE")

# 初始化数据库连接
def init_db():
    """初始化数据库连接"""
    try:
        conn = mysql.connector.connect(
            host=GLOSS_DATABASE.split(":")[0],
            port=int(GLOSS_DATABASE.split(":")[1]),
            user=os.getenv("GLOSS_DB_USER"),
            password=os.getenv("GLOSS_DB_PASSWORD"),
            database="sign_language_db"
        )
        return conn
    except mysql.connector.Error as e:
        print(f"数据库连接失败: {e}")
        return None

# ================= 数据库查询函数 =================
def upsert_missing_gloss(conn, word):
    try:
        w = str(word).strip()
        if not w:
            return
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO missing_gloss (word, count) VALUES (%s, 1) "
                "ON DUPLICATE KEY UPDATE count = count + 1",
                (w,),
            )
        conn.commit()
    except mysql.connector.Error as e:
        print(f"missing_gloss 写入失败: {e}")

def get_id_from_db(word):
    # """
    # 模拟查数据库的操作。
    # """
    # # 这里我们造一个假的字典当数据库用
    # mock_db = {
    #     "下午": 1001,
    #     "5": 1002,
    #     "时": 1003,
    #     "气温": 1004,
    #     "30": 1005,
    #     "度": 1006
    # }
    # 如果找不到，返回 None，或者你可以返回 0
    # return mock_db.get(word, None)
    
    # 实际数据库查询逻辑
    # version 1.0
    conn = init_db()
    if not conn:
        return None
    try:
        normalized = OPENCC_T2S.convert(str(word).strip())
        with conn.cursor(dictionary=True) as cursor:
            query = "SELECT word_id FROM search WHERE synonym = %s LIMIT 1"
            cursor.execute(query, (normalized,))
            result = cursor.fetchone()
            if result:
                return result['word_id']
        upsert_missing_gloss(conn, normalized)
        return None
    except mysql.connector.Error as e:
        print(f"数据库查询失败: {e}")
        return None
    finally:
        conn.close()

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

# ================= Web 接口逻辑 =================
def process_request_logic(user_input_text, current_config):
    """
    处理单个请求的核心流程：
    切分 -> 翻译 -> 清洗 -> 查库 -> 组装
    :param user_input_text: 用户文本
    :param current_config:  从前端传来的、当前选中的模型配置字典
    """
    system_prompt = load_prompt()
    segments = split_text_by_punctuation(user_input_text)
    
    final_result_list = []
    
    # 【修改点】：这里不再写死 MODELS_CONFIG[0]，而是直接使用传入的 current_config
    # current_config = MODELS_CONFIG[0]  <-- 这行删掉

    for seg in segments:
        # 1. 如果是标点，直接返回
        if is_punctuation(seg):
            final_result_list.append({
                "type": "punctuation",
                "word": seg,
                "id": None
            })
            continue
        
        # 2. 如果是文本，调用翻译
        try:
            # 调用 LLM (传入当前选中的配置)
            res = call_translation_api_generic(seg, system_prompt, current_config)
            
            # 假设 call_translation_api_generic 返回的是 {'hksl': '翻译结果...'}
            # 如果你的 api 返回结构不同，请根据实际情况调整这里
            if isinstance(res, str):
                # 兼容性处理：如果只返回了字符串
                cleaned_gloss_str = clean_gloss_text(res)
            else:
                cleaned_gloss_str = clean_gloss_text(res.get('hksl', ''))
            
            # 3. 拆解 Gloss 句子，逐词查 ID
            gloss_words = cleaned_gloss_str.split(" ")
            
            for word in gloss_words:
                if not word.strip(): continue
                
                # 查库获取 ID
                word_id = get_id_from_db(word)
                # for demo only
                if word_id is None:
                    continue
                final_result_list.append({
                    "type": "gloss",
                    "word": word,
                    "id": word_id 
                })
                
        except Exception as e:
            print(f"Translation failed for segment '{seg}': {e}")
            # 发生错误时，返回原文并标记 error
            final_result_list.append({
                "type": "error",
                "word": seg,
                "id": None
            })

    return final_result_list

# ================= Flask 路由 =================
@app.route('/api/translate', methods=['POST'])
def api_translate():
    """
    POST /api/translate
    Body: { 
        "text": "直至下午5時，錄得氣溫30度。",
        "model_name": "deepseek-chat"  <-- 可选参数
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
            
        user_text = data.get('text', '')
        # 1. 获取前端传来的模型名字 (如果没有传，就是 None)
        requested_model_name = data.get('model_name') 
        
        if not user_text:
            return jsonify({"error": "No text provided"}), 400
        
        # 2. 获取对应的配置
        selected_config = get_model_config(requested_model_name)
        
        # 3. 如果名字传错了，找不到配置，报错返回
        if selected_config is None:
            return jsonify({
                "error": f"Model '{requested_model_name}' not supported. Available: {[m['name'] for m in MODELS_CONFIG]}"
            }), 400

        print(f"📥 收到请求: {user_text}")
        print(f"🤖 使用模型: {selected_config['name']}")
        
        # 4. 【关键】把选中的配置传给处理逻辑
        result_data = process_request_logic(user_text, selected_config)
        
        return jsonify(result_data)
        
    except Exception as e:
        logging.error(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok", 
        "default_model": MODELS_CONFIG[0]['name'],
        "supported_models": [m['name'] for m in MODELS_CONFIG] # 告诉前端支持哪些
    })
# ================= 启动入口 =================

if __name__ == "__main__":
    print(f"🚀 Web 服务器启动中...")
    print(f"📡 监听地址: http://0.0.0.0:5000")
    print(f"🔧 当前使用模型: {MODELS_CONFIG[0]['name']}")
    
    # debug=True 方便开发调试，生产环境建议改为 False
    # app.run(host='127.0.0.1', port=5000, debug=False) 
    app.run(host='0.0.0.0', port=5000, debug=False)
