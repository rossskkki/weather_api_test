# HKSL Translation System (文书转手语 Gloss 系统)

这是一个基于大语言模型 (LLM) 的自动化工具集，用于将气象文本翻译为香港手语 (HKSL) Gloss 格式。项目包含批量评测、Web API 服务以及单模型测试工具。

## 📂 项目结构

| 文件名 | 类型 | 描述 |
| :--- | :--- | :--- |
| **`all_model_test.py`** | 核心脚本 | **多模型批量对比工具**。支持同时调用多个模型（GPT-4, DeepSeek, Claude 等），自动切分长句，并将结果生成为格式精美的 Excel 报表。 |
| **`server.py`** | 后端服务 | **Flask Web 服务器**。提供 REST API 接口，支持切分、翻译、清洗以及模拟数据库 ID 匹配，供前端调用。 |
| **`api_test.py`** | 测试脚本 | **单模型测试工具**。用于快速验证后端服务
| `requirements.txt` | 依赖清单 | 项目所需的 Python 库。 |
| `.env` | 配置文件 | 存放 API Key 和 URL 等敏感信息（需自行创建）。 |
| `prompt_template.txt` | 提示词 | 存放 System Prompt 提示词模板。 |
| `weather_data_150.csv` | 数据源 | 输入的测试数据文件 (需包含 `input_text` 列)。 |

---

## 🛠️ 安装与配置

### 1. 环境准备
确保已安装 Python 3.8+。建议使用虚拟环境：

```bash
# 创建虚拟环境 (可选)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 【重要】由于 all_model_test.py 需要生成带样式的 Excel，请务必安装 XlsxWriter：
pip install XlsxWriter

2. 配置环境变量 (.env)

在项目根目录创建一个 .env 文件，填入你的模型 API 信息。参考格式如下：
ini

# OpenAI / 通用格式
OPENAI_API_URL=https://api.example.com/v1/chat/completions
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL_NAME=gpt-4o

# DeepSeek (如果使用独立配置)
OPENAI_API_URL_DEEPSEEK=https://api.deepseek.com/v1/chat/completions
OPENAI_API_KEY_DEEPSEEK=sk-xxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL_NAME_DEEPSEEK=deepseek-chat

# Qwen / Claude 等其他模型配置...

3. 准备 Prompt

确保根目录下有 prompt_template.txt 文件，里面写好了你的 System Prompt。
🚀 使用指南
1. 批量多模型评测 (all_model_test.py)

这是功能最强大的脚本，用于生成对比评测报告。

    功能特点：
        自动按标点切分长句，提高翻译准确度。
        自动清洗模型输出（去除 Markdown、JSON 格式，仅保留 Gloss）。
        Excel 格式化：生成的 Excel 会自动换行、调整列宽、设置行高，方便人工打分。
        断点续传/重试：内置 tenacity 库，API 失败会自动重试。

    运行：
    bash

    python all_model_test.py

    输出：
    生成类似 formatted_result_20240112_120000.xlsx 的文件。

2. 启动 Web 服务 (server.py)

用于对接前端或进行实时调用。

    功能特点：
        提供 POST /api/translate 接口。
        模拟数据库查词功能（将 Gloss 词转换为 ID）。
        返回结构化的 JSON 数据。

    运行：
    bash

    python server.py

默认监听：http://127.0.0.1:5000

API 调用示例：
bash

curl -X POST http://127.0.0.1:5000/api/translate \
     -H "Content-Type: application/json" \
     -d '{"text": "直至下午5時，錄得氣溫30度。"}'

3. 快速单模型测试 (api_test.py)

用于调试 API 连接或快速查看单个模型的翻译效果。

    运行：
    bash

    python api_test.py

    输出：
    生成 weather_result_xxxx.xlsx 或 .csv。

