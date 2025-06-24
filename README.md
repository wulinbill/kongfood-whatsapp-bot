# Kong Food WhatsApp AI Bot

🤖 智能餐厅订餐机器人，支持多语言语音/文本订单处理，集成 Loyverse POS 系统。

## 🌟 核心特性

- **多语言支持**: 西班牙语、英语、中文无缝切换
- **语音识别**: Deepgram 驱动的高精度语音转文本
- **智能解析**: O_co MicroCore 自然语言理解引擎
- **菜单匹配**: 模糊匹配 + 张力评估自学习系统
- **POS 集成**: Loyverse OAuth2 自动下单
- **WhatsApp**: Twilio/360Dialog 双通道支持

## 🏗 架构设计

```
WhatsApp → Speech-to-Text → O_co Engine → POS System
    ↓           ↓               ↓            ↓
 Twilio    Deepgram      AI Parser    Loyverse
360Dialog              Claude LLM    
```

### O_co MicroCore 流程
```
User Input → seed_parser → jump_planner → tension_eval → output_director
               ↓              ↓             ↓              ↓
           CO Object     Menu Matches   Action Score   Reply/Execute
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/your-repo/kongfood-whatsapp-bot.git
cd kongfood-whatsapp-bot

# 复制环境变量
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
nano .env
```

### 2. 必需的 API 密钥

```env
# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# Deepgram 语音识别
DEEPGRAM_API_KEY=...

# Twilio WhatsApp
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...

# Loyverse POS
LOYVERSE_CLIENT_ID=...
LOYVERSE_CLIENT_SECRET=...
```

### 3. Docker 部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 4. 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📱 WhatsApp 配置

### Twilio 设置

1. 访问 [Twilio Console](https://console.twilio.com/)
2. 购买电话号码并启用 WhatsApp
3. 设置 Webhook URL: `https://your-domain.com/webhook/whatsapp`
4. 获取 Account SID 和 Auth Token

### 360Dialog 设置

1. 注册 [360Dialog](https://www.360dialog.com/)
2. 获取 API Key
3. 配置 Webhook URL
4. 设置 `WHATSAPP_VENDOR=360dialog`

## 🍽 菜单配置

### 添加新门店

1. 创建菜单知识库文件:
```bash
cp app/knowledge_base/kong-food-store-1_menu_kb.json \
   app/knowledge_base/new-store-id_menu_kb.json
```

2. 编辑菜单内容:
```json
{
  "restaurant_info": {
    "name": "New Restaurant",
    "default_store_id": "new-store-id"
  },
  "menu_categories": {
    "combinaciones": {
      "items": [
        {
          "item_id": "...",
          "item_name": "New Dish",
          "price": 12.99,
          "aliases": ["Alternative Name"],
          "keywords": ["search", "terms"]
        }
      ]
    }
  }
}
```

3. 更新环境变量:
```env
STORE_ID=new-store-id
MENU_KB_FILE=app/knowledge_base/new-store-id_menu_kb.json
```

## 🔧 Loyverse POS 集成

### OAuth2 设置

1. 在 Loyverse 开发者中心创建应用
2. 配置回调 URL: `https://your-domain.com/oauth/callback`
3. 获取授权链接:
```python
from app.pos.loyverse_client import get_authorization_url
print(get_authorization_url())
```
4. 完成授权后，refresh_token 自动保存

### 测试订单

```python
from app.pos.loyverse_client import place_order_async

path_data = {
    'path': [
        {
            'variant_id': 'your-variant-id',
            'price': 11.99,
            'quantity': 1
        }
    ]
}

result = await place_order_async(
    path_data,
    customer_phone="+1234567890",
    customer_name="Test Customer"
)

print(result)
```

## 🧪 测试

### 运行所有测试
```bash
pytest app/tests/ -v
```

### 测试特定功能
```bash
# 语音解析测试
pytest app/tests/test_integration.py::TestSpeechTranscription -v

# 订单流程测试
pytest app/tests/test_integration.py::TestEndToEndScenarios -v

# 性能测试
pytest app/tests/test_integration.py::TestPerformanceAndReliability -v
```

### 手动测试 WhatsApp

1. 发送消息到你的 WhatsApp 号码
2. 查看实时日志:
```bash
docker-compose logs -f bot
```

## 📊 监控和调试

### 日志追踪

每个请求都有唯一的 `trace_id`，便于问题排查:

```bash
# 搜索特定会话
grep "trace_id_abc123" logs/app.log

# 实时监控
tail -f logs/app.log | grep ERROR
```

### 张力评估监控

系统会自动学习和调整决策阈值，数据存储在:
- `/mnt/data/tension_history.json` - 历史记录
- `/mnt/data/tension_metrics.json` - 当前指标

### 健康检查

```bash
curl http://localhost:8000/
# 返回: {"status": "live"}
```

## 🔄 对话流程

### 标准订餐流程

1. **问候**: "Hola, restaurante Kong Food. ¿Qué desea ordenar hoy?"
2. **捕获菜品**: 识别和匹配菜单项
3. **澄清**: 如需要，询问具体选项
4. **客户信息**: 首次订餐时询问姓名
5. **POS 注册**: 自动提交到 Loyverse
6. **最终确认**: 显示订单详情和准备时间
7. **结束**: "¡Muchas gracias!"

### 澄清场景

- **多重匹配**: "Encontré varias opciones para 'pollo'..."
- **低置信度**: "¿Te refieres a alguno de estos platos?"
- **无匹配**: "No encontré ese plato en nuestro menú..."

## 🌍 多语言示例

### 西班牙语
```
用户: "Quiero 2 pollo teriyaki sin salsa"
机器人: "Perfecto, 2x Pollo Teriyaki sin salsa. ¿Algo más?"
```

### 英语
```
用户: "I want chicken teriyaki, no sauce"
机器人: "Perfect, Chicken Teriyaki without sauce. Anything else?"
```

### 中文
```
用户: "我要照烧鸡肉，不要酱"
机器人: "好的，照烧鸡肉不要酱。还要别的吗？"
```

## 🚨 故障排除

### 常见问题

1. **Claude API 错误**
```bash
# 检查 API 密钥
echo $ANTHROPIC_API_KEY

# 测试连接
python -c "from app.llm.claude_client import test_claude_connection; print(test_claude_connection())"
```

2. **Deepgram 语音识别失败**
```bash
# 检查音频 URL 是否可访问
curl -I $AUDIO_URL

# 验证 API 密钥
python -c "from app.speech.deepgram_client import transcribe; print(transcribe('test_url'))"
```

3. **Loyverse 订单失败**
```bash
# 检查 OAuth token
python -c "from app.pos.loyverse_client import _client; print(asyncio.run(_client.auth_manager.get_valid_access_token()))"

# 测试 API 连接
python -c "from app.pos.loyverse_client import get_stores_async; print(asyncio.run(get_stores_async()))"
```

### 性能优化

- **并发处理**: 使用 `asyncio` 处理多个请求
- **缓存策略**: 菜单数据本地缓存，减少 API 调用
- **张力学习**: 系统自动优化决策阈值

## 📈 扩展开发

### 添加新语言

1. 更新 `seed_parser.py` 中的语言检测
2. 在 `clarify_engine.py` 添加新语言模板
3. 更新 `claude_client.py` 的语言映射

### 集成其他 POS 系统

1. 实现新的 POS 客户端类
2. 遵循 `loyverse_client.py` 的接口标准
3. 更新配置选择 POS 提供商

### 自定义修改规则

编辑 `menu_kb.json` 中的 `ai_parsing_rules`:
```json
{
  "natural_language_patterns": {
    "custom_patterns": [
      "my_pattern {ingredient}",
      "another_pattern"
    ]
  }
}
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Pull Request 和 Issue！

1. Fork 项目
2. 创建特性分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 提交 Pull Request

## 📞 支持

- 📧 邮箱: support@kongfood.com
- 📱 WhatsApp: +1 (787) 123-4567
- 🌐 网站: https://kongfood.com

---

**Built with ❤️ for Kong Food Restaurant**
