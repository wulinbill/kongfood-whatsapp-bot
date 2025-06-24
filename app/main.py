#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude API 客户端 (增强版)
与 Anthropic Claude API 交互，支持 Claude Sonnet 4
"""

import os
import json
import asyncio
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from ..config import settings
from ..logger import logger

@dataclass
class ClaudeMessage:
    """Claude消息结构"""
    role: str  # "user" or "assistant"
    content: str

@dataclass
class ClaudeResponse:
    """Claude响应结构"""
    content: str
    usage: Dict[str, int]
    model: str
    stop_reason: Optional[str] = None

class ClaudeAPIClient:
    """Claude API 客户端"""
    
    def __init__(self):
        self.api_key = settings.ANTHROPIC_API_KEY
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.api_version = "2023-06-01"
        
        # 模型配置
        self.available_models = {
            "claude-3-5-sonnet-20241022": {
                "max_tokens": 8192,
                "context_window": 200000,
                "description": "Most capable model for complex tasks"
            },
            "claude-3-5-haiku-20241022": {
                "max_tokens": 8192,
                "context_window": 200000,
                "description": "Fastest model for simple tasks"
            },
            "claude-3-opus-20240229": {
                "max_tokens": 4096,
                "context_window": 200000,
                "description": "Most powerful model for complex reasoning"
            },
            "claude-3-sonnet-20240229": {
                "max_tokens": 4096,
                "context_window": 200000,
                "description": "Balanced model for most tasks"
            }
        }
        
        # 默认配置
        self.default_model = "claude-3-5-sonnet-20241022"
        self.default_max_tokens = 1500  # 适合WhatsApp消息的长度
        self.default_temperature = 0.7
        
        if not self.api_key:
            logger.warning("Anthropic API key not found in environment")
    
    def _prepare_headers(self) -> Dict[str, str]:
        """准备请求头"""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json"
        }
    
    def _prepare_messages(self, messages: List[ClaudeMessage], system_prompt: str = None) -> List[Dict[str, str]]:
        """准备消息格式"""
        formatted_messages = []
        
        # 如果有系统提示，添加到消息开头
        if system_prompt:
            formatted_messages.append({
                "role": "user",
                "content": f"System: {system_prompt}\n\nUser: {messages[0].content if messages else ''}"
            })
            # 如果有多条消息，添加剩余的
            for msg in messages[1:]:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        else:
            # 直接转换消息
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return formatted_messages
    
    async def _make_request(
        self, 
        messages: List[ClaudeMessage], 
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: str = None,
        **kwargs
    ) -> Optional[ClaudeResponse]:
        """发起API请求"""
        if not self.api_key:
            logger.error("No API key available for Claude")
            return None
        
        headers = self._prepare_headers()
        formatted_messages = self._prepare_messages(messages, system_prompt)
        
        # 使用默认值
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": formatted_messages,
            "temperature": temperature,
            **kwargs
        }
        
        # 如果有独立的系统提示，使用新的API格式
        if system_prompt and not formatted_messages[0]["content"].startswith("System:"):
            payload["system"] = system_prompt
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 解析响应
                    content_blocks = result.get('content', [])
                    if content_blocks and len(content_blocks) > 0:
                        content = content_blocks[0].get('text', '').strip()
                        
                        return ClaudeResponse(
                            content=content,
                            usage=result.get('usage', {}),
                            model=result.get('model', model),
                            stop_reason=result.get('stop_reason')
                        )
                elif response.status_code == 429:
                    logger.warning("Claude API rate limit exceeded")
                    return None
                elif response.status_code == 400:
                    logger.error(f"Claude API bad request: {response.text}")
                    return None
                else:
                    logger.error(f"Claude API error: {response.status_code} - {response.text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("Claude API request timeout")
            return None
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return None
    
    def ask_claude_sync(
        self, 
        prompt: str, 
        system_prompt: str = None,
        model: str = None,
        **kwargs
    ) -> Optional[str]:
        """同步调用Claude API"""
        try:
            messages = [ClaudeMessage(role="user", content=prompt)]
            
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在事件循环中，需要在新线程中运行
                import concurrent.futures
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(
                            self._make_request(messages, model, system_prompt=system_prompt, **kwargs)
                        )
                        return result.content if result else None
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result(timeout=35)
            else:
                # 直接在当前循环中运行
                result = loop.run_until_complete(
                    self._make_request(messages, model, system_prompt=system_prompt, **kwargs)
                )
                return result.content if result else None
        except Exception as e:
            logger.error(f"Error in sync Claude call: {e}")
            return None
    
    async def ask_claude_async(
        self, 
        prompt: str, 
        system_prompt: str = None,
        model: str = None,
        **kwargs
    ) -> Optional[str]:
        """异步调用Claude API"""
        messages = [ClaudeMessage(role="user", content=prompt)]
        result = await self._make_request(messages, model, system_prompt=system_prompt, **kwargs)
        return result.content if result else None
    
    async def conversation_async(
        self,
        messages: List[ClaudeMessage],
        system_prompt: str = None,
        model: str = None,
        **kwargs
    ) -> Optional[str]:
        """异步多轮对话"""
        result = await self._make_request(messages, model, system_prompt=system_prompt, **kwargs)
        return result.content if result else None
    
    def build_restaurant_context(self, co_data: Dict[str, Any], menu_candidates: str) -> str:
        """构建餐厅订单上下文"""
        language = co_data.get('language', 'es')
        intent = co_data.get('intent', 'inquiry')
        objects = co_data.get('objects', [])
        raw_text = co_data.get('raw_text', '')
        confidence = co_data.get('confidence', 0.0)
        
        context = f"""
CUSTOMER INPUT: "{raw_text}"

ANALYSIS RESULTS:
- Detected Language: {language}
- Intent: {intent}
- Confidence: {confidence:.2f}
- Extracted Objects: {len(objects)}

EXTRACTED OBJECTS:
{json.dumps(objects, indent=2, ensure_ascii=False)}

AVAILABLE MENU ITEMS:
{menu_candidates}

CONTEXT: You are an AI assistant for Kong Food restaurant. The customer has sent a message and we need to provide an appropriate response based on the analysis above.
"""
        return context
    
    def build_system_prompt(self, language: str = 'es', context: str = "restaurant") -> str:
        """构建系统提示"""
        
        if context == "restaurant":
            if language == 'zh':
                return """
你是Kong Food餐厅的AI订餐助手。请遵循以下规则：

**核心原则：**
1. 语言：始终用中文回复
2. 角色：友好、专业的餐厅服务员
3. 流程：问候 → 点餐 → 澄清 → 确认 → 下单
4. 菜单约束：只推荐提供的菜单中的菜品，绝不虚构
5. 简洁回复：保持回复简洁（少于200字）

**对话策略：**
- 如需澄清，询问具体选项
- 主动提供帮助和建议
- 保持自然、热情的语调
- 使用适当的餐厅用语

**重要提醒：**
你代表Kong Food餐厅为客户提供优质服务。准确性和客户满意度是第一位的。
"""
            elif language == 'en':
                return """
You are Kong Food restaurant's AI ordering assistant. Follow these rules:

**Core Principles:**
1. Language: Always respond in English
2. Role: Friendly, professional restaurant server
3. Flow: Greeting → Order Taking → Clarification → Confirmation → Ordering
4. Menu Constraint: Only suggest items from the provided menu, never hallucinate
5. Concise Responses: Keep responses under 200 words

**Conversation Strategy:**
- If clarification needed, ask for specific options
- Proactively offer help and suggestions
- Maintain natural and enthusiastic tone
- Use appropriate restaurant terminology

**Important:**
You represent Kong Food restaurant providing excellent customer service. Accuracy and customer satisfaction are paramount.
"""
            else:  # Spanish default
                return """
Eres el asistente de IA para pedidos del restaurante Kong Food. Sigue estas reglas:

**Principios Fundamentales:**
1. Idioma: Siempre responde en español
2. Rol: Mesero amigable y profesional
3. Flujo: Saludo → Tomar Pedido → Aclaración → Confirmación → Ordenar
4. Restricción de Menú: Solo sugiere elementos del menú proporcionado, nunca inventes
5. Respuestas Concisas: Mantén respuestas menores a 200 palabras

**Estrategia de Conversación:**
- Si necesitas aclaración, pregunta por opciones específicas
- Ofrece ayuda y sugerencias proactivamente
- Mantén un tono natural y entusiasta
- Usa terminología apropiada de restaurante

**Importante:**
Representas al restaurante Kong Food brindando excelente servicio al cliente. La precisión y satisfacción del cliente son primordiales.
"""
        
        elif context == "clarification":
            if language == 'zh':
                return "你是Kong Food餐厅的AI助手。请帮助澄清客户的订单，提供清晰的选项让客户选择。"
            elif language == 'en':
                return "You are Kong Food restaurant's AI assistant. Help clarify the customer's order by providing clear options for them to choose from."
            else:
                return "Eres el asistente de IA del restaurante Kong Food. Ayuda a aclarar el pedido del cliente proporcionando opciones claras para elegir."
        
        else:
            # 通用系统提示
            return "You are a helpful AI assistant for Kong Food restaurant. Provide accurate and helpful responses to customer inquiries."
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """获取模型信息"""
        if model_name:
            return self.available_models.get(model_name, {})
        return self.available_models
    
    def estimate_tokens(self, text: str) -> int:
        """估算token数量 (粗略估算)"""
        # 粗略估算：平均4个字符 = 1个token
        return len(text) // 4
    
    def optimize_for_whatsapp(self, text: str, max_length: int = 1600) -> str:
        """优化WhatsApp消息长度"""
        if len(text) <= max_length:
            return text
        
        # 尝试在句子边界截断
        sentences = text.split('.')
        result = ""
        
        for sentence in sentences:
            if len(result + sentence + ".") <= max_length:
                result += sentence + "."
            else:
                break
        
        if result:
            return result.strip()
        
        # 如果没有合适的句子边界，直接截断
        return text[:max_length-3] + "..."

# 全局客户端实例
_global_client = ClaudeAPIClient()

def ask_claude(
    prompt: str, 
    system_prompt: str = None, 
    model: str = None,
    **kwargs
) -> Optional[str]:
    """
    外部调用接口 - 同步调用Claude
    
    Args:
        prompt: 用户提示
        system_prompt: 系统提示
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        Claude的响应文本
    """
    return _global_client.ask_claude_sync(prompt, system_prompt, model, **kwargs)

async def ask_claude_async(
    prompt: str, 
    system_prompt: str = None, 
    model: str = None,
    **kwargs
) -> Optional[str]:
    """
    外部调用接口 - 异步调用Claude
    
    Args:
        prompt: 用户提示
        system_prompt: 系统提示
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        Claude的响应文本
    """
    return await _global_client.ask_claude_async(prompt, system_prompt, model, **kwargs)

async def conversation_async(
    messages: List[ClaudeMessage],
    system_prompt: str = None,
    model: str = None,
    **kwargs
) -> Optional[str]:
    """
    外部调用接口 - 异步多轮对话
    
    Args:
        messages: 消息列表
        system_prompt: 系统提示
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        Claude的响应文本
    """
    return await _global_client.conversation_async(messages, system_prompt, model, **kwargs)

def build_restaurant_prompt(co_data: Dict[str, Any], menu_candidates: str, language: str = None) -> Tuple[str, str]:
    """构建餐厅订单提示"""
    language = language or co_data.get('language', 'es')
    system_prompt = _global_client.build_system_prompt(language, "restaurant")
    context = _global_client.build_restaurant_context(co_data, menu_candidates)
    
    return context, system_prompt

def test_claude_connection() -> bool:
    """测试Claude连接"""
    try:
        response = ask_claude(
            "Please respond with exactly 'Connection successful' to test the API.", 
            model="claude-3-5-haiku-20241022"  # 使用最快的模型测试
        )
        return response is not None and "successful" in response.lower()
    except Exception as e:
        logger.error(f"Claude connection test failed: {e}")
        return False

def get_available_models() -> Dict[str, Any]:
    """获取可用模型列表"""
    return _global_client.get_model_info()

def optimize_for_whatsapp(text: str, max_length: int = 1600) -> str:
    """优化WhatsApp消息长度"""
    return _global_client.optimize_for_whatsapp(text, max_length)

# 向后兼容
def get_client():
    """获取客户端实例"""
    return _global_client


# 测试函数
if __name__ == "__main__":
    import asyncio
    
    async def test_claude():
        """测试Claude客户端"""
        print("Testing Enhanced Claude API Client...")
        
        # 测试连接
        connection_ok = test_claude_connection()
        print(f"Connection test: {'Passed' if connection_ok else 'Failed'}")
        
        # 测试可用模型
        models = get_available_models()
        print(f
