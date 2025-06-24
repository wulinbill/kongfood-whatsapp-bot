#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude API 客户端
与 Anthropic Claude API 交互
"""

import os
import json
import asyncio
import httpx
from typing import Dict, Any, Optional, List
from ..config import settings
from ..logger import logger

class ClaudeClient:
    """Claude API 客户端"""
    
    def __init__(self):
        self.api_key = settings.ANTHROPIC_API_KEY
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.api_version = "2023-06-01"
        self.max_tokens = 1500  # 适合WhatsApp消息的长度
        
        if not self.api_key:
            logger.warning("Anthropic API key not found in environment")
    
    def _prepare_headers(self) -> Dict[str, str]:
        """准备请求头"""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json"
        }
    
    def _prepare_messages(self, prompt: str, system_prompt: str = None) -> List[Dict[str, str]]:
        """准备消息格式"""
        messages = []
        
        if system_prompt:
            # Claude API 使用系统消息和用户消息分离
            messages.append({
                "role": "user",
                "content": f"System instructions: {system_prompt}\n\nUser message: {prompt}"
            })
        else:
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        return messages
    
    async def _make_request(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """发起API请求"""
        if not self.api_key:
            logger.error("No API key available for Claude")
            return None
        
        headers = self._prepare_headers()
        messages = self._prepare_messages(prompt, system_prompt)
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": self.max_tokens,
            "messages": messages
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('content', [])
                    if content and len(content) > 0:
                        return content[0].get('text', '').strip()
                else:
                    logger.error(f"Claude API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return None
    
    def ask_claude_sync(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """同步调用Claude API"""
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在事件循环中，需要在新线程中运行
                import concurrent.futures
                import threading
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self._make_request(prompt, system_prompt)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result(timeout=35)
            else:
                # 直接在当前循环中运行
                return loop.run_until_complete(
                    self._make_request(prompt, system_prompt)
                )
        except Exception as e:
            logger.error(f"Error in sync Claude call: {e}")
            return None
    
    async def ask_claude_async(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """异步调用Claude API"""
        return await self._make_request(prompt, system_prompt)
    
    def build_order_context(self, co_data: Dict[str, Any], menu_candidates: str) -> str:
        """构建订单上下文"""
        language = co_data.get('language', 'es')
        intent = co_data.get('intent', 'inquiry')
        objects = co_data.get('objects', [])
        raw_text = co_data.get('raw_text', '')
        
        context = f"""
User Input: "{raw_text}"
Detected Language: {language}
Intent: {intent}
Confidence: {co_data.get('confidence', 0.0)}

Extracted Objects:
{json.dumps(objects, indent=2, ensure_ascii=False)}

Available Menu Items:
{menu_candidates}

Please respond naturally as a Kong Food restaurant assistant in {language} language.
"""
        return context
    
    def build_system_prompt(self, language: str = 'es') -> str:
        """构建系统提示"""
        if language == 'zh':
            return """
你是Kong Food餐厅的AI订餐助手。请遵循以下规则：

1. 语言：用中文回复
2. 角色：友好、专业的餐厅服务员
3. 流程：问候 → 点餐 → 澄清 → 确认 → 下单
4. 只推荐菜单中的菜品
5. 保持回复简洁（少于200字）
6. 如需澄清，询问具体选项
7. 对话要自然、热情

记住：你代表Kong Food餐厅为客户提供优质服务。
"""
        elif language == 'en':
            return """
You are Kong Food restaurant's AI ordering assistant. Follow these rules:

1. Language: Respond in English
2. Role: Friendly, professional restaurant server
3. Flow: Greeting → Order Taking → Clarification → Confirmation → Ordering
4. Only suggest items from the provided menu
5. Keep responses concise (under 200 words)
6. If clarification needed, ask for specific options
7. Be natural and enthusiastic

Remember: You represent Kong Food restaurant providing excellent customer service.
"""
        else:  # Spanish default
            return """
Eres el asistente de IA para pedidos del restaurante Kong Food. Sigue estas reglas:

1. Idioma: Responde en español
2. Rol: Mesero amigable y profesional
3. Flujo: Saludo → Tomar Pedido → Aclaración → Confirmación → Ordenar
4. Solo sugiere elementos del menú proporcionado
5. Mantén respuestas concisas (menos de 200 palabras)
6. Si necesitas aclaración, pregunta por opciones específicas
7. Sé natural y entusiasta

Recuerda: Representas al restaurante Kong Food brindando excelente servicio al cliente.
"""

# 全局客户端实例
_global_client = ClaudeClient()

def ask_claude(prompt: str, system_prompt: str = None) -> Optional[str]:
    """
    外部调用接口 - 同步调用Claude
    
    Args:
        prompt: 用户提示
        system_prompt: 系统提示
        
    Returns:
        Claude的响应文本
    """
    return _global_client.ask_claude_sync(prompt, system_prompt)

async def ask_claude_async(prompt: str, system_prompt: str = None) -> Optional[str]:
    """
    外部调用接口 - 异步调用Claude
    
    Args:
        prompt: 用户提示
        system_prompt: 系统提示
        
    Returns:
        Claude的响应文本
    """
    return await _global_client.ask_claude_async(prompt, system_prompt)

def build_restaurant_prompt(co_data: Dict[str, Any], menu_candidates: str) -> str:
    """构建餐厅订单提示"""
    language = co_data.get('language', 'es')
    system_prompt = _global_client.build_system_prompt(language)
    context = _global_client.build_order_context(co_data, menu_candidates)
    
    return system_prompt + "\n\n" + context

def test_claude_connection() -> bool:
    """测试Claude连接"""
    try:
        response = ask_claude("Hello, please respond with 'Connection successful'")
        return response is not None and "successful" in response.lower()
    except Exception as e:
        logger.error(f"Claude connection test failed: {e}")
        return False


# 测试函数
if __name__ == "__main__":
    import asyncio
    
    async def test_claude():
        """测试Claude客户端"""
        print("Testing Claude API Client...")
        
        # 测试同步调用
        response = ask_claude("Hello, how are you?")
        print(f"Sync response: {response}")
        
        # 测试异步调用
        async_response = await ask_claude_async("What's the weather like today?")
        print(f"Async response: {async_response}")
        
        # 测试连接
        connection_ok = test_claude_connection()
        print(f"Connection test: {'Passed' if connection_ok else 'Failed'}")
    
    asyncio.run(test_claude())
