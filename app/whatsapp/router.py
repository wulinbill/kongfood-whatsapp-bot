#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WhatsApp 消息路由器
处理WhatsApp消息并协调各个组件
"""

import uuid
import asyncio
from typing import Dict, Any, Optional
from ..config import settings
from .twilio_adapter import send_message as twilio_send
from .dialog360_adapter import send_message as dialog_send
from ..oco_core.seed_parser import parse
from ..oco_core.jump_planner import plan
from ..oco_core.tension_eval import score
from ..oco_core.output_director import reply
from ..speech.deepgram_client import transcribe_whatsapp
from ..logger import logger

class WhatsAppMessageProcessor:
    """WhatsApp消息处理器"""
    
    def __init__(self):
        self.session_store = {}  # 简单的会话存储
        
    def _get_session_id(self, phone_number: str) -> str:
        """获取或创建会话ID"""
        if phone_number not in self.session_store:
            self.session_store[phone_number] = {
                'session_id': str(uuid.uuid4())[:8],
                'conversation_state': 'greeting',
                'current_order': [],
                'customer_name': None
            }
        return self.session_store[phone_number]['session_id']
    
    def _update_session_state(self, phone_number: str, **kwargs):
        """更新会话状态"""
        if phone_number in self.session_store:
            self.session_store[phone_number].update(kwargs)
    
    def _get_send_function(self):
        """获取发送消息函数"""
        if settings.WHATSAPP_VENDOR == 'twilio':
            return twilio_send
        else:
            return dialog_send
    
    def _extract_phone_number(self, payload: Dict[str, Any]) -> Optional[str]:
        """提取电话号码"""
        # Twilio format
        phone = payload.get('From')
        if phone:
            return phone
        
        # 360Dialog format
        if 'contacts' in payload and len(payload['contacts']) > 0:
            return payload['contacts'][0].get('wa_id')
        
        # 通用格式
        if 'phone' in payload:
            return payload['phone']
        
        return None
    
    def _extract_message_content(self, payload: Dict[str, Any]) -> tuple[str, str, Optional[str]]:
        """
        提取消息内容
        
        Returns:
            (text, message_type, media_url)
        """
        # Twilio format
        text = payload.get('Body', '').strip()
        media_url = payload.get('MediaUrl0')
        
        if media_url:
            return text, 'audio' if 'audio' in payload.get('MediaContentType0', '') else 'media', media_url
        
        # 360Dialog format
        if 'messages' in payload and len(payload['messages']) > 0:
            message = payload['messages'][0]
            
            if message.get('type') == 'text':
                text = message.get('text', {}).get('body', '').strip()
                return text, 'text', None
            elif message.get('type') == 'audio':
                audio_data = message.get('audio', {})
                media_url = audio_data.get('url') or audio_data.get('link')
                return '', 'audio', media_url
        
        return text, 'text', None
    
    async def process_message_async(self, payload: Dict[str, Any], trace_id: str):
        """异步处理消息"""
        try:
            # 提取基本信息
            phone_number = self._extract_phone_number(payload)
            if not phone_number:
                logger.error("No phone number found in payload", extra={'trace_id': trace_id})
                return
            
            text, message_type, media_url = self._extract_message_content(payload)
            session_id = self._get_session_id(phone_number)
            
            logger.info(f"Processing message from {phone_number}: type={message_type}", 
                       extra={'trace_id': trace_id})
            
            # 处理语音消息
            if message_type == 'audio' and media_url:
                text, detected_language = transcribe_whatsapp(media_url)
                if not text:
                    send_function = self._get_send_function()
                    send_function(phone_number, "Lo siento, no pude entender el audio. ¿Podrías escribir tu orden?")
                    return
                
                logger.info(f"Transcribed audio: '{text}' (language: {detected_language})", 
                           extra={'trace_id': trace_id})
            
            # 如果没有文本内容，返回
            if not text:
                return
            
            # 解析用户输入
            co = parse(text)
            co['customer_phone'] = phone_number
            
            # 获取会话状态
            session = self.session_store.get(phone_number, {})
            if session.get('customer_name'):
                co['customer_name'] = session['customer_name']
            
            logger.info(f"Parsed CO: intent={co.get('intent')}, confidence={co.get('confidence')}", 
                       extra={'trace_id': trace_id})
            
            # 规划路径
            path_data = plan(co)
            
            logger.info(f"Path planning: score={path_data.get('score')}, requires_clarification={path_data.get('requires_clarification')}", 
                       extra={'trace_id': trace_id})
            
            # 生成响应
            response = reply(co, path_data, session_id)
            
            # 发送响应
            send_function = self._get_send_function()
            send_function(phone_number, response)
            
            logger.info(f"Response sent to {phone_number}", extra={'trace_id': trace_id})
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", extra={'trace_id': trace_id})
            # 发送错误响应
            try:
                phone_number = self._extract_phone_number(payload)
                if phone_number:
                    send_function = self._get_send_function()
                    send_function(phone_number, "Disculpa, hubo un error. ¿Podrías intentar de nuevo?")
            except:
                pass
    
    def process_message_sync(self, payload: Dict[str, Any], trace_id: str):
        """同步处理消息（用于后台任务）"""
        try:
            # 在新的事件循环中运行异步处理
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.process_message_async(payload, trace_id))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in sync message processing: {e}", extra={'trace_id': trace_id})

# 全局处理器实例
_global_processor = WhatsAppMessageProcessor()

def handle_whatsapp_event(payload: Dict[str, Any], trace_id: str):
    """
    处理WhatsApp事件 - 外部调用接口
    
    Args:
        payload: WhatsApp webhook负载
        trace_id: 追踪ID
    """
    logger.info("Handling WhatsApp event", extra={'trace_id': trace_id})
    
    # 检查是否是状态更新或其他非消息事件
    if _is_status_update(payload):
        logger.info("Ignoring status update", extra={'trace_id': trace_id})
        return
    
    # 处理消息
    _global_processor.process_message_sync(payload, trace_id)

def _is_status_update(payload: Dict[str, Any]) -> bool:
    """检查是否是状态更新"""
    # Twilio状态更新通常包含MessageStatus
    if 'MessageStatus' in payload:
        return True
    
    # 360Dialog状态更新
    if 'statuses' in payload:
        return True
    
    # 没有消息内容的payload
    text = payload.get('Body', '').strip()
    if not text and 'MediaUrl0' not in payload:
        # 检查360Dialog格式
        if 'messages' not in payload or len(payload.get('messages', [])) == 0:
            return True
    
    return False

async def handle_whatsapp_event_async(payload: Dict[str, Any], trace_id: str):
    """异步处理WhatsApp事件"""
    if _is_status_update(payload):
        return
    
    await _global_processor.process_message_async(payload, trace_id)

def get_session_info(phone_number: str) -> Dict[str, Any]:
    """获取会话信息"""
    return _global_processor.session_store.get(phone_number, {})

def update_session_info(phone_number: str, **kwargs):
    """更新会话信息"""
    _global_processor._update_session_state(phone_number, **kwargs)

def clear_session(phone_number: str):
    """清除会话"""
    if phone_number in _global_processor.session_store:
        del _global_processor.session_store[phone_number]


# 测试函数
if __name__ == "__main__":
    # 测试消息处理
    test_payload = {
        'From': 'whatsapp:+1234567890',
        'Body': 'Quiero pollo teriyaki',
        'MessageSid': 'test123'
    }
    
    test_trace_id = str(uuid.uuid4())
    
    print("Testing WhatsApp message processing...")
    handle_whatsapp_event(test_payload, test_trace_id)
    print("Test completed")
