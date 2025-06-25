#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版 WhatsApp 消息路由器
支持 Redis 会话存储、消息去重、错误恢复和多轮对话状态管理
"""

import uuid
import json
import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis.asyncio as redis
from contextlib import asynccontextmanager
import backoff

from ..config import settings
from .twilio_adapter import send_message as twilio_send
from .dialog360_adapter import send_message as dialog_send
from ..oco_core.seed_parser import parse
from ..oco_core.jump_planner import plan
from ..oco_core.tension_eval import score
from ..oco_core.output_director import reply
from ..speech.deepgram_client import transcribe_whatsapp
from ..logger import logger

class ConversationState(Enum):
    """对话状态枚举"""
    GREETING = "greeting"
    MENU_BROWSING = "menu_browsing"
    ORDERING = "ordering"
    ORDER_CONFIRMATION = "order_confirmation"
    PAYMENT = "payment"
    ORDER_COMPLETE = "order_complete"
    HELP = "help"
    ERROR_RECOVERY = "error_recovery"
    IDLE = "idle"

class MessageType(Enum):
    """消息类型"""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    LOCATION = "location"
    CONTACT = "contact"

@dataclass
class MessageContext:
    """消息上下文"""
    message_id: str
    phone_number: str
    message_type: MessageType
    content: str
    media_url: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False
    retry_count: int = 0
    error_message: Optional[str] = None

@dataclass
class ConversationSession:
    """对话会话"""
    session_id: str
    phone_number: str
    state: ConversationState = ConversationState.GREETING
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    current_order: List[Dict[str, Any]] = field(default_factory=list)
    order_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    context_stack: List[str] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    total_messages: int = 0
    error_count: int = 0
    language: str = "es"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换datetime对象为ISO字符串
        data['last_activity'] = self.last_activity.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['state'] = self.state.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """从字典创建"""
        # 转换日期字符串
        if isinstance(data.get('last_activity'), str):
            data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # 转换状态枚举
        if isinstance(data.get('state'), str):
            data['state'] = ConversationState(data['state'])
        
        return cls(**data)
    
    def update_activity(self):
        """更新活动时间"""
        self.last_activity = datetime.now()
        self.total_messages += 1
    
    def push_context(self, context: str):
        """推入上下文"""
        self.context_stack.append(context)
        # 限制上下文栈大小
        if len(self.context_stack) > 10:
            self.context_stack.pop(0)
    
    def pop_context(self) -> Optional[str]:
        """弹出上下文"""
        return self.context_stack.pop() if self.context_stack else None
    
    def get_current_context(self) -> Optional[str]:
        """获取当前上下文"""
        return self.context_stack[-1] if self.context_stack else None

class SessionManager:
    """会话管理器"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL or "redis://localhost:6379"
        self.redis_client: Optional[redis.Redis] = None
        self.session_timeout = timedelta(hours=24)  # 会话超时时间
        self.memory_fallback: Dict[str, ConversationSession] = {}  # 内存备份
        
    async def _get_redis_client(self) -> redis.Redis:
        """获取Redis客户端"""
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory fallback")
                self.redis_client = None
        return self.redis_client
    
    async def get_session(self, phone_number: str) -> ConversationSession:
        """获取或创建会话"""
        session_key = f"whatsapp_session:{phone_number}"
        
        try:
            redis_client = await self._get_redis_client()
            if redis_client:
                session_data = await redis_client.get(session_key)
                if session_data:
                    data = json.loads(session_data)
                    session = ConversationSession.from_dict(data)
                    
                    # 检查会话是否过期
                    if datetime.now() - session.last_activity > self.session_timeout:
                        logger.info(f"Session expired for {phone_number}, creating new one")
                        session = self._create_new_session(phone_number)
                    else:
                        logger.info(f"Retrieved session for {phone_number}")
                    
                    return session
        except Exception as e:
            logger.error(f"Error retrieving session from Redis: {e}")
        
        # 尝试从内存备份获取
        if phone_number in self.memory_fallback:
            session = self.memory_fallback[phone_number]
            if datetime.now() - session.last_activity <= self.session_timeout:
                return session
        
        # 创建新会话
        session = self._create_new_session(phone_number)
        logger.info(f"Created new session for {phone_number}")
        return session
    
    def _create_new_session(self, phone_number: str) -> ConversationSession:
        """创建新会话"""
        return ConversationSession(
            session_id=str(uuid.uuid4())[:8],
            phone_number=phone_number
        )
    
    async def save_session(self, session: ConversationSession):
        """保存会话"""
        session_key = f"whatsapp_session:{session.phone_number}"
        session.update_activity()
        
        try:
            redis_client = await self._get_redis_client()
            if redis_client:
                session_data = json.dumps(session.to_dict())
                await redis_client.setex(
                    session_key, 
                    int(self.session_timeout.total_seconds()), 
                    session_data
                )
                logger.debug(f"Session saved to Redis for {session.phone_number}")
            else:
                # 保存到内存备份
                self.memory_fallback[session.phone_number] = session
                logger.debug(f"Session saved to memory for {session.phone_number}")
                
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            # 备份到内存
            self.memory_fallback[session.phone_number] = session
    
    async def delete_session(self, phone_number: str):
        """删除会话"""
        session_key = f"whatsapp_session:{phone_number}"
        
        try:
            redis_client = await self._get_redis_client()
            if redis_client:
                await redis_client.delete(session_key)
            
            self.memory_fallback.pop(phone_number, None)
            logger.info(f"Session deleted for {phone_number}")
            
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
    
    async def get_all_active_sessions(self) -> List[ConversationSession]:
        """获取所有活跃会话"""
        sessions = []
        
        try:
            redis_client = await self._get_redis_client()
            if redis_client:
                keys = await redis_client.keys("whatsapp_session:*")
                for key in keys:
                    session_data = await redis_client.get(key)
                    if session_data:
                        data = json.loads(session_data)
                        session = ConversationSession.from_dict(data)
                        sessions.append(session)
        except Exception as e:
            logger.error(f"Error retrieving active sessions: {e}")
        
        # 添加内存中的会话
        for session in self.memory_fallback.values():
            if datetime.now() - session.last_activity <= self.session_timeout:
                sessions.append(session)
        
        return sessions

class MessageDeduplicator:
    """消息去重器"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL or "redis://localhost:6379"
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, float] = {}  # 内存缓存 {hash: timestamp}
        self.cache_duration = 300  # 5分钟
    
    async def _get_redis_client(self) -> redis.Redis:
        """获取Redis客户端"""
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
            except Exception:
                self.redis_client = None
        return self.redis_client
    
    def _generate_message_hash(self, payload: Dict[str, Any]) -> str:
        """生成消息哈希"""
        # 提取关键字段
        key_fields = {
            'from': self._extract_phone_number(payload),
            'body': payload.get('Body', ''),
            'timestamp': payload.get('timestamp'),
            'message_id': payload.get('MessageSid') or payload.get('id')
        }
        
        # 如果没有时间戳，使用当前时间的分钟级别
        if not key_fields['timestamp']:
            key_fields['timestamp'] = int(time.time() / 60) * 60
        
        message_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.md5(message_str.encode()).hexdigest()
    
    def _extract_phone_number(self, payload: Dict[str, Any]) -> Optional[str]:
        """提取电话号码"""
        phone = payload.get('From')
        if phone:
            return phone
        
        if 'contacts' in payload and len(payload['contacts']) > 0:
            return payload['contacts'][0].get('wa_id')
        
        return payload.get('phone')
    
    async def is_duplicate(self, payload: Dict[str, Any]) -> bool:
        """检查是否是重复消息"""
        message_hash = self._generate_message_hash(payload)
        current_time = time.time()
        
        # 检查Redis缓存
        try:
            redis_client = await self._get_redis_client()
            if redis_client:
                exists = await redis_client.get(f"msg_hash:{message_hash}")
                if exists:
                    logger.info(f"Duplicate message detected (Redis): {message_hash}")
                    return True
                
                # 记录消息哈希
                await redis_client.setex(f"msg_hash:{message_hash}", self.cache_duration, "1")
                return False
        except Exception as e:
            logger.warning(f"Redis deduplication failed: {e}")
        
        # 使用内存缓存作为备份
        if message_hash in self.memory_cache:
            if current_time - self.memory_cache[message_hash] < self.cache_duration:
                logger.info(f"Duplicate message detected (memory): {message_hash}")
                return True
        
        # 清理过期的内存缓存
        expired_hashes = [
            h for h, t in self.memory_cache.items() 
            if current_time - t >= self.cache_duration
        ]
        for h in expired_hashes:
            del self.memory_cache[h]
        
        # 记录新消息
        self.memory_cache[message_hash] = current_time
        return False

class ErrorRecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self):
        self.error_patterns = {
            'parse_error': {
                'keywords': ['parse', 'invalid', 'format'],
                'recovery_action': 'ask_clarification',
                'message': "No entendí bien tu mensaje. ¿Podrías reformularlo?"
            },
            'service_error': {
                'keywords': ['connection', 'timeout', 'service'],
                'recovery_action': 'retry',
                'message': "Hay un problema temporal. Intentando de nuevo..."
            },
            'order_error': {
                'keywords': ['order', 'item', 'unavailable'],
                'recovery_action': 'suggest_alternative',
                'message': "Ese producto no está disponible. ¿Te gustaría ver otras opciones?"
            }
        }
    
    def categorize_error(self, error_message: str) -> str:
        """分类错误"""
        error_message_lower = error_message.lower()
        
        for error_type, config in self.error_patterns.items():
            if any(keyword in error_message_lower for keyword in config['keywords']):
                return error_type
        
        return 'unknown_error'
    
    def get_recovery_action(self, error_type: str) -> Dict[str, Any]:
        """获取恢复动作"""
        if error_type in self.error_patterns:
            return self.error_patterns[error_type]
        
        return {
            'recovery_action': 'generic_error',
            'message': "Ocurrió un error inesperado. ¿Podrías intentar de nuevo?"
        }
    
    async def handle_error(self, session: ConversationSession, error: Exception, context: str) -> str:
        """处理错误并返回恢复消息"""
        error_type = self.categorize_error(str(error))
        recovery_config = self.get_recovery_action(error_type)
        
        # 更新会话状态
        session.error_count += 1
        session.push_context(f"error:{error_type}:{context}")
        
        # 如果错误太多，切换到人工客服模式
        if session.error_count >= 3:
            session.state = ConversationState.HELP
            return "Parece que estamos teniendo dificultades. Un momento, te conectaré con un representante."
        
        # 根据错误类型调整状态
        if error_type == 'order_error':
            session.state = ConversationState.MENU_BROWSING
        elif session.state != ConversationState.ERROR_RECOVERY:
            session.state = ConversationState.ERROR_RECOVERY
        
        logger.warning(f"Error recovery triggered for {session.phone_number}: {error_type}")
        
        return recovery_config['message']

class EnhancedWhatsAppMessageProcessor:
    """增强版 WhatsApp 消息处理器"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.deduplicator = MessageDeduplicator()
        self.error_recovery = ErrorRecoveryManager()
        self.processing_lock = asyncio.Lock()
        
    def _get_send_function(self):
        """获取发送消息函数"""
        if settings.WHATSAPP_VENDOR == 'twilio':
            return twilio_send
        else:
            return dialog_send
    
    def _extract_phone_number(self, payload: Dict[str, Any]) -> Optional[str]:
        """提取电话号码"""
        phone = payload.get('From')
        if phone:
            return phone
        
        if 'contacts' in payload and len(payload['contacts']) > 0:
            return payload['contacts'][0].get('wa_id')
        
        return payload.get('phone')
    
    def _extract_message_content(self, payload: Dict[str, Any]) -> tuple[str, MessageType, Optional[str]]:
        """提取消息内容"""
        # Twilio format
        text = payload.get('Body', '').strip()
        media_url = payload.get('MediaUrl0')
        
        if media_url:
            content_type = payload.get('MediaContentType0', '')
            if 'audio' in content_type:
                return text, MessageType.AUDIO, media_url
            elif 'image' in content_type:
                return text, MessageType.IMAGE, media_url
            else:
                return text, MessageType.DOCUMENT, media_url
        
        # 360Dialog format
        if 'messages' in payload and len(payload['messages']) > 0:
            message = payload['messages'][0]
            
            if message.get('type') == 'text':
                text = message.get('text', {}).get('body', '').strip()
                return text, MessageType.TEXT, None
            elif message.get('type') == 'audio':
                audio_data = message.get('audio', {})
                media_url = audio_data.get('url') or audio_data.get('link')
                return '', MessageType.AUDIO, media_url
            elif message.get('type') == 'image':
                image_data = message.get('image', {})
                media_url = image_data.get('url') or image_data.get('link')
                caption = image_data.get('caption', '')
                return caption, MessageType.IMAGE, media_url
        
        return text, MessageType.TEXT, None
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def process_message_async(self, payload: Dict[str, Any], trace_id: str):
        """异步处理消息"""
        async with self.processing_lock:
            try:
                # 消息去重检查
                if await self.deduplicator.is_duplicate(payload):
                    logger.info("Duplicate message ignored", extra={'trace_id': trace_id})
                    return
                
                # 提取基本信息
                phone_number = self._extract_phone_number(payload)
                if not phone_number:
                    logger.error("No phone number found in payload", extra={'trace_id': trace_id})
                    return
                
                text, message_type, media_url = self._extract_message_content(payload)
                
                # 获取会话
                session = await self.session_manager.get_session(phone_number)
                
                logger.info(
                    f"Processing message from {phone_number}: type={message_type.value}, state={session.state.value}",
                    extra={'trace_id': trace_id}
                )
                
                # 创建消息上下文
                message_context = MessageContext(
                    message_id=payload.get('MessageSid') or payload.get('id') or str(uuid.uuid4()),
                    phone_number=phone_number,
                    message_type=message_type,
                    content=text,
                    media_url=media_url
                )
                
                # 处理不同类型的消息
                response = await self._process_by_type(message_context, session, trace_id)
                
                # 发送响应
                if response:
                    send_function = self._get_send_function()
                    success = send_function(phone_number, response)
                    
                    if success:
                        message_context.processed = True
                        logger.info(f"Response sent to {phone_number}", extra={'trace_id': trace_id})
                    else:
                        raise Exception("Failed to send response")
                
                # 保存会话
                await self.session_manager.save_session(session)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", extra={'trace_id': trace_id})
                await self._handle_processing_error(payload, e, trace_id)
    
    async def _process_by_type(self, message_ctx: MessageContext, session: ConversationSession, trace_id: str) -> Optional[str]:
        """根据消息类型处理"""
        try:
            text = message_ctx.content
            
            # 处理语音消息
            if message_ctx.message_type == MessageType.AUDIO and message_ctx.media_url:
                transcribed, detected_language = transcribe_whatsapp(message_ctx.media_url)
                if transcribed:
                    text = transcribed
                    session.language = detected_language or session.language
                    logger.info(f"Transcribed audio: '{text}' (language: {detected_language})", 
                               extra={'trace_id': trace_id})
                else:
                    return "Lo siento, no pude entender el audio. ¿Podrías escribir tu mensaje?"
            
            # 处理图片消息
            elif message_ctx.message_type == MessageType.IMAGE:
                # 这里可以添加图片识别逻辑
                if text:
                    # 图片有文字说明，正常处理
                    pass
                else:
                    return "He recibido tu imagen. ¿Podrías decirme cómo puedo ayudarte?"
            
            # 如果没有文本内容，请求澄清
            if not text:
                return "No entendí tu mensaje. ¿Podrías escribirlo de otra manera?"
            
            # 处理特殊命令
            if text.lower() in ['/start', 'start', 'hola', 'hello']:
                session.state = ConversationState.GREETING
                session.error_count = 0  # 重置错误计数
                return self._get_greeting_message(session)
            
            elif text.lower() in ['/help', 'help', 'ayuda']:
                session.state = ConversationState.HELP
                return self._get_help_message(session)
            
            elif text.lower() in ['/reset', 'reset', 'reiniciar']:
                await self.session_manager.delete_session(session.phone_number)
                return "Tu sesión ha sido reiniciada. ¡Hola! ¿En qué puedo ayudarte hoy?"
            
            # 根据当前状态处理消息
            return await self._process_by_state(text, session, trace_id)
            
        except Exception as e:
            error_msg = await self.error_recovery.handle_error(session, e, "message_processing")
            return error_msg
    
    async def _process_by_state(self, text: str, session: ConversationSession, trace_id: str) -> str:
        """根据对话状态处理消息"""
        try:
            # 解析用户输入
            co = parse(text)
            co['customer_phone'] = session.phone_number
            co['session_id'] = session.session_id
            
            # 添加会话上下文
            if session.customer_name:
                co['customer_name'] = session.customer_name
            
            # 添加对话历史上下文
            co['conversation_state'] = session.state.value
            co['current_context'] = session.get_current_context()
            
            logger.info(f"Parsed CO: intent={co.get('intent')}, confidence={co.get('confidence')}", 
                       extra={'trace_id': trace_id})
            
            # 根据状态调整解析结果
            if session.state == ConversationState.ERROR_RECOVERY:
                # 在错误恢复状态下，更宽松地处理用户输入
                co['recovery_mode'] = True
            
            # 规划路径
            path_data = plan(co)
            
            # 更新会话信息
            self._update_session_from_co(session, co, path_data)
            
            logger.info(f"Path planning: score={path_data.get('score')}, requires_clarification={path_data.get('requires_clarification')}", 
                       extra={'trace_id': trace_id})
            
            # 生成响应
            response = reply(co, path_data, session.session_id)
            
            # 更新对话状态
            self._update_conversation_state(session, co, path_data)
            
            return response
            
        except Exception as e:
            error_msg = await self.error_recovery.handle_error(session, e, "state_processing")
            return error_msg
    
    def _update_session_from_co(self, session: ConversationSession, co: Dict[str, Any], path_data: Dict[str, Any]):
        """从CO和路径数据更新会话信息"""
        # 提取客户姓名
        if 'customer_name' in co and co['customer_name']:
            session.customer_name = co['customer_name']
        
        # 更新当前订单
        if path_data.get('path') and path_data.get('score', 0) > 0.7:
            # 高置信度的订单，添加到当前订单
            for item in path_data['path']:
                session.current_order.append(item)
        
        # 更新偏好
        if co.get('preferences'):
            session.preferences.update(co['preferences'])
        
        # 添加上下文
        if co.get('intent'):
            session.push_context(f"intent:{co['intent']}")
    
    def _update_conversation_state(self, session: ConversationSession, co: Dict[str, Any], path_data: Dict[str, Any]):
        """更新对话状态"""
        intent = co.get('intent', '')
        current_state = session.state
        
        # 状态转换逻辑
        if intent in ['greeting', 'start']:
            session.state = ConversationState.GREETING
        elif intent in ['menu', 'browse']:
            session.state = ConversationState.MENU_BROWSING
        elif intent in ['order', 'add_item']:
            session.state = ConversationState.ORDERING
        elif intent in ['confirm', 'finalize']:
            session.state = ConversationState.ORDER_CONFIRMATION
        elif intent in ['pay', 'payment']:
            session.state = ConversationState.PAYMENT
        elif intent in ['help', 'support']:
            session.state = ConversationState.HELP
        elif current_state == ConversationState.ERROR_RECOVERY and co.get('recovery_mode'):
            # 从错误恢复状态回到正常状态
            session.state = ConversationState.MENU_BROWSING
        
        # 如果订单完成，更新历史并清空当前订单
        if session.state == ConversationState.ORDER_COMPLETE and session.current_order:
            session.order_history.append({
                'items': session.current_order.copy(),
                'timestamp': datetime.now().isoformat(),
                'total': sum(item.get('price', 0) * item.get('quantity', 1) for item in session.current_order)
            })
            session.current_order.clear()
    
    def _get_greeting_message(self, session: ConversationSession) -> str:
        """获取问候消息"""
        if session.customer_name:
            return f"¡Hola de nuevo, {session.customer_name}! ¿En qué puedo ayudarte hoy?"
        else:
            return "¡Hola! Bienvenido a nuestro restaurante. ¿En qué puedo ayudarte hoy? Puedes ver nuestro menú o hacer tu pedido directamente."
    
    def _get_help_message(self, session: ConversationSession) -> str:
        """获取帮助消息"""
        return """Puedo ayudarte con:
🍽️ Ver el menú - escribe "menú" o "carta"
🛍️ Hacer un pedido - dime qué quieres ordenar
📞 Información del restaurante - escribe "info"
🔄 Reiniciar - escribe "reiniciar" para empezar de nuevo

¿En qué puedo ayudarte?"""
    
    async def _handle_processing_error(self, payload: Dict[str, Any], error: Exception, trace_id: str):
        """处理处理错误"""
        try:
            phone_number = self._extract_phone_number(payload)
            if phone_number:
                session = await self.session_manager.get_session(phone_number)
                error_msg = await self.error_recovery.handle_error(session, error, "processing")
                
                send_function = self._get_send_function()
                send_function(phone_number, error_msg)
                
                await self.session_manager.save_session(session)
                
        except Exception as e:
            logger.error(f"Error in error handling: {e}", extra={'trace_id': trace_id})

# 全局处理器实例
_enhanced_processor = EnhancedWhatsAppMessageProcessor()

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
    
    # 使用线程池异步处理消息，避免阻塞webhook响应
    import concurrent.futures
    import threading
    
    def run_async():
        """在新线程中运行异步处理"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_enhanced_processor.process_message_async(payload, trace_id))
        except Exception as e:
            logger.error(f"Error in async message processing: {e}", extra={'trace_id': trace_id})
        finally:
            loop.close()
    
    # 使用线程池执行
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    executor.submit(run_async)

async def handle_whatsapp_event_async(payload: Dict[str, Any], trace_id: str):
    """异步处理WhatsApp事件"""
    if _is_status_update(payload):
        return
    
    await _enhanced_processor.process_message_async(payload, trace_id)

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

# 会话管理接口
async def get_session_info_async(phone_number: str) -> Dict[str, Any]:
    """异步获取会话信息"""
    session = await _enhanced_processor.session_manager.get_session(phone_number)
    return session.to_dict()

def get_session_info(phone_number: str) -> Dict[str, Any]:
    """同步获取会话信息"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果在运行的事件循环中，创建任务
            import concurrent.futures
            
            def get_session():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(get_session_info_async(phone_number))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_session)
                return future.result(timeout=5)
        else:
            return loop.run_until_complete(get_session_info_async(phone_number))
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return {}

async def update_session_info_async(phone_number: str, **kwargs):
    """异步更新会话信息"""
    session = await _enhanced_processor.session_manager.get_session(phone_number)
    
    # 更新字段
    for key, value in kwargs.items():
        if hasattr(session, key):
            setattr(session, key, value)
    
    await _enhanced_processor.session_manager.save_session(session)

def update_session_info(phone_number: str, **kwargs):
    """同步更新会话信息"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(update_session_info_async(phone_number, **kwargs))
        else:
            loop.run_until_complete(update_session_info_async(phone_number, **kwargs))
    except Exception as e:
        logger.error(f"Error updating session info: {e}")

async def clear_session_async(phone_number: str):
    """异步清除会话"""
    await _enhanced_processor.session_manager.delete_session(phone_number)

def clear_session(phone_number: str):
    """同步清除会话"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(clear_session_async(phone_number))
        else:
            loop.run_until_complete(clear_session_async(phone_number))
    except Exception as e:
        logger.error(f"Error clearing session: {e}")

# 监控和分析接口
async def get_active_sessions_async() -> List[Dict[str, Any]]:
    """获取所有活跃会话"""
    sessions = await _enhanced_processor.session_manager.get_all_active_sessions()
    return [session.to_dict() for session in sessions]

async def get_session_analytics_async() -> Dict[str, Any]:
    """获取会话分析数据"""
    sessions = await _enhanced_processor.session_manager.get_all_active_sessions()
    
    # 统计分析
    total_sessions = len(sessions)
    state_counts = {}
    language_counts = {}
    error_stats = {'total_errors': 0, 'sessions_with_errors': 0}
    
    for session in sessions:
        # 状态统计
        state = session.state.value
        state_counts[state] = state_counts.get(state, 0) + 1
        
        # 语言统计
        language_counts[session.language] = language_counts.get(session.language, 0) + 1
        
        # 错误统计
        if session.error_count > 0:
            error_stats['sessions_with_errors'] += 1
            error_stats['total_errors'] += session.error_count
    
    # 计算活跃度
    now = datetime.now()
    active_last_hour = sum(1 for s in sessions if (now - s.last_activity).total_seconds() < 3600)
    active_last_day = sum(1 for s in sessions if (now - s.last_activity).total_seconds() < 86400)
    
    return {
        'total_sessions': total_sessions,
        'active_last_hour': active_last_hour,
        'active_last_day': active_last_day,
        'state_distribution': state_counts,
        'language_distribution': language_counts,
        'error_statistics': error_stats,
        'average_messages_per_session': sum(s.total_messages for s in sessions) / max(total_sessions, 1)
    }

# 错误恢复接口
async def trigger_error_recovery_async(phone_number: str, error_type: str = 'manual'):
    """手动触发错误恢复"""
    session = await _enhanced_processor.session_manager.get_session(phone_number)
    
    # 创建人工错误来触发恢复
    fake_error = Exception(f"Manual recovery triggered: {error_type}")
    error_msg = await _enhanced_processor.error_recovery.handle_error(session, fake_error, "manual")
    
    # 发送恢复消息
    send_function = _enhanced_processor._get_send_function()
    send_function(phone_number, error_msg)
    
    await _enhanced_processor.session_manager.save_session(session)

# 测试和调试接口
async def simulate_message_async(phone_number: str, message_text: str, trace_id: str = None) -> Dict[str, Any]:
    """模拟消息处理（用于测试）"""
    if not trace_id:
        trace_id = str(uuid.uuid4())
    
    # 创建模拟payload
    test_payload = {
        'From': phone_number,
        'Body': message_text,
        'MessageSid': f'test_{uuid.uuid4()}'
    }
    
    try:
        await _enhanced_processor.process_message_async(test_payload, trace_id)
        session = await _enhanced_processor.session_manager.get_session(phone_number)
        
        return {
            'success': True,
            'session_state': session.state.value,
            'message_count': session.total_messages,
            'error_count': session.error_count,
            'trace_id': trace_id
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'trace_id': trace_id
        }

def simulate_message(phone_number: str, message_text: str) -> Dict[str, Any]:
    """同步模拟消息处理"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            
            def simulate():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        simulate_message_async(phone_number, message_text)
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(simulate)
                return future.result(timeout=10)
        else:
            return loop.run_until_complete(simulate_message_async(phone_number, message_text))
    except Exception as e:
        return {'success': False, 'error': str(e)}

# 健康检查接口
async def health_check_async() -> Dict[str, Any]:
    """系统健康检查"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {}
    }
    
    # 检查Redis连接
    try:
        redis_client = await _enhanced_processor.session_manager._get_redis_client()
        if redis_client:
            await redis_client.ping()
            health_status['components']['redis'] = 'healthy'
        else:
            health_status['components']['redis'] = 'unavailable (using memory fallback)'
    except Exception as e:
        health_status['components']['redis'] = f'error: {str(e)}'
        health_status['status'] = 'degraded'
    
    # 检查会话管理器
    try:
        test_session = await _enhanced_processor.session_manager.get_session('health_check_test')
        health_status['components']['session_manager'] = 'healthy'
    except Exception as e:
        health_status['components']['session_manager'] = f'error: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    # 检查消息去重器
    try:
        test_payload = {'From': 'test', 'Body': 'health check', 'timestamp': time.time()}
        await _enhanced_processor.deduplicator.is_duplicate(test_payload)
        health_status['components']['deduplicator'] = 'healthy'
    except Exception as e:
        health_status['components']['deduplicator'] = f'error: {str(e)}'
        health_status['status'] = 'degraded'
    
    return health_status

def health_check() -> Dict[str, Any]:
    """同步健康检查"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            
            def check():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(health_check_async())
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(check)
                return future.result(timeout=5)
        else:
            return loop.run_until_complete(health_check_async())
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# 配置管理
def configure_session_timeout(hours: int):
    """配置会话超时时间"""
    _enhanced_processor.session_manager.session_timeout = timedelta(hours=hours)
    logger.info(f"Session timeout configured to {hours} hours")

def configure_deduplication_cache(duration_seconds: int):
    """配置去重缓存时长"""
    _enhanced_processor.deduplicator.cache_duration = duration_seconds
    logger.info(f"Deduplication cache configured to {duration_seconds} seconds")

# 测试函数
if __name__ == "__main__":
    import time
    
    async def test_enhanced_router():
        """测试增强路由器功能"""
        print("Testing Enhanced WhatsApp Router...")
        
        test_phone = "whatsapp:+1234567890"
        trace_id = str(uuid.uuid4())
        
        # 测试消息处理
        test_messages = [
            "Hola",
            "Quiero pollo teriyaki",
            "¿Cuánto cuesta?",
            "Confirmado, quiero ordenar"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Test Message {i}: '{message}' ---")
            
            # 模拟消息
            result = await simulate_message_async(test_phone, message, f"{trace_id}_{i}")
            print(f"Processing result: {result}")
            
            # 获取会话状态
            session_info = await get_session_info_async(test_phone)
            print(f"Session state: {session_info.get('state')}")
            print(f"Total messages: {session_info.get('total_messages')}")
            
            await asyncio.sleep(0.5)  # 短暂延迟
        
        # 测试分析功能
        print("\n--- Analytics ---")
        analytics = await get_session_analytics_async()
        print(f"Analytics: {analytics}")
        
        # 测试健康检查
        print("\n--- Health Check ---")
        health = await health_check_async()
        print(f"Health status: {health}")
        
        # 清理测试会话
        await clear_session_async(test_phone)
        print("\nTest session cleared")
    
    # 运行测试
    asyncio.run(test_enhanced_router())
