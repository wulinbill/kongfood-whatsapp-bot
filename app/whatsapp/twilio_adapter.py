#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版 Twilio WhatsApp 适配器
支持消息模板、媒体处理、队列管理和状态追踪
"""

import os
import asyncio
import json
import uuid
import mimetypes
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiofiles
import aiohttp
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from ..config import settings
from ..logger import logger

class MessageStatus(Enum):
    """消息状态枚举"""
    PENDING = "pending"
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    UNDELIVERED = "undelivered"

class MediaType(Enum):
    """支持的媒体类型"""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    STICKER = "sticker"

class MessageType(Enum):
    """消息类型"""
    TEXT = "text"
    MEDIA = "media"
    TEMPLATE = "template"
    INTERACTIVE = "interactive"
    LOCATION = "location"

@dataclass
class MessageTemplate:
    """消息模板"""
    name: str
    language: str = "en"
    header: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None
    footer: Optional[str] = None
    buttons: Optional[List[Dict[str, Any]]] = None
    
    def format(self, **kwargs) -> Dict[str, Any]:
        """格式化模板"""
        template_data = {
            "name": self.name,
            "language": {"code": self.language}
        }
        
        components = []
        
        if self.header:
            header_component = {"type": "header"}
            if "text" in self.header:
                header_component["parameters"] = [
                    {"type": "text", "text": self.header["text"].format(**kwargs)}
                ]
            elif "media" in self.header:
                media_info = self.header["media"]
                header_component["parameters"] = [
                    {"type": media_info["type"], media_info["type"]: {"link": media_info["url"]}}
                ]
            components.append(header_component)
        
        if self.body:
            body_component = {
                "type": "body",
                "parameters": [
                    {"type": "text", "text": str(value)}
                    for value in kwargs.values()
                ]
            }
            components.append(body_component)
        
        if components:
            template_data["components"] = components
        
        return template_data

@dataclass
class MediaFile:
    """媒体文件信息"""
    file_path: Optional[str] = None
    url: Optional[str] = None
    content: Optional[bytes] = None
    mime_type: Optional[str] = None
    filename: Optional[str] = None
    size: Optional[int] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.file_path:
            path = Path(self.file_path)
            if path.exists():
                self.filename = self.filename or path.name
                self.mime_type = self.mime_type or mimetypes.guess_type(self.file_path)[0]
                self.size = path.stat().st_size
        
        if self.content and not self.size:
            self.size = len(self.content)
    
    @property
    def media_type(self) -> Optional[MediaType]:
        """获取媒体类型"""
        if not self.mime_type:
            return None
        
        if self.mime_type.startswith('image/'):
            return MediaType.IMAGE
        elif self.mime_type.startswith('audio/'):
            return MediaType.AUDIO
        elif self.mime_type.startswith('video/'):
            return MediaType.VIDEO
        elif self.mime_type == 'application/pdf' or self.mime_type.startswith('text/'):
            return MediaType.DOCUMENT
        else:
            return MediaType.DOCUMENT
    
    def validate(self) -> bool:
        """验证媒体文件"""
        if not any([self.file_path, self.url, self.content]):
            return False
        
        # 检查文件大小限制
        max_sizes = {
            MediaType.IMAGE: 5 * 1024 * 1024,  # 5MB
            MediaType.AUDIO: 16 * 1024 * 1024,  # 16MB
            MediaType.VIDEO: 16 * 1024 * 1024,  # 16MB
            MediaType.DOCUMENT: 100 * 1024 * 1024,  # 100MB
        }
        
        media_type = self.media_type
        if media_type and self.size:
            max_size = max_sizes.get(media_type, 5 * 1024 * 1024)
            if self.size > max_size:
                logger.warning(f"Media file too large: {self.size} > {max_size}")
                return False
        
        return True

@dataclass
class MessageRecord:
    """消息记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    to_number: str = ""
    message_type: MessageType = MessageType.TEXT
    content: str = ""
    media: Optional[MediaFile] = None
    template: Optional[MessageTemplate] = None
    template_params: Dict[str, Any] = field(default_factory=dict)
    status: MessageStatus = MessageStatus.PENDING
    twilio_sid: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 0  # 0=highest, 9=lowest
    scheduled_at: Optional[datetime] = None
    
    def update_status(self, status: MessageStatus, error_message: str = None):
        """更新消息状态"""
        self.status = status
        if error_message:
            self.error_message = error_message
        
        if status == MessageStatus.SENT and not self.sent_at:
            self.sent_at = datetime.now()
        elif status == MessageStatus.DELIVERED and not self.delivered_at:
            self.delivered_at = datetime.now()

class MessageQueue:
    """消息队列管理器"""
    
    def __init__(self, max_size: int = 1000, batch_size: int = 10):
        self.max_size = max_size
        self.batch_size = batch_size
        self.queue: List[MessageRecord] = []
        self.processing = False
        self.lock = asyncio.Lock()
        self.rate_limit = {
            'messages_per_second': 1,
            'last_sent': 0
        }
    
    async def add_message(self, message: MessageRecord) -> bool:
        """添加消息到队列"""
        async with self.lock:
            if len(self.queue) >= self.max_size:
                # 移除最旧的低优先级消息
                self.queue.sort(key=lambda x: (x.priority, x.created_at))
                removed = self.queue.pop()
                logger.warning(f"Queue full, removed message {removed.id}")
            
            # 按优先级和创建时间排序插入
            self.queue.append(message)
            self.queue.sort(key=lambda x: (x.priority, x.created_at))
            
            logger.info(f"Message {message.id} added to queue. Queue size: {len(self.queue)}")
            return True
    
    async def get_next_batch(self) -> List[MessageRecord]:
        """获取下一批待发送消息"""
        async with self.lock:
            now = datetime.now()
            ready_messages = [
                msg for msg in self.queue
                if msg.status == MessageStatus.PENDING and 
                (msg.scheduled_at is None or msg.scheduled_at <= now)
            ]
            
            batch = ready_messages[:self.batch_size]
            for msg in batch:
                msg.status = MessageStatus.QUEUED
            
            return batch
    
    async def remove_message(self, message_id: str):
        """从队列中移除消息"""
        async with self.lock:
            self.queue = [msg for msg in self.queue if msg.id != message_id]
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        async with self.lock:
            status_counts = {}
            for status in MessageStatus:
                status_counts[status.value] = sum(1 for msg in self.queue if msg.status == status)
            
            return {
                "total_messages": len(self.queue),
                "status_breakdown": status_counts,
                "processing": self.processing,
                "oldest_pending": min(
                    (msg.created_at for msg in self.queue if msg.status == MessageStatus.PENDING),
                    default=None
                )
            }
    
    async def apply_rate_limiting(self):
        """应用速率限制"""
        now = time.time()
        time_since_last = now - self.rate_limit['last_sent']
        min_interval = 1.0 / self.rate_limit['messages_per_second']
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.rate_limit['last_sent'] = time.time()

class EnhancedTwilioWhatsAppAdapter:
    """增强版 Twilio WhatsApp 适配器"""
    
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.whatsapp_number = os.getenv("WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.webhook_url = os.getenv("TWILIO_WEBHOOK_URL")
        
        # 初始化组件
        self.client = None
        self.message_queue = MessageQueue()
        self.templates: Dict[str, MessageTemplate] = {}
        self.status_callbacks: Dict[str, Callable] = {}
        self.media_storage_path = Path(settings.MEDIA_DIR if hasattr(settings, 'MEDIA_DIR') else "/tmp/whatsapp_media")
        self.media_storage_path.mkdir(exist_ok=True)
        
        # 消息记录
        self.message_records: Dict[str, MessageRecord] = {}
        
        if not self.account_sid or not self.auth_token:
            logger.warning("Twilio credentials not found in environment")
        else:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("Enhanced Twilio client initialized successfully")
                
                # 启动队列处理器
                asyncio.create_task(self._queue_processor())
                
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
    
    def add_template(self, template: MessageTemplate):
        """添加消息模板"""
        self.templates[template.name] = template
        logger.info(f"Template '{template.name}' added")
    
    def register_status_callback(self, message_id: str, callback: Callable):
        """注册状态回调"""
        self.status_callbacks[message_id] = callback
    
    async def send_text_message(self, to_number: str, text: str, 
                               priority: int = 0, scheduled_at: datetime = None) -> str:
        """发送文本消息"""
        message = MessageRecord(
            to_number=self._format_phone_number(to_number),
            message_type=MessageType.TEXT,
            content=text,
            priority=priority,
            scheduled_at=scheduled_at
        )
        
        await self.message_queue.add_message(message)
        self.message_records[message.id] = message
        
        return message.id
    
    async def send_media_message(self, to_number: str, text: str, media: MediaFile,
                                priority: int = 0, scheduled_at: datetime = None) -> str:
        """发送媒体消息"""
        if not media.validate():
            raise ValueError("Invalid media file")
        
        message = MessageRecord(
            to_number=self._format_phone_number(to_number),
            message_type=MessageType.MEDIA,
            content=text,
            media=media,
            priority=priority,
            scheduled_at=scheduled_at
        )
        
        await self.message_queue.add_message(message)
        self.message_records[message.id] = message
        
        return message.id
    
    async def send_template_message(self, to_number: str, template_name: str,
                                   params: Dict[str, Any] = None, priority: int = 0) -> str:
        """发送模板消息"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        message = MessageRecord(
            to_number=self._format_phone_number(to_number),
            message_type=MessageType.TEMPLATE,
            template=template,
            template_params=params or {},
            priority=priority
        )
        
        await self.message_queue.add_message(message)
        self.message_records[message.id] = message
        
        return message.id
    
    async def _queue_processor(self):
        """队列处理器"""
        while True:
            try:
                if not self.client:
                    await asyncio.sleep(5)
                    continue
                
                batch = await self.message_queue.get_next_batch()
                if not batch:
                    await asyncio.sleep(1)
                    continue
                
                self.message_queue.processing = True
                
                for message in batch:
                    try:
                        await self.message_queue.apply_rate_limiting()
                        await self._send_message_direct(message)
                        
                    except Exception as e:
                        logger.error(f"Error sending message {message.id}: {e}")
                        await self._handle_send_error(message, str(e))
                
                self.message_queue.processing = False
                
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(5)
    
    async def _send_message_direct(self, message: MessageRecord):
        """直接发送消息"""
        try:
            if message.message_type == MessageType.TEXT:
                result = await self._send_text_direct(message)
            elif message.message_type == MessageType.MEDIA:
                result = await self._send_media_direct(message)
            elif message.message_type == MessageType.TEMPLATE:
                result = await self._send_template_direct(message)
            else:
                raise ValueError(f"Unsupported message type: {message.message_type}")
            
            if result:
                message.twilio_sid = result.sid
                message.update_status(MessageStatus.SENT)
                logger.info(f"Message {message.id} sent successfully: {result.sid}")
                
                # 执行回调
                await self._execute_callback(message.id, message.status)
            else:
                raise Exception("Failed to send message")
                
        except Exception as e:
            await self._handle_send_error(message, str(e))
    
    async def _send_text_direct(self, message: MessageRecord):
        """发送文本消息"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(
                body=message.content,
                from_=self.whatsapp_number,
                to=message.to_number,
                status_callback=f"{self.webhook_url}/status" if self.webhook_url else None
            )
        )
    
    async def _send_media_direct(self, message: MessageRecord):
        """发送媒体消息"""
        media_url = await self._prepare_media_url(message.media)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(
                body=message.content,
                from_=self.whatsapp_number,
                to=message.to_number,
                media_url=[media_url],
                status_callback=f"{self.webhook_url}/status" if self.webhook_url else None
            )
        )
    
    async def _send_template_direct(self, message: MessageRecord):
        """发送模板消息"""
        template_data = message.template.format(**message.template_params)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(
                from_=self.whatsapp_number,
                to=message.to_number,
                content_sid=None,  # 对于模板消息
                content_variables=json.dumps(template_data),
                status_callback=f"{self.webhook_url}/status" if self.webhook_url else None
            )
        )
    
    async def _prepare_media_url(self, media: MediaFile) -> str:
        """准备媒体URL"""
        if media.url:
            return media.url
        
        # 如果是本地文件或字节数据，需要上传到可访问的URL
        if media.file_path:
            # 复制到媒体存储目录
            filename = f"{uuid.uuid4()}_{media.filename}"
            storage_path = self.media_storage_path / filename
            
            async with aiofiles.open(media.file_path, 'rb') as src:
                content = await src.read()
            
            async with aiofiles.open(storage_path, 'wb') as dst:
                await dst.write(content)
            
            # 返回可访问的URL（需要配置Web服务器）
            base_url = os.getenv("MEDIA_BASE_URL", "http://localhost:8000/media")
            return f"{base_url}/{filename}"
        
        elif media.content:
            # 保存字节数据到文件
            filename = f"{uuid.uuid4()}_{media.filename or 'file'}"
            storage_path = self.media_storage_path / filename
            
            async with aiofiles.open(storage_path, 'wb') as f:
                await f.write(media.content)
            
            base_url = os.getenv("MEDIA_BASE_URL", "http://localhost:8000/media")
            return f"{base_url}/{filename}"
        
        raise ValueError("No valid media source found")
    
    async def _handle_send_error(self, message: MessageRecord, error_msg: str):
        """处理发送错误"""
        message.retry_count += 1
        message.update_status(MessageStatus.FAILED, error_msg)
        
        if message.retry_count < message.max_retries:
            # 重新加入队列，降低优先级
            message.priority = min(message.priority + 1, 9)
            message.status = MessageStatus.PENDING
            message.scheduled_at = datetime.now() + timedelta(minutes=message.retry_count * 2)
            
            logger.info(f"Message {message.id} scheduled for retry {message.retry_count}")
        else:
            logger.error(f"Message {message.id} failed permanently after {message.retry_count} retries")
            await self.message_queue.remove_message(message.id)
        
        await self._execute_callback(message.id, message.status)
    
    async def _execute_callback(self, message_id: str, status: MessageStatus):
        """执行状态回调"""
        if message_id in self.status_callbacks:
            try:
                callback = self.status_callbacks[message_id]
                if asyncio.iscoroutinefunction(callback):
                    await callback(message_id, status)
                else:
                    callback(message_id, status)
            except Exception as e:
                logger.error(f"Callback execution failed for message {message_id}: {e}")
    
    def _format_phone_number(self, phone_number: str) -> str:
        """格式化电话号码"""
        if phone_number.startswith('whatsapp:'):
            return phone_number
        
        # 移除所有非数字字符
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # 如果没有国家代码，假设是美国号码
        if len(clean_number) == 10:
            clean_number = '1' + clean_number
        
        return f'whatsapp:+{clean_number}'
    
    async def handle_webhook(self, webhook_data: Dict[str, Any]) -> bool:
        """处理Twilio webhook"""
        try:
            message_sid = webhook_data.get('MessageSid')
            message_status = webhook_data.get('MessageStatus')
            
            if not message_sid or not message_status:
                return False
            
            # 查找对应的消息记录
            message_record = None
            for record in self.message_records.values():
                if record.twilio_sid == message_sid:
                    message_record = record
                    break
            
            if not message_record:
                logger.warning(f"Message record not found for SID: {message_sid}")
                return False
            
            # 更新状态
            status_mapping = {
                'sent': MessageStatus.SENT,
                'delivered': MessageStatus.DELIVERED,
                'read': MessageStatus.READ,
                'failed': MessageStatus.FAILED,
                'undelivered': MessageStatus.UNDELIVERED
            }
            
            new_status = status_mapping.get(message_status.lower())
            if new_status:
                message_record.update_status(new_status)
                await self._execute_callback(message_record.id, new_status)
                logger.info(f"Message {message_record.id} status updated to {new_status.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return False
    
    async def download_media_async(self, media_url: str, save_path: str = None) -> Optional[MediaFile]:
        """异步下载媒体文件"""
        try:
            auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
            
            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.get(media_url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.read()
                        content_type = response.headers.get('content-type')
                        
                        # 如果指定了保存路径，保存文件
                        if save_path:
                            async with aiofiles.open(save_path, 'wb') as f:
                                await f.write(content)
                        
                        return MediaFile(
                            content=content,
                            mime_type=content_type,
                            size=len(content),
                            file_path=save_path
                        )
                    else:
                        logger.error(f"Failed to download media: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
            return None
    
    async def get_message_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """获取消息状态"""
        if message_id not in self.message_records:
            return None
        
        message = self.message_records[message_id]
        
        status_info = {
            'id': message.id,
            'to_number': message.to_number,
            'message_type': message.message_type.value,
            'status': message.status.value,
            'twilio_sid': message.twilio_sid,
            'created_at': message.created_at.isoformat(),
            'sent_at': message.sent_at.isoformat() if message.sent_at else None,
            'delivered_at': message.delivered_at.isoformat() if message.delivered_at else None,
            'error_message': message.error_message,
            'retry_count': message.retry_count
        }
        
        # 如果有Twilio SID，获取Twilio的状态
        if message.twilio_sid and self.client:
            try:
                loop = asyncio.get_event_loop()
                twilio_message = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages(message.twilio_sid).fetch()
                )
                
                status_info['twilio_status'] = {
                    'status': twilio_message.status,
                    'error_code': twilio_message.error_code,
                    'error_message': twilio_message.error_message,
                    'price': twilio_message.price,
                    'date_sent': twilio_message.date_sent.isoformat() if twilio_message.date_sent else None,
                    'date_updated': twilio_message.date_updated.isoformat() if twilio_message.date_updated else None
                }
                
            except Exception as e:
                logger.warning(f"Could not fetch Twilio status for message {message_id}: {e}")
        
        return status_info
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return await self.message_queue.get_queue_status()
    
    def validate_webhook_signature(self, url: str, post_data: Dict[str, Any], signature: str) -> bool:
        """验证webhook签名"""
        try:
            from twilio.request_validator import RequestValidator
            validator = RequestValidator(self.auth_token)
            return validator.validate(url, post_data, signature)
        except Exception as e:
            logger.error(f"Error validating webhook signature: {e}")
            return False

# 全局实例
_enhanced_adapter = EnhancedTwilioWhatsAppAdapter()

# 预定义模板
_enhanced_adapter.add_template(MessageTemplate(
    name="order_confirmation",
    language="en",
    header={"text": "Order Confirmation #{order_id}"},
    body={"text": "Your order for {items} has been confirmed. Total: ${total}. Estimated delivery: {delivery_time}."},
    footer="Thank you for your order!"
))

_enhanced_adapter.add_template(MessageTemplate(
    name="welcome_message",
    language="en",
    body={"text": "Welcome to {restaurant_name}! We're excited to serve you. Type 'menu' to see our offerings."}
))

# 公共接口
async def send_text_message_async(phone_number: str, text: str, priority: int = 0) -> str:
    """发送文本消息"""
    return await _enhanced_adapter.send_text_message(phone_number, text, priority)

async def send_media_message_async(phone_number: str, text: str, media: MediaFile, priority: int = 0) -> str:
    """发送媒体消息"""
    return await _enhanced_adapter.send_media_message(phone_number, text, media, priority)

async def send_template_message_async(phone_number: str, template_name: str, params: Dict[str, Any] = None) -> str:
    """发送模板消息"""
    return await _enhanced_adapter.send_template_message(phone_number, template_name, params)

async def get_message_status_async(message_id: str) -> Optional[Dict[str, Any]]:
    """获取消息状态"""
    return await _enhanced_adapter.get_message_status(message_id)

async def download_media_async(media_url: str, save_path: str = None) -> Optional[MediaFile]:
    """下载媒体文件"""
    return await _enhanced_adapter.download_media_async(media_url, save_path)

def register_status_callback(message_id: str, callback: Callable):
    """注册状态回调"""
    _enhanced_adapter.register_status_callback(message_id, callback)

def add_template(template: MessageTemplate):
    """添加消息模板"""
    _enhanced_adapter.add_template(template)

async def handle_webhook_async(webhook_data: Dict[str, Any]) -> bool:
    """处理webhook"""
    return await _enhanced_adapter.handle_webhook(webhook_data)

def get_adapter():
    """获取适配器实例"""
    return _enhanced_adapter

# 向后兼容的同步接口
def send_message(phone_number: str, text: str) -> bool:
    """同步发送消息接口（向后兼容）"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在已运行的事件循环中创建任务
            task = asyncio.create_task(send_text_message_async(phone_number, text))
            return True  # 返回True表示已加入队列
        else:
            message_id = loop.run_until_complete(send_text_message_async(phone_number, text))
            return bool(message_id)
    except Exception as e:
        logger.error(f"Error in sync send_message: {e}")
        return False

def send_media_message(phone_number: str, text: str, media_url: str) -> bool:
    """同步发送媒体消息接口（向后兼容）"""
    try:
        media = MediaFile(url=media_url)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(send_media_message_async(phone_number, text, media))
            return True
        else:
            message_id = loop.run_until_complete(send_media_message_async(phone_number, text, media))
            return bool(message_id)
    except Exception as e:
        logger.error(f"Error in sync send_media_message: {e}")
        return False

# 测试和工具函数
async def test_enhanced_adapter():
    """测试增强适配器功能"""
    print("Testing Enhanced Twilio WhatsApp Adapter...")
    
    # 测试文本消息
    text_message_id = await send_text_message_async("+1234567890", "Hello from enhanced adapter!")
    print(f"Text message queued: {text_message_id}")
    
    # 测试媒体消息
    test_image = MediaFile(
        url="https://example.com/test-image.jpg",
        mime_type="image/jpeg",
        filename="test.jpg"
    )
    media_message_id = await send_media_message_async("+1234567890", "Check out this image!", test_image)
    print(f"Media message queued: {media_message_id}")
    
    # 测试模板消息
    template_message_id = await send_template_message_async(
        "+1234567890", 
        "order_confirmation",
        {
            "order_id": "12345",
            "items": "2x Chicken Teriyaki",
            "total": "25.99",
            "delivery_time": "30 minutes"
        }
    )
    print(f"Template message queued: {template_message_id}")
    
    # 等待一段时间后检查状态
    await asyncio.sleep(2)
    
    for msg_id in [text_message_id, media_message_id, template_message_id]:
        status = await get_message_status_async(msg_id)
        print(f"Message {msg_id} status: {status['status'] if status else 'Not found'}")
    
    # 检查队列状态
    queue_status = await _enhanced_adapter.get_queue_status()
    print(f"Queue status: {queue_status}")

if __name__ == "__main__":
    import time
    
    # 运行测试
    asyncio.run(test_enhanced_adapter())
