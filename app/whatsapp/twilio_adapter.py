#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twilio WhatsApp 适配器
完整的消息发送和媒体处理功能
"""

import os
import asyncio
from typing import Optional, Dict, Any
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from ..config import settings
from ..logger import logger

class TwilioWhatsAppAdapter:
    """Twilio WhatsApp 消息适配器"""
    
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN") 
        self.whatsapp_number = os.getenv("WHATSAPP_NUMBER", "whatsapp:+14155238886")
        
        if not self.account_sid or not self.auth_token:
            logger.warning("Twilio credentials not found in environment")
            self.client = None
        else:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("Twilio client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
                self.client = None
    
    def send_message(self, to_number: str, message_text: str) -> bool:
        """
        发送文本消息到WhatsApp
        
        Args:
            to_number: 目标号码 (格式: whatsapp:+1234567890)
            message_text: 消息文本
            
        Returns:
            发送是否成功
        """
        if not self.client:
            logger.error("Twilio client not initialized")
            return False
        
        try:
            # 确保号码格式正确
            if not to_number.startswith('whatsapp:'):
                to_number = f'whatsapp:{to_number}'
            
            message = self.client.messages.create(
                body=message_text,
                from_=self.whatsapp_number,
                to=to_number
            )
            
            logger.info(f"Message sent successfully: {message.sid}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Twilio error sending message: {e.msg} (Code: {e.code})")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return False
    
    def send_media_message(self, to_number: str, message_text: str, media_url: str) -> bool:
        """
        发送包含媒体的消息
        
        Args:
            to_number: 目标号码
            message_text: 消息文本
            media_url: 媒体文件URL
            
        Returns:
            发送是否成功
        """
        if not self.client:
            logger.error("Twilio client not initialized")
            return False
        
        try:
            if not to_number.startswith('whatsapp:'):
                to_number = f'whatsapp:{to_number}'
            
            message = self.client.messages.create(
                body=message_text,
                from_=self.whatsapp_number,
                to=to_number,
                media_url=[media_url]
            )
            
            logger.info(f"Media message sent successfully: {message.sid}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Twilio error sending media message: {e.msg} (Code: {e.code})")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending media message: {e}")
            return False
    
    async def send_message_async(self, to_number: str, message_text: str) -> bool:
        """异步发送消息"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.send_message, to_number, message_text)
    
    def get_message_status(self, message_sid: str) -> Optional[Dict[str, Any]]:
        """
        获取消息状态
        
        Args:
            message_sid: 消息SID
            
        Returns:
            消息状态信息
        """
        if not self.client:
            return None
        
        try:
            message = self.client.messages(message_sid).fetch()
            return {
                'sid': message.sid,
                'status': message.status,
                'error_code': message.error_code,
                'error_message': message.error_message,
                'date_sent': message.date_sent,
                'date_updated': message.date_updated
            }
        except Exception as e:
            logger.error(f"Error fetching message status: {e}")
            return None
    
    def download_media(self, media_url: str) -> Optional[bytes]:
        """
        下载媒体文件
        
        Args:
            media_url: Twilio媒体URL
            
        Returns:
            媒体文件字节数据
        """
        if not self.client:
            return None
        
        try:
            # 从Twilio获取媒体文件
            import requests
            
            # Twilio媒体URL需要认证
            auth = (self.account_sid, self.auth_token)
            response = requests.get(media_url, auth=auth, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Media downloaded successfully: {len(response.content)} bytes")
                return response.content
            else:
                logger.error(f"Failed to download media: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
            return None
    
    def validate_webhook_signature(self, url: str, post_data: Dict[str, Any], signature: str) -> bool:
        """
        验证Twilio webhook签名
        
        Args:
            url: webhook URL
            post_data: POST数据
            signature: X-Twilio-Signature header
            
        Returns:
            签名是否有效
        """
        try:
            from twilio.request_validator import RequestValidator
            
            validator = RequestValidator(self.auth_token)
            return validator.validate(url, post_data, signature)
            
        except Exception as e:
            logger.error(f"Error validating webhook signature: {e}")
            return False
    
    def format_phone_number(self, phone_number: str) -> str:
        """
        格式化电话号码为WhatsApp格式
        
        Args:
            phone_number: 原始电话号码
            
        Returns:
            格式化的号码
        """
        # 移除所有非数字字符
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # 如果没有国家代码，假设是美国号码
        if len(clean_number) == 10:
            clean_number = '1' + clean_number
        
        return f'whatsapp:+{clean_number}'
    
    def test_connection(self) -> bool:
        """测试Twilio连接"""
        if not self.client:
            return False
        
        try:
            # 获取账户信息来测试连接
            account = self.client.api.accounts(self.account_sid).fetch()
            logger.info(f"Twilio connection test successful. Account: {account.friendly_name}")
            return True
        except Exception as e:
            logger.error(f"Twilio connection test failed: {e}")
            return False

# 全局适配器实例
_adapter = TwilioWhatsAppAdapter()

def send_message(phone_number: str, text: str) -> bool:
    """外部调用接口 - 发送消息"""
    return _adapter.send_message(phone_number, text)

def send_media_message(phone_number: str, text: str, media_url: str) -> bool:
    """外部调用接口 - 发送媒体消息"""
    return _adapter.send_media_message(phone_number, text, media_url)

async def send_message_async(phone_number: str, text: str) -> bool:
    """外部调用接口 - 异步发送消息"""
    return await _adapter.send_message_async(phone_number, text)

def validate_webhook(url: str, post_data: Dict[str, Any], signature: str) -> bool:
    """外部调用接口 - 验证webhook"""
    return _adapter.validate_webhook_signature(url, post_data, signature)

def test_twilio_connection() -> bool:
    """外部调用接口 - 测试连接"""
    return _adapter.test_connection()

def download_media(media_url: str) -> Optional[bytes]:
    """外部调用接口 - 下载媒体"""
    return _adapter.download_media(media_url)

# 向后兼容
def get_adapter():
    """获取适配器实例"""
    return _adapter
