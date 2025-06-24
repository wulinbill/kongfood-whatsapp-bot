#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deepgram 语音转录客户端 (优化版)
支持WhatsApp语音消息转录，使用最新Deepgram SDK v3
"""

import os
import asyncio
import httpx
from typing import Tuple, Optional, Dict, Any
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    UrlSource,
    LiveTranscriptionEvents,
    LiveOptions
)
from ..config import settings
from ..logger import logger

class DeepgramTranscriptionClient:
    """Deepgram语音转录客户端 (SDK v3)"""
    
    def __init__(self):
        self.api_key = settings.DEEPGRAM_API_KEY
        self.client = None
        
        if self.api_key:
            try:
                self.client = DeepgramClient(self.api_key)
                logger.info("Deepgram client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Deepgram client: {e}")
        else:
            logger.warning("Deepgram API key not found in environment")
    
    async def transcribe_url_async(self, audio_url: str, **options) -> Tuple[Optional[str], Optional[str]]:
        """
        异步转录音频URL
        
        Args:
            audio_url: 音频文件URL
            **options: 转录选项
            
        Returns:
            (转录文本, 检测的语言)
        """
        if not self.client:
            logger.error("Deepgram client not initialized")
            return None, None
        
        try:
            # 设置默认转录选项
            default_options = {
                "model": "nova-2",
                "punctuate": True,
                "detect_language": True,
                "smart_format": True,
                "diarize": False,
                "filler_words": False,
                "utterances": True,
                "language": "multi"
            }
            
            # 合并用户选项
            default_options.update(options)
            transcription_options = PrerecordedOptions(**default_options)
            
            # 准备音频源
            from deepgram import BufferSource
            audio_source = BufferSource(buffer=audio_data, mimetype=mimetype)
            
            # 发起转录请求
            response = await self.client.listen.asyncrest.v("1").transcribe_file(
                source=audio_source,
                options=transcription_options
            )
            
            # 解析响应（与URL方法相同）
            if response and response.results:
                channels = response.results.channels
                
                if channels and len(channels) > 0:
                    channel = channels[0]
                    
                    if channel.alternatives and len(channel.alternatives) > 0:
                        alternative = channel.alternatives[0]
                        transcript = alternative.transcript.strip()
                        
                        detected_language = "es"
                        
                        if hasattr(alternative, 'language') and alternative.language:
                            lang = alternative.language
                            if lang.startswith("es"):
                                detected_language = "es"
                            elif lang.startswith("en"):
                                detected_language = "en"
                            elif lang.startswith("zh"):
                                detected_language = "zh"
                        
                        logger.info(f"File transcription successful: '{transcript}' (language: {detected_language})")
                        return transcript, detected_language
            
            logger.warning("No transcription results found in file")
            return None, None
            
        except Exception as e:
            logger.error(f"Error transcribing audio file: {e}")
            return None, None
    
    def transcribe_url_sync(self, audio_url: str, **options) -> Tuple[Optional[str], Optional[str]]:
        """同步转录URL接口"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在新线程中运行
                import concurrent.futures
                import threading
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.transcribe_url_async(audio_url, **options)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.transcribe_url_async(audio_url, **options))
        except Exception as e:
            logger.error(f"Error in sync transcription: {e}")
            return None, None
    
    def transcribe_file_sync(self, audio_data: bytes, mimetype: str = "audio/ogg", **options) -> Tuple[Optional[str], Optional[str]]:
        """同步转录文件接口"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.transcribe_file_async(audio_data, mimetype, **options)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.transcribe_file_async(audio_data, mimetype, **options))
        except Exception as e:
            logger.error(f"Error in sync file transcription: {e}")
            return None, None
    
    async def setup_live_transcription(self, on_message_callback, **options) -> Optional[Any]:
        """
        设置实时转录
        
        Args:
            on_message_callback: 消息回调函数
            **options: 转录选项
            
        Returns:
            WebSocket连接对象
        """
        if not self.client:
            logger.error("Deepgram client not initialized")
            return None
        
        try:
            # 设置实时转录选项
            default_options = {
                "model": "nova-2",
                "punctuate": True,
                "smart_format": True,
                "interim_results": True,
                "utterance_end_ms": "1000",
                "vad_events": True,
                "language": "multi"
            }
            
            default_options.update(options)
            live_options = LiveOptions(**default_options)
            
            # 创建WebSocket连接
            dg_connection = self.client.listen.asyncwebsocket.v("1")
            
            # 设置事件处理器
            @dg_connection.on(LiveTranscriptionEvents.Open)
            async def on_open(self, open, **kwargs):
                logger.info("Deepgram live connection opened")
            
            @dg_connection.on(LiveTranscriptionEvents.Transcript)
            async def on_message(self, result, **kwargs):
                if result.is_final:
                    transcript = result.channel.alternatives[0].transcript
                    if transcript:
                        await on_message_callback(transcript)
            
            @dg_connection.on(LiveTranscriptionEvents.Close)
            async def on_close(self, close, **kwargs):
                logger.info("Deepgram live connection closed")
            
            @dg_connection.on(LiveTranscriptionEvents.Error)
            async def on_error(self, error, **kwargs):
                logger.error(f"Deepgram live connection error: {error}")
            
            # 启动连接
            await dg_connection.start(live_options)
            
            return dg_connection
            
        except Exception as e:
            logger.error(f"Error setting up live transcription: {e}")
            return None
    
    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        return [
            "en", "en-US", "en-GB", "en-AU", "en-NZ", "en-IN",
            "es", "es-ES", "es-419", "es-MX", "es-AR",
            "fr", "fr-FR", "fr-CA",
            "de", "de-DE", "de-CH",
            "it", "it-IT",
            "pt", "pt-BR", "pt-PT",
            "ru", "ru-RU",
            "zh", "zh-CN", "zh-TW",
            "ja", "ja-JP",
            "ko", "ko-KR",
            "hi", "hi-IN",
            "ar", "ar-SA",
            "nl", "nl-NL",
            "sv", "sv-SE",
            "no", "no-NO",
            "da", "da-DK",
            "pl", "pl-PL",
            "tr", "tr-TR"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "nova-2": {
                "description": "Latest general-purpose model with 30% lower error rates",
                "languages": self.get_supported_languages(),
                "features": ["punctuation", "formatting", "diarization", "utterances"]
            },
            "nova": {
                "description": "Previous generation general-purpose model",
                "languages": self.get_supported_languages(),
                "features": ["punctuation", "formatting", "diarization"]
            },
            "enhanced": {
                "description": "Enhanced model for phone call audio",
                "languages": ["en", "es"],
                "features": ["punctuation", "phone_call_optimization"]
            },
            "base": {
                "description": "Fastest model with basic features",
                "languages": ["en"],
                "features": ["basic_transcription"]
            }
        }

class WhatsAppAudioProcessor:
    """WhatsApp音频处理器 (优化版)"""
    
    def __init__(self):
        self.deepgram_client = DeepgramTranscriptionClient()
        self.supported_formats = ["audio/ogg", "audio/mpeg", "audio/mp4", "audio/wav"]
    
    async def download_whatsapp_audio(self, media_url: str, auth_headers: Dict[str, str] = None) -> Optional[bytes]:
        """
        下载WhatsApp音频文件
        
        Args:
            media_url: 音频文件URL
            auth_headers: 认证头 (Twilio需要)
            
        Returns:
            音频文件字节数据
        """
        try:
            headers = auth_headers or {}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(media_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', 'audio/ogg')
                    logger.info(f"Audio downloaded: {len(response.content)} bytes, type: {content_type}")
                    return response.content
                else:
                    logger.error(f"Failed to download audio: HTTP {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading WhatsApp audio: {e}")
            return None
    
    async def process_whatsapp_audio(self, media_url: str, auth_headers: Dict[str, str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        处理WhatsApp音频消息
        
        Args:
            media_url: 音频文件URL
            auth_headers: 认证头
            
        Returns:
            (转录文本, 语言)
        """
        try:
            # 方式1：直接使用URL（推荐，速度更快）
            text, language = await self.deepgram_client.transcribe_url_async(media_url)
            
            if text:
                logger.info(f"Direct URL transcription successful: '{text}' ({language})")
                return text, language
            
            # 方式2：下载后处理（备用方案）
            logger.info("Direct URL transcription failed, trying download method")
            audio_data = await self.download_whatsapp_audio(media_url, auth_headers)
            
            if audio_data:
                # 检测音频格式
                mimetype = self._detect_audio_format(audio_data)
                text, language = await self.deepgram_client.transcribe_file_async(audio_data, mimetype)
                
                if text:
                    logger.info(f"Downloaded file transcription successful: '{text}' ({language})")
                    return text, language
            
            logger.warning("Both transcription methods failed")
            return None, None
            
        except Exception as e:
            logger.error(f"Error processing WhatsApp audio: {e}")
            return None, None
    
    def _detect_audio_format(self, audio_data: bytes) -> str:
        """
        检测音频格式
        
        Args:
            audio_data: 音频文件字节数据
            
        Returns:
            MIME类型
        """
        # 检查文件头来确定格式
        if audio_data.startswith(b'OggS'):
            return "audio/ogg"
        elif audio_data.startswith(b'ID3') or audio_data[4:8] == b'ftyp':
            return "audio/mp4"
        elif audio_data.startswith(b'\xff\xfb') or audio_data.startswith(b'\xff\xf3'):
            return "audio/mpeg"
        elif audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
            return "audio/wav"
        else:
            # 默认为OGG (WhatsApp常用格式)
            return "audio/ogg"
    
    def process_whatsapp_audio_sync(self, media_url: str, auth_headers: Dict[str, str] = None) -> Tuple[Optional[str], Optional[str]]:
        """同步处理WhatsApp音频"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.process_whatsapp_audio(media_url, auth_headers)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result(timeout=35)
            else:
                return loop.run_until_complete(self.process_whatsapp_audio(media_url, auth_headers))
        except Exception as e:
            logger.error(f"Error in sync WhatsApp audio processing: {e}")
            return None, None

# 全局实例
_global_processor = WhatsAppAudioProcessor()
_global_client = DeepgramTranscriptionClient()

def transcribe(audio_url: str, **options) -> Tuple[Optional[str], Optional[str]]:
    """
    外部调用接口 - 同步转录URL
    
    Args:
        audio_url: 音频URL
        **options: 转录选项
        
    Returns:
        (转录文本, 语言)
    """
    return _global_client.transcribe_url_sync(audio_url, **options)

def transcribe_file(audio_data: bytes, mimetype: str = "audio/ogg", **options) -> Tuple[Optional[str], Optional[str]]:
    """
    外部调用接口 - 同步转录文件
    
    Args:
        audio_data: 音频数据
        mimetype: 文件类型
        **options: 转录选项
        
    Returns:
        (转录文本, 语言)
    """
    return _global_client.transcribe_file_sync(audio_data, mimetype, **options)

async def transcribe_async(audio_url: str, **options) -> Tuple[Optional[str], Optional[str]]:
    """
    外部调用接口 - 异步转录URL
    
    Args:
        audio_url: 音频URL
        **options: 转录选项
        
    Returns:
        (转录文本, 语言)
    """
    return await _global_client.transcribe_url_async(audio_url, **options)

async def transcribe_file_async(audio_data: bytes, mimetype: str = "audio/ogg", **options) -> Tuple[Optional[str], Optional[str]]:
    """
    外部调用接口 - 异步转录文件
    
    Args:
        audio_data: 音频数据
        mimetype: 文件类型
        **options: 转录选项
        
    Returns:
        (转录文本, 语言)
    """
    return await _global_client.transcribe_file_async(audio_data, mimetype, **options)

def transcribe_whatsapp(media_url: str, auth_headers: Dict[str, str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    WhatsApp语音消息转录 - 同步接口
    
    Args:
        media_url: WhatsApp媒体URL
        auth_headers: 认证头
        
    Returns:
        (转录文本, 语言)
    """
    return _global_processor.process_whatsapp_audio_sync(media_url, auth_headers)

async def transcribe_whatsapp_async(media_url: str, auth_headers: Dict[str, str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    WhatsApp语音消息转录 - 异步接口
    
    Args:
        media_url: WhatsApp媒体URL
        auth_headers: 认证头
        
    Returns:
        (转录文本, 语言)
    """
    return await _global_processor.process_whatsapp_audio(media_url, auth_headers)

def test_deepgram_connection() -> bool:
    """测试Deepgram连接"""
    try:
        # 使用公开的测试音频文件
        test_url = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"
        text, language = transcribe(test_url)
        success = text is not None and len(text) > 0
        
        if success:
            logger.info(f"Deepgram connection test successful: '{text}' ({language})")
        else:
            logger.error("Deepgram connection test failed: no transcription result")
            
        return success
    except Exception as e:
        logger.error(f"Deepgram connection test failed: {e}")
        return False

def get_supported_languages() -> List[str]:
    """获取支持的语言列表"""
    return _global_client.get_supported_languages()

def get_model_info() -> Dict[str, Any]:
    """获取模型信息"""
    return _global_client.get_model_info()

# 向后兼容
def get_client():
    """获取客户端实例"""
    return _global_client

def get_processor():
    """获取处理器实例"""
    return _global_processor


# 测试函数
if __name__ == "__main__":
    import asyncio
    
    async def test_deepgram():
        """测试Deepgram客户端"""
        print("Testing Enhanced Deepgram API Client...")
        
        # 测试连接
        connection_ok = test_deepgram_connection()
        print(f"Connection test: {'Passed' if connection_ok else 'Failed'}")
        
        # 测试支持的语言
        languages = get_supported_languages()
        print(f"Supported languages: {len(languages)} languages")
        
        # 测试模型信息
        models = get_model_info()
        print(f"Available models: {list(models.keys())}")
        
        # 测试公开音频URL
        test_url = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"
        
        # 异步测试
        text, language = await transcribe_async(test_url, model="nova-2")
        print(f"Async transcription: '{text}' (language: {language})")
        
        # 同步测试
        sync_text, sync_language = transcribe(test_url, model="nova-2")
        print(f"Sync transcription: '{sync_text}' (language: {sync_language})")
        
        # 测试不同模型
        enhanced_text, enhanced_lang = await transcribe_async(test_url, model="enhanced")
        print(f"Enhanced model: '{enhanced_text}' (language: {enhanced_lang})")
    
    asyncio.run(test_deepgram()),
                "filler_words": False,
                "utterances": True,
                "language": "multi"  # 多语言检测
            }
            
            # 合并用户选项
            default_options.update(options)
            transcription_options = PrerecordedOptions(**default_options)
            
            # 准备音频源
            audio_source = UrlSource(url=audio_url)
            
            # 发起转录请求
            response = await self.client.listen.asyncrest.v("1").transcribe_url(
                source=audio_source,
                options=transcription_options
            )
            
            # 解析响应
            if response and response.results:
                channels = response.results.channels
                
                if channels and len(channels) > 0:
                    channel = channels[0]
                    
                    if channel.alternatives and len(channel.alternatives) > 0:
                        alternative = channel.alternatives[0]
                        transcript = alternative.transcript.strip()
                        
                        # 获取检测的语言
                        detected_language = "es"  # 默认西班牙语
                        
                        # 从语言检测结果获取
                        if hasattr(alternative, 'language') and alternative.language:
                            lang = alternative.language
                            if lang.startswith("es"):
                                detected_language = "es"
                            elif lang.startswith("en"):
                                detected_language = "en"
                            elif lang.startswith("zh"):
                                detected_language = "zh"
                        
                        # 从utterances获取语言信息（如果可用）
                        if response.results.utterances and len(response.results.utterances) > 0:
                            utterance = response.results.utterances[0]
                            if hasattr(utterance, 'language') and utterance.language:
                                detected_lang = utterance.language
                                if detected_lang.startswith("es"):
                                    detected_language = "es"
                                elif detected_lang.startswith("en"):
                                    detected_language = "en"
                                elif detected_lang.startswith("zh"):
                                    detected_language = "zh"
                        
                        logger.info(f"Transcription successful: '{transcript}' (language: {detected_language})")
                        return transcript, detected_language
            
            logger.warning("No transcription results found")
            return None, None
            
        except Exception as e:
            logger.error(f"Error transcribing audio URL: {e}")
            return None, None
    
    async def transcribe_file_async(self, audio_data: bytes, mimetype: str = "audio/ogg", **options) -> Tuple[Optional[str], Optional[str]]:
        """
        异步转录音频文件数据
        
        Args:
            audio_data: 音频文件字节数据
            mimetype: 音频文件类型
            **options: 转录选项
            
        Returns:
            (转录文本, 检测的语言)
        """
        if not self.client:
            logger.error("Deepgram client not initialized")
            return None, None
        
        try:
            # 设置默认转录选项
            default_options = {
                "model": "nova-2",
                "punctuate": True,
                "detect_language": True,
                "smart_format": True,
                "diarize": False
