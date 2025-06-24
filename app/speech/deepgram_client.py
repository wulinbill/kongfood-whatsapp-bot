#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deepgram 语音转录客户端
支持WhatsApp语音消息转录
"""

import os
import asyncio
import httpx
from typing import Tuple, Optional
from deepgram import Deepgram
from ..config import settings
from ..logger import logger

class DeepgramClient:
    """Deepgram语音转录客户端"""
    
    def __init__(self):
        self.api_key = settings.DEEPGRAM_API_KEY
        self.dg_client = None
        
        if self.api_key:
            try:
                self.dg_client = Deepgram(self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Deepgram client: {e}")
        else:
            logger.warning("Deepgram API key not found in environment")
    
    async def transcribe_audio_url(self, audio_url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        转录音频URL
        
        Args:
            audio_url: 音频文件URL
            
        Returns:
            (转录文本, 检测的语言)
        """
        if not self.dg_client:
            logger.error("Deepgram client not initialized")
            return None, None
        
        try:
            # Deepgram转录选项
            options = {
                "punctuate": True,
                "model": "nova-2",  # 最新的模型
                "language": "multi",  # 自动检测语言
                "detect_language": True,
                "smart_format": True,
                "diarize": False,  # 不需要说话者分离
                "filler_words": False,  # 去除填充词
                "utterances": True  # 获取语言信息
            }
            
            # 发起转录请求
            response = await self.dg_client.transcription.prerecorded({
                "url": audio_url
            }, options)
            
            # 解析响应
            if response and "results" in response:
                results = response["results"]
                
                if "channels" in results and len(results["channels"]) > 0:
                    channel = results["channels"][0]
                    
                    if "alternatives" in channel and len(channel["alternatives"]) > 0:
                        alternative = channel["alternatives"][0]
                        transcript = alternative.get("transcript", "").strip()
                        
                        # 获取检测的语言
                        detected_language = "es"  # 默认西班牙语
                        
                        if "language" in alternative:
                            lang = alternative["language"]
                            if lang.startswith("es"):
                                detected_language = "es"
                            elif lang.startswith("en"):
                                detected_language = "en"
                            elif lang.startswith("zh"):
                                detected_language = "zh"
                        
                        # 也可以从utterances中获取语言信息
                        if "utterances" in results and len(results["utterances"]) > 0:
                            utterance = results["utterances"][0]
                            if "languages" in utterance and len(utterance["languages"]) > 0:
                                detected_lang = utterance["languages"][0]["language"]
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
            logger.error(f"Error transcribing audio: {e}")
            return None, None
    
    async def transcribe_audio_file(self, audio_data: bytes, mimetype: str = "audio/ogg") -> Tuple[Optional[str], Optional[str]]:
        """
        转录音频文件数据
        
        Args:
            audio_data: 音频文件字节数据
            mimetype: 音频文件类型
            
        Returns:
            (转录文本, 检测的语言)
        """
        if not self.dg_client:
            logger.error("Deepgram client not initialized")
            return None, None
        
        try:
            options = {
                "punctuate": True,
                "model": "nova-2",
                "language": "multi",
                "detect_language": True,
                "smart_format": True,
                "diarize": False,
                "filler_words": False,
                "utterances": True
            }
            
            # 发起转录请求
            response = await self.dg_client.transcription.prerecorded({
                "buffer": audio_data,
                "mimetype": mimetype
            }, options)
            
            # 解析响应（与URL方法相同）
            if response and "results" in response:
                results = response["results"]
                
                if "channels" in results and len(results["channels"]) > 0:
                    channel = results["channels"][0]
                    
                    if "alternatives" in channel and len(channel["alternatives"]) > 0:
                        alternative = channel["alternatives"][0]
                        transcript = alternative.get("transcript", "").strip()
                        
                        detected_language = "es"
                        
                        if "language" in alternative:
                            lang = alternative["language"]
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
    
    def transcribe_sync(self, audio_url: str) -> Tuple[Optional[str], Optional[str]]:
        """同步转录接口"""
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
                            self.transcribe_audio_url(audio_url)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.transcribe_audio_url(audio_url))
        except Exception as e:
            logger.error(f"Error in sync transcription: {e}")
            return None, None

class WhatsAppAudioProcessor:
    """WhatsApp音频处理器"""
    
    def __init__(self):
        self.deepgram_client = DeepgramClient()
    
    async def download_whatsapp_audio(self, media_url: str) -> Optional[bytes]:
        """下载WhatsApp音频文件"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(media_url, timeout=30)
                
                if response.status_code == 200:
                    return response.content
                else:
                    logger.error(f"Failed to download audio: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading WhatsApp audio: {e}")
            return None
    
    async def process_whatsapp_audio(self, media_url: str) -> Tuple[Optional[str], Optional[str]]:
        """处理WhatsApp音频消息"""
        try:
            # 方式1：直接使用URL（推荐）
            text, language = await self.deepgram_client.transcribe_audio_url(media_url)
            
            if text:
                return text, language
            
            # 方式2：下载后处理（备用）
            logger.info("Direct URL transcription failed, trying download method")
            audio_data = await self.download_whatsapp_audio(media_url)
            
            if audio_data:
                return await self.deepgram_client.transcribe_audio_file(audio_data, "audio/ogg")
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error processing WhatsApp audio: {e}")
            return None, None

# 全局实例
_global_processor = WhatsAppAudioProcessor()
_global_client = DeepgramClient()

def transcribe(audio_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    外部调用接口 - 同步转录
    
    Args:
        audio_url: 音频URL
        
    Returns:
        (转录文本, 语言)
    """
    return _global_client.transcribe_sync(audio_url)

async def transcribe_async(audio_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    外部调用接口 - 异步转录
    
    Args:
        audio_url: 音频URL
        
    Returns:
        (转录文本, 语言)
    """
    return await _global_client.transcribe_audio_url(audio_url)

def transcribe_whatsapp(media_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    WhatsApp语音消息转录 - 同步接口
    
    Args:
        media_url: WhatsApp媒体URL
        
    Returns:
        (转录文本, 语言)
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在新线程中运行
            import concurrent.futures
            
            def run_async():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        _global_processor.process_whatsapp_audio(media_url)
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result(timeout=35)
        else:
            return loop.run_until_complete(_global_processor.process_whatsapp_audio(media_url))
    except Exception as e:
        logger.error(f"Error in WhatsApp transcription: {e}")
        return None, None

async def transcribe_whatsapp_async(media_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    WhatsApp语音消息转录 - 异步接口
    
    Args:
        media_url: WhatsApp媒体URL
        
    Returns:
        (转录文本, 语言)
    """
    return await _global_processor.process_whatsapp_audio(media_url)

def test_deepgram_connection() -> bool:
    """测试Deepgram连接"""
    try:
        # 使用公开的测试音频文件
        test_url = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"
        text, language = transcribe(test_url)
        return text is not None and len(text) > 0
    except Exception as e:
        logger.error(f"Deepgram connection test failed: {e}")
        return False


# 测试函数
if __name__ == "__main__":
    import asyncio
    
    async def test_deepgram():
        """测试Deepgram客户端"""
        print("Testing Deepgram API Client...")
        
        # 测试连接
        connection_ok = test_deepgram_connection()
        print(f"Connection test: {'Passed' if connection_ok else 'Failed'}")
        
        # 测试公开音频URL
        test_url = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"
        
        # 异步测试
        text, language = await transcribe_async(test_url)
        print(f"Async transcription: '{text}' (language: {language})")
        
        # 同步测试
        sync_text, sync_language = transcribe(test_url)
        print(f"Sync transcription: '{sync_text}' (language: {sync_language})")
    
    asyncio.run(test_deepgram())
