#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deepgram 语音转录客户端 (全面优化版)
支持WhatsApp语音消息转录，减少代码重复，增强错误处理，音频质量检测
"""

import os
import asyncio
import httpx
import wave
import struct
import audioop
import io
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import hashlib
from collections import deque
import threading

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    UrlSource,
    BufferSource,
    LiveTranscriptionEvents,
    LiveOptions
)
from ..config import settings
from ..logger import logger

class AudioQuality(Enum):
    """音频质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"

class TranscriptionMethod(Enum):
    """转录方法"""
    URL_DIRECT = "url_direct"
    FILE_UPLOAD = "file_upload"
    LIVE_STREAM = "live_stream"

@dataclass
class AudioAnalysis:
    """音频分析结果"""
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bit_depth: int = 0
    file_size: int = 0
    format_type: str = "unknown"
    quality: AudioQuality = AudioQuality.FAIR
    signal_to_noise_ratio: float = 0.0
    has_speech: bool = True
    recommended_preprocessing: List[str] = field(default_factory=list)

@dataclass
class TranscriptionResult:
    """转录结果"""
    text: Optional[str] = None
    language: Optional[str] = None
    confidence: float = 0.0
    method_used: TranscriptionMethod = TranscriptionMethod.URL_DIRECT
    processing_time: float = 0.0
    audio_analysis: Optional[AudioAnalysis] = None
    word_timestamps: List[Dict] = field(default_factory=list)
    utterances: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None

class AudioPreprocessor:
    """音频预处理器"""
    
    @staticmethod
    def analyze_audio_data(audio_data: bytes, mimetype: str = "audio/ogg") -> AudioAnalysis:
        """
        分析音频数据质量
        
        Args:
            audio_data: 音频字节数据
            mimetype: 音频类型
            
        Returns:
            音频分析结果
        """
        analysis = AudioAnalysis()
        analysis.file_size = len(audio_data)
        analysis.format_type = AudioPreprocessor._detect_audio_format(audio_data)
        
        try:
            # 基本检查
            if len(audio_data) < 1024:  # 太小的文件
                analysis.quality = AudioQuality.UNUSABLE
                analysis.recommended_preprocessing.append("file_too_small")
                return analysis
            
            # 检测格式特征
            if analysis.format_type == "audio/wav":
                analysis = AudioPreprocessor._analyze_wav_audio(audio_data, analysis)
            elif analysis.format_type == "audio/ogg":
                analysis = AudioPreprocessor._analyze_ogg_audio(audio_data, analysis)
            else:
                # 其他格式的基本分析
                analysis.quality = AudioQuality.FAIR
                analysis.has_speech = True
            
            # 基于文件大小和时长的质量评估
            if analysis.duration > 0:
                bitrate = (analysis.file_size * 8) / analysis.duration
                
                if bitrate > 128000:  # 高质量
                    analysis.quality = AudioQuality.EXCELLENT
                elif bitrate > 64000:  # 中等质量
                    analysis.quality = AudioQuality.GOOD
                elif bitrate > 32000:  # 可接受质量
                    analysis.quality = AudioQuality.FAIR
                else:  # 低质量
                    analysis.quality = AudioQuality.POOR
                    analysis.recommended_preprocessing.append("low_bitrate")
            
            # 时长检查
            if analysis.duration < 0.5:
                analysis.recommended_preprocessing.append("too_short")
            elif analysis.duration > 300:  # 超过5分钟
                analysis.recommended_preprocessing.append("segment_long_audio")
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            analysis.quality = AudioQuality.FAIR
        
        return analysis
    
    @staticmethod
    def _detect_audio_format(audio_data: bytes) -> str:
        """检测音频格式"""
        if audio_data.startswith(b'OggS'):
            return "audio/ogg"
        elif audio_data.startswith(b'ID3') or (len(audio_data) > 8 and audio_data[4:8] == b'ftyp'):
            return "audio/mp4"
        elif audio_data.startswith(b'\xff\xfb') or audio_data.startswith(b'\xff\xf3'):
            return "audio/mpeg"
        elif audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
            return "audio/wav"
        elif audio_data.startswith(b'fLaC'):
            return "audio/flac"
        else:
            return "audio/ogg"  # 默认
    
    @staticmethod
    def _analyze_wav_audio(audio_data: bytes, analysis: AudioAnalysis) -> AudioAnalysis:
        """分析WAV音频"""
        try:
            audio_file = io.BytesIO(audio_data)
            with wave.open(audio_file, 'rb') as wav_file:
                analysis.sample_rate = wav_file.getframerate()
                analysis.channels = wav_file.getnchannels()
                analysis.bit_depth = wav_file.getsampwidth() * 8
                analysis.duration = wav_file.getnframes() / analysis.sample_rate
                
                # 读取音频数据进行简单的信号分析
                frames = wav_file.readframes(wav_file.getnframes())
                if frames:
                    # 计算RMS (Root Mean Square) 作为音量指标
                    if analysis.bit_depth == 16:
                        samples = struct.unpack(f'<{len(frames)//2}h', frames)
                        rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
                        analysis.signal_to_noise_ratio = min(100, rms / 1000)  # 简化的SNR估算
                    
                    # 检查是否有语音信号
                    max_amplitude = max(abs(s) for s in samples) if samples else 0
                    analysis.has_speech = max_amplitude > 100  # 简单的阈值检查
                
        except Exception as e:
            logger.warning(f"WAV analysis failed: {e}")
            analysis.duration = 1.0  # 默认时长
        
        return analysis
    
    @staticmethod
    def _analyze_ogg_audio(audio_data: bytes, analysis: AudioAnalysis) -> AudioAnalysis:
        """分析OGG音频（简化版）"""
        try:
            # OGG分析比较复杂，这里做简化处理
            # 基于文件大小估算时长
            estimated_bitrate = 32000  # 假设32kbps（WhatsApp常用）
            analysis.duration = (len(audio_data) * 8) / estimated_bitrate
            analysis.sample_rate = 16000  # WhatsApp常用采样率
            analysis.channels = 1  # 单声道
            analysis.bit_depth = 16
            analysis.has_speech = True
            
        except Exception as e:
            logger.warning(f"OGG analysis failed: {e}")
            analysis.duration = 1.0
        
        return analysis
    
    @staticmethod
    def should_preprocess(analysis: AudioAnalysis) -> bool:
        """判断是否需要预处理"""
        return (analysis.quality in [AudioQuality.POOR, AudioQuality.UNUSABLE] or
                len(analysis.recommended_preprocessing) > 0)
    
    @staticmethod
    def apply_preprocessing(audio_data: bytes, analysis: AudioAnalysis) -> bytes:
        """应用音频预处理"""
        # 这里可以实现音频预处理逻辑
        # 比如噪声减少、音量标准化等
        # 目前返回原始数据
        logger.info(f"Audio preprocessing recommendations: {analysis.recommended_preprocessing}")
        return audio_data

class EnhancedDeepgramClient:
    """增强的Deepgram客户端"""
    
    def __init__(self):
        self.api_key = settings.DEEPGRAM_API_KEY
        self.client = None
        self.preprocessor = AudioPreprocessor()
        
        # 性能监控
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'quality_distribution': {quality.value: 0 for quality in AudioQuality}
        }
        
        # 缓存系统
        self.cache = {}
        self.cache_max_size = 100
        
        if self.api_key:
            try:
                self.client = DeepgramClient(self.api_key)
                logger.info("Enhanced Deepgram client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Deepgram client: {e}")
        else:
            logger.warning("Deepgram API key not found in environment")
    
    def _get_cache_key(self, audio_data_or_url: Union[str, bytes], options: Dict) -> str:
        """生成缓存键"""
        if isinstance(audio_data_or_url, str):
            content = audio_data_or_url
        else:
            content = hashlib.md5(audio_data_or_url[:1024]).hexdigest()  # 使用前1KB的hash
        
        options_str = str(sorted(options.items()))
        return hashlib.md5(f"{content}_{options_str}".encode()).hexdigest()
    
    def _update_performance_stats(self, success: bool, processing_time: float, 
                                 audio_quality: AudioQuality = None):
        """更新性能统计"""
        self.performance_stats['total_requests'] += 1
        
        if success:
            self.performance_stats['successful_requests'] += 1
        else:
            self.performance_stats['failed_requests'] += 1
        
        # 更新平均处理时间
        total = self.performance_stats['total_requests']
        old_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (old_avg * (total - 1) + processing_time) / total
        
        # 更新质量分布
        if audio_quality:
            self.performance_stats['quality_distribution'][audio_quality.value] += 1
    
    def _build_transcription_options(self, **user_options) -> PrerecordedOptions:
        """构建转录选项"""
        default_options = {
            "model": "nova-2",
            "punctuate": True,
            "detect_language": True,
            "smart_format": True,
            "diarize": False,
            "filler_words": False,
            "utterances": True,
            "language": "multi",
            "keywords": ["Kong Food", "pollo", "carne", "arroz", "combo"],  # 餐厅相关关键词
            "redact": False,
            "numerals": True
        }
        
        # 合并用户选项
        default_options.update(user_options)
        return PrerecordedOptions(**default_options)
    
    def _extract_transcription_data(self, response) -> Tuple[Optional[str], Optional[str], float, List, List]:
        """从响应中提取转录数据"""
        if not response or not response.results:
            return None, None, 0.0, [], []
        
        channels = response.results.channels
        if not channels or len(channels) == 0:
            return None, None, 0.0, [], []
        
        channel = channels[0]
        if not channel.alternatives or len(channel.alternatives) == 0:
            return None, None, 0.0, [], []
        
        alternative = channel.alternatives[0]
        transcript = alternative.transcript.strip() if alternative.transcript else None
        
        # 获取置信度
        confidence = getattr(alternative, 'confidence', 0.0)
        
        # 获取语言
        detected_language = "es"  # 默认
        if hasattr(alternative, 'language') and alternative.language:
            lang = alternative.language.lower()
            if lang.startswith("es"):
                detected_language = "es"
            elif lang.startswith("en"):
                detected_language = "en"
            elif lang.startswith("zh"):
                detected_language = "zh"
        
        # 获取词级时间戳
        word_timestamps = []
        if hasattr(alternative, 'words') and alternative.words:
            for word in alternative.words:
                word_timestamps.append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'confidence': getattr(word, 'confidence', 0.0)
                })
        
        # 获取话语
        utterances = []
        if response.results.utterances:
            for utterance in response.results.utterances:
                utterances.append({
                    'transcript': utterance.transcript,
                    'start': utterance.start,
                    'end': utterance.end,
                    'confidence': getattr(utterance, 'confidence', 0.0),
                    'channel': utterance.channel,
                    'speaker': getattr(utterance, 'speaker', None)
                })
        
        return transcript, detected_language, confidence, word_timestamps, utterances
    
    async def transcribe_url_async(self, audio_url: str, **options) -> TranscriptionResult:
        """异步转录URL"""
        start_time = time.time()
        result = TranscriptionResult(method_used=TranscriptionMethod.URL_DIRECT)
        
        if not self.client:
            result.error_details = "Deepgram client not initialized"
            return result
        
        try:
            # 检查缓存
            cache_key = self._get_cache_key(audio_url, options)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                cached_result.processing_time = time.time() - start_time
                logger.info("Retrieved transcription from cache")
                return cached_result
            
            # 构建选项
            transcription_options = self._build_transcription_options(**options)
            audio_source = UrlSource(url=audio_url)
            
            # 发起转录请求
            response = await self.client.listen.asyncrest.v("1").transcribe_url(
                source=audio_source,
                options=transcription_options
            )
            
            # 提取数据
            text, language, confidence, word_timestamps, utterances = self._extract_transcription_data(response)
            
            if text:
                result.text = text
                result.language = language
                result.confidence = confidence
                result.word_timestamps = word_timestamps
                result.utterances = utterances
                result.processing_time = time.time() - start_time
                
                # 缓存结果
                if len(self.cache) < self.cache_max_size:
                    self.cache[cache_key] = result
                
                logger.info(f"URL transcription successful: '{text}' (lang: {language}, conf: {confidence:.3f})")
                self._update_performance_stats(True, result.processing_time)
            else:
                result.error_details = "No transcription results found"
                self._update_performance_stats(False, time.time() - start_time)
            
        except Exception as e:
            result.error_details = str(e)
            result.processing_time = time.time() - start_time
            logger.error(f"Error transcribing audio URL: {e}")
            self._update_performance_stats(False, result.processing_time)
        
        return result
    
    async def transcribe_file_async(self, audio_data: bytes, mimetype: str = "audio/ogg", **options) -> TranscriptionResult:
        """异步转录文件"""
        start_time = time.time()
        result = TranscriptionResult(method_used=TranscriptionMethod.FILE_UPLOAD)
        
        if not self.client:
            result.error_details = "Deepgram client not initialized"
            return result
        
        try:
            # 音频分析
            audio_analysis = self.preprocessor.analyze_audio_data(audio_data, mimetype)
            result.audio_analysis = audio_analysis
            
            # 更新质量统计
            self.performance_stats['quality_distribution'][audio_analysis.quality.value] += 1
            
            # 检查是否可用
            if audio_analysis.quality == AudioQuality.UNUSABLE:
                result.error_details = f"Audio quality unusable: {audio_analysis.recommended_preprocessing}"
                return result
            
            # 预处理（如果需要）
            if self.preprocessor.should_preprocess(audio_analysis):
                audio_data = self.preprocessor.apply_preprocessing(audio_data, audio_analysis)
            
            # 检查缓存
            cache_key = self._get_cache_key(audio_data, options)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                cached_result.processing_time = time.time() - start_time
                cached_result.audio_analysis = audio_analysis
                logger.info("Retrieved file transcription from cache")
                return cached_result
            
            # 构建选项
            transcription_options = self._build_transcription_options(**options)
            
            # 根据分析结果调整选项
            if audio_analysis.quality == AudioQuality.POOR:
                # 对低质量音频使用增强模型
                transcription_options.model = "enhanced"
                transcription_options.filler_words = True  # 低质量音频可能需要填充词过滤
            
            audio_source = BufferSource(buffer=audio_data, mimetype=mimetype)
            
            # 发起转录请求
            response = await self.client.listen.asyncrest.v("1").transcribe_file(
                source=audio_source,
                options=transcription_options
            )
            
            # 提取数据
            text, language, confidence, word_timestamps, utterances = self._extract_transcription_data(response)
            
            if text:
                result.text = text
                result.language = language
                result.confidence = confidence
                result.word_timestamps = word_timestamps
                result.utterances = utterances
                result.processing_time = time.time() - start_time
                result.metadata = {
                    'audio_duration': audio_analysis.duration,
                    'audio_quality': audio_analysis.quality.value,
                    'preprocessing_applied': len(audio_analysis.recommended_preprocessing) > 0
                }
                
                # 缓存结果
                if len(self.cache) < self.cache_max_size:
                    self.cache[cache_key] = result
                
                logger.info(f"File transcription successful: '{text}' (lang: {language}, conf: {confidence:.3f}, quality: {audio_analysis.quality.value})")
                self._update_performance_stats(True, result.processing_time, audio_analysis.quality)
            else:
                result.error_details = "No transcription results found"
                self._update_performance_stats(False, time.time() - start_time, audio_analysis.quality)
            
        except Exception as e:
            result.error_details = str(e)
            result.processing_time = time.time() - start_time
            logger.error(f"Error transcribing audio file: {e}")
            self._update_performance_stats(False, result.processing_time)
        
        return result
    
    def transcribe_sync(self, audio_data_or_url: Union[str, bytes], 
                       mimetype: str = "audio/ogg", **options) -> TranscriptionResult:
        """统一的同步转录接口"""
        try:
            # 获取或创建事件循环
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 在运行的循环中，使用线程池
                    return self._run_in_thread(audio_data_or_url, mimetype, **options)
                else:
                    # 循环未运行，直接运行
                    if isinstance(audio_data_or_url, str):
                        return loop.run_until_complete(self.transcribe_url_async(audio_data_or_url, **options))
                    else:
                        return loop.run_until_complete(self.transcribe_file_async(audio_data_or_url, mimetype, **options))
            except RuntimeError:
                # 没有事件循环，创建新的
                if isinstance(audio_data_or_url, str):
                    return asyncio.run(self.transcribe_url_async(audio_data_or_url, **options))
                else:
                    return asyncio.run(self.transcribe_file_async(audio_data_or_url, mimetype, **options))
        
        except Exception as e:
            logger.error(f"Error in sync transcription: {e}")
            result = TranscriptionResult()
            result.error_details = str(e)
            return result
    
    def _run_in_thread(self, audio_data_or_url: Union[str, bytes], 
                      mimetype: str = "audio/ogg", **options) -> TranscriptionResult:
        """在单独线程中运行异步函数"""
        import concurrent.futures
        
        def run_async():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                if isinstance(audio_data_or_url, str):
                    return new_loop.run_until_complete(
                        self.transcribe_url_async(audio_data_or_url, **options)
                    )
                else:
                    return new_loop.run_until_complete(
                        self.transcribe_file_async(audio_data_or_url, mimetype, **options)
                    )
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async)
            return future.result(timeout=40)  # 40秒超时
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['failure_rate'] = stats['failed_requests'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        stats['cache_size'] = len(self.cache)
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("Transcription cache cleared")
    
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

class OptimizedWhatsAppProcessor:
    """优化的WhatsApp音频处理器"""
    
    def __init__(self):
        self.deepgram_client = EnhancedDeepgramClient()
        self.supported_formats = ["audio/ogg", "audio/mpeg", "audio/mp4", "audio/wav"]
        
        # 重试配置
        self.retry_config = {
            'max_retries': 3,
            'retry_delay': 1.0,
            'backoff_factor': 2.0
        }
    
    async def download_whatsapp_audio(self, media_url: str, 
                                    auth_headers: Dict[str, str] = None) -> Optional[bytes]:
        """优化的WhatsApp音频下载"""
        headers = auth_headers or {}
        
        for attempt in range(self.retry_config['max_retries']):
            try:
                timeout_config = httpx.Timeout(30.0, connect=10.0)
                
                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    response = await client.get(media_url, headers=headers)
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', 'audio/ogg')
                        content_length = len(response.content)
                        
                        logger.info(f"Audio downloaded: {content_length} bytes, type: {content_type}")
                        
                        # 基本验证
                        if content_length < 100:  # 太小的文件
                            logger.warning("Downloaded audio file is too small")
                            return None
                        
                        return response.content
                    else:
                        logger.warning(f"Download attempt {attempt + 1} failed: HTTP {response.status_code}")
                        
            except httpx.TimeoutException:
                logger.warning(f"Download attempt {attempt + 1} timed out")
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} error: {e}")
            
            # 重试延迟
            if attempt < self.retry_config['max_retries'] - 1:
                delay = self.retry_config['retry_delay'] * (self.retry_config['backoff_factor'] ** attempt)
                await asyncio.sleep(delay)
        
        logger.error("All download attempts failed")
        return None
    
    async def process_whatsapp_audio(self, media_url: str, 
                                   auth_headers: Dict[str, str] = None) -> TranscriptionResult:
        """统一的WhatsApp音频处理"""
        
        # 方法1：直接URL转录（推荐）
        logger.info("Attempting direct URL transcription")
        result = await self.deepgram_client.transcribe_url_async(
            media_url, 
            model="nova-2",
            smart_format=True
        )
        
        if result.text and result.confidence > 0.3:
            logger.info(f"Direct URL transcription successful: confidence={result.confidence:.3f}")
            return result
        
        # 方法2：下载后转录（备用）
        logger.info("Direct URL failed, trying download method")
        audio_data = await self.download_whatsapp_audio(media_url, auth_headers)
        
        if audio_data:
            # 检测格式
            detected_format = AudioPreprocessor._detect_audio_format(audio_data)
            
            # 转录文件
            file_result = await self.deepgram_client.transcribe_file_async(
                audio_data, 
                detected_format,
                model="nova-2",
                smart_format=True
            )
            
            if file_result.text:
                logger.info(f"File transcription successful: confidence={file_result.confidence:.3f}")
                return file_result
        
        # 如果都失败了，返回最好的结果
        if result.text:
            return result
        elif 'file_result' in locals() and file_result.text:
            return file_result
        else:
            # 创建失败结果
            failed_result = TranscriptionResult()
            failed_result.error_details = "All transcription methods failed"
            return failed_result
    
    def process_whatsapp_audio_sync(self, media_url: str, 
                                  auth_headers: Dict[str, str] = None) -> TranscriptionResult:
        """同步WhatsApp音频处理"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._run_whatsapp_in_thread(media_url, auth_headers)
            else:
                return loop.run_until_complete(self.process_whatsapp_audio(media_url, auth_headers))
        except RuntimeError:
            return asyncio.run(self.process_whatsapp_audio(media_url, auth_headers))
        except Exception as e:
            logger.error(f"Error in sync WhatsApp processing: {e}")
            result = TranscriptionResult()
            result.error_details = str(e)
            return result
    
    def _run_whatsapp_in_thread(self, media_url: str, auth_headers: Dict[str, str] = None) -> TranscriptionResult:
        """在单独线程中运行WhatsApp处理"""
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
        
        with concurrent.futures.Threa
