#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kong Food WhatsApp AI Bot - 主应用入口
集成 WhatsApp, 语音识别, AI解析, POS系统
"""

import asyncio
import json
import uuid
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .logger import logger
from .webhooks.whatsapp_handler import WhatsAppHandler
from .ai.o_co_engine import OCoEngine
from .pos.loyverse_client import get_authorization_url, handle_oauth_callback
from .speech.deepgram_client import DeepgramClient
from .utils.session_manager import SessionManager

# 全局组件实例
whatsapp_handler = WhatsAppHandler()
oco_engine = OCoEngine()
deepgram_client = DeepgramClient()
session_manager = SessionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("🚀 Kong Food WhatsApp AI Bot starting up...")
    
    # 初始化AI引擎
    await oco_engine.initialize()
    logger.info("✅ O_co AI Engine initialized")
    
    # 初始化语音识别
    await deepgram_client.initialize()
    logger.info("✅ Deepgram Speech Client initialized")
    
    # 初始化会话管理器
    await session_manager.initialize()
    logger.info("✅ Session Manager initialized")
    
    logger.info("🎯 Bot is ready to serve customers!")
    
    yield
    
    # 关闭时清理
    logger.info("🛑 Kong Food WhatsApp AI Bot shutting down...")
    await session_manager.cleanup()
    await deepgram_client.cleanup()
    await oco_engine.cleanup()
    logger.info("✅ Cleanup completed")

# 创建 FastAPI 应用
app = FastAPI(
    title="Kong Food WhatsApp AI Bot",
    description="智能餐厅订餐机器人 - 支持多语言语音/文本订单处理",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求跟踪中间件
@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """为每个请求添加跟踪ID"""
    trace_id = str(uuid.uuid4())[:8]
    request.state.trace_id = trace_id
    
    # 添加到日志上下文
    import logging
    logger_adapter = logging.LoggerAdapter(logger, {"trace_id": trace_id})
    request.state.logger = logger_adapter
    
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response

@app.get("/")
async def health_check():
    """健康检查端点"""
    return {"status": "live", "service": "kong-food-whatsapp-bot"}

@app.get("/health")
async def detailed_health():
    """详细健康检查"""
    health_status = {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "components": {
            "ai_engine": await oco_engine.health_check(),
            "speech_client": await deepgram_client.health_check(),
            "session_manager": await session_manager.health_check(),
            "whatsapp_handler": whatsapp_handler.health_check()
        }
    }
    
    # 检查是否所有组件都健康
    all_healthy = all(
        comp.get("status") == "healthy" 
        for comp in health_status["components"].values()
    )
    
    if not all_healthy:
        health_status["status"] = "degraded"
    
    return health_status

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """WhatsApp Webhook处理器"""
    trace_id = request.state.trace_id
    req_logger = request.state.logger
    
    try:
        # 获取请求数据
        body = await request.body()
        headers = dict(request.headers)
        
        req_logger.info(f"📱 Received WhatsApp webhook")
        
        # 验证webhook签名
        if not whatsapp_handler.verify_webhook(body, headers):
            req_logger.warning("❌ Webhook signature verification failed")
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # 解析webhook数据
        webhook_data = json.loads(body) if body else {}
        
        # 异步处理消息（避免超时）
        background_tasks.add_task(
            process_whatsapp_message,
            webhook_data,
            trace_id
        )
        
        req_logger.info("✅ Webhook processed successfully")
        return {"status": "ok"}
        
    except json.JSONDecodeError:
        req_logger.error("❌ Invalid JSON in webhook")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        req_logger.error(f"❌ Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_whatsapp_message(webhook_data: Dict[str, Any], trace_id: str):
    """处理 WhatsApp 消息（后台任务）"""
    logger_adapter = logging.LoggerAdapter(logger, {"trace_id": trace_id})
    
    try:
        # 提取消息信息
        message_info = whatsapp_handler.extract_message_info(webhook_data)
        if not message_info:
            logger_adapter.info("📭 No processable message found")
            return
        
        from_number = message_info["from"]
        message_type = message_info["type"]
        content = message_info["content"]
        message_id = message_info["id"]
        
        logger_adapter.info(f"📨 Processing {message_type} from {from_number}")
        
        # 获取或创建会话
        session = await session_manager.get_or_create_session(from_number)
        
        # 处理不同类型的消息
        if message_type == "text":
            response_text = await handle_text_message(content, session, logger_adapter)
        elif message_type == "audio":
            response_text = await handle_audio_message(content, session, logger_adapter)
        else:
            response_text = "Lo siento, solo puedo procesar mensajes de texto y audio."
            logger_adapter.warning(f"⚠️ Unsupported message type: {message_type}")
        
        # 发送回复
        if response_text:
            await whatsapp_handler.send_message(from_number, response_text)
            logger_adapter.info("✅ Response sent successfully")
        
        # 更新会话
        await session_manager.update_session(from_number, {
            "last_message_id": message_id,
            "last_activity": asyncio.get_event_loop().time()
        })
        
    except Exception as e:
        logger_adapter.error(f"❌ Error in background message processing: {e}")

async def handle_text_message(text: str, session: Dict[str, Any], logger_adapter) -> str:
    """处理文本消息"""
    try:
        logger_adapter.info(f"💬 Processing text: {text[:100]}...")
        
        # 使用 O_co 引擎处理
        result = await oco_engine.process_message(
            text=text,
            session_data=session,
            message_type="text"
        )
        
        # 更新会话状态
        if result.get("session_updates"):
            session.update(result["session_updates"])
        
        return result.get("response", "Disculpe, no pude procesar su mensaje.")
        
    except Exception as e:
        logger_adapter.error(f"❌ Error processing text message: {e}")
        return "Disculpe, hubo un error procesando su mensaje. Por favor intente de nuevo."

async def handle_audio_message(audio_url: str, session: Dict[str, Any], logger_adapter) -> str:
    """处理语音消息"""
    try:
        logger_adapter.info(f"🎤 Processing audio from: {audio_url[:100]}...")
        
        # 语音转文本
        transcription = await deepgram_client.transcribe_audio(audio_url)
        if not transcription:
            logger_adapter.warning("⚠️ Audio transcription failed")
            return "Disculpe, no pude entender el audio. ¿Podría enviarlo como texto?"
        
        logger_adapter.info(f"📝 Transcribed: {transcription[:100]}...")
        
        # 处理转录文本
        return await handle_text_message(transcription, session, logger_adapter)
        
    except Exception as e:
        logger_adapter.error(f"❌ Error processing audio message: {e}")
        return "Disculpe, hubo un error procesando el audio. Por favor intente de nuevo."

# OAuth 端点
@app.get("/oauth/loyverse/authorize")
async def loyverse_authorize():
    """Loyverse OAuth 授权"""
    auth_url = get_authorization_url()
    return {"authorization_url": auth_url}

@app.get("/oauth/loyverse/callback")
async def loyverse_callback(code: str, state: str = None):
    """Loyverse OAuth 回调"""
    try:
        success = await handle_oauth_callback(code)
        if success:
            return {"status": "success", "message": "Loyverse integration authorized"}
        else:
            raise HTTPException(status_code=400, detail="Authorization failed")
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 管理端点
@app.get("/admin/sessions")
async def get_active_sessions():
    """获取活跃会话（仅调试用）"""
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    sessions = await session_manager.get_all_sessions()
    return {"active_sessions": len(sessions), "sessions": sessions}

@app.post("/admin/sessions/{phone}/reset")
async def reset_session(phone: str):
    """重置特定会话（仅调试用）"""
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    await session_manager.reset_session(phone)
    return {"status": "success", "message": f"Session reset for {phone}"}

@app.get("/admin/ai/stats")
async def get_ai_stats():
    """获取AI引擎统计（仅调试用）"""
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    stats = await oco_engine.get_statistics()
    return stats

# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.error(f"🚨 Global exception [{trace_id}]: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "trace_id": trace_id,
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    # 开发环境启动
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
        log_level="info"
    )
