#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude API 客户端
与 Anthropic Claude API 交互
"""

import os
import json
import requests
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
            messages.append({
                "role": "user",
                "content": f"System: {system_prompt}\n\nUser: {prompt}"
            })
        else:
            messages.append({
                "role": "user",
