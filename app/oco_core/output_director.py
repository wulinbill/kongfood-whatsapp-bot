#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Output Director
智能输出决策引擎，决定沉默/澄清/回复/执行
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from ..llm.claude_client import ask_claude
from .tension_eval import score, should_clarify, should_execute, get_action_recommendation
from .tension_eval import start_session_tracking, add_clarification, complete_session
from .clarify_engine import build as build_clarification
from .clarify_engine import build_order_confirmation
from ..pos.loyverse_client import place_order
from ..logger import logger

class OutputDirector:
    """输出决策引擎"""
    
    def __init__(self):
        self.flow_rules = self._load_flow_rules()
        self.response_templates = self._load_response_templates()
        
    def _load_flow_rules(self) -> Dict[str, Any]:
        """加载对话流规则"""
        return {
            "greeting": {
                "spanish": "Hola, restaurante Kong Food. ¿Qué desea ordenar hoy?",
                "english": "Hello, Kong Food restaurant. What would you like to order today?",
                "chinese": "您好，Kong Food餐厅。今天想点什么？"
            },
            "execution_thresholds": {
                "clarify_threshold": 0.4,
                "execute_threshold": 0.8,
                "confidence_minimum": 0.3
            },
            "order_flow": [
                "greeting",
                "capture_dishes", 
                "clarifications",
                "customer_name",
                "pos_registration",
                "final_summary",
                "closing"
            ]
        }
    
    def _load_response_templates(self) -> Dict[str, Any]:
        """加载响应模板"""
        return {
            "order_success": {
                "spanish": "Gracias, {name}. Confirmo:\n{order_summary}\nTotal **con impuesto** es ${total}.\nSu orden estará lista en {prep_time} minutos.",
                "english": "Thank you, {name}. I confirm:\n{order_summary}\nTotal **with tax** is ${total}.\nYour order will be ready in {prep_time} minutes.",
                "chinese": "谢谢您，{name}。确认订单：\n{order_summary}\n含税总计：${total}\n您的订单将在{prep_time}分钟内准备好。"
            },
            "order_error": {
                "spanish": "Disculpe, hubo un problema al procesar su orden. ¿Podría intentar de nuevo?",
                "english": "Sorry, there was a problem processing your order. Could you please try again?",
                "chinese": "抱歉，处理您的订单时出现问题。能否重新尝试？"
            },
            "add_more": {
                "spanish": "Perfecto, {item}. ¿Algo más?",
                "english": "Perfect, {item}. Anything else?",
                "chinese": "好的，{item}。还要别的吗？"
            },
            "closing": {
                "spanish": "¡Muchas gracias!",
                "english": "Thank you very much!",
                "chinese": "非常感谢！"
            }
        }
    
    def _get_template(self, template_key: str, language: str, **kwargs) -> str:
        """获取模板并格式化"""
        templates = self.response_templates.get(template_key, {})
        
        if language == 'zh' or language == 'zh-CN':
            template = templates.get('chinese', templates.get('spanish', ''))
        elif language == 'en' or language == 'en-US':
            template = templates.get('english', templates.get('spanish', ''))
        else:
            template = templates.get('spanish', '')
        
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    
    def _determine_prep_time(self, order_items: List[Dict[str, Any]]) -> int:
        """根据订单项目数量确定准备时间"""
        main_dishes = [item for item in order_items 
                      if item.get('category_name') not in ['Adicionales']]
        
        if len(main_dishes) < 3:
            return 10  # 10分钟
        else:
            return 15  # 15分钟
    
    def _calculate_total_with_tax(self, subtotal: float, tax_rate: float = 0.11) -> float:
        """计算含税总价"""
        return round(subtotal * (1 + tax_rate), 2)
    
    def _format_order_summary(self, order_items: List[Dict[str, Any]], language: str) -> str:
        """格式化订单摘要"""
        if not order_items:
            return ""
        
        summary_lines = []
        for item in order_items:
            quantity = item.get('quantity', 1)
            name = item.get('item_name', '')
            modifications = item.get('modifications', [])
            
            if language == 'zh':
                line = f"-- {quantity}份 {name}"
                if modifications:
                    mod_text = '，'.join(modifications)
                    line += f"（{mod_text}）"
            elif language == 'en':
                line = f"-- {quantity}x {name}"
                if modifications:
                    mod_text = ', '.join(modifications)
                    line += f" ({mod_text})"
            else:  # Spanish
                line = f"-- {quantity}x {name}"
                if modifications:
                    mod_text = ', '.join(modifications)
                    line += f" con {mod_text}"
            
            summary_lines.append(line)
        
        return '\n'.join(summary_lines)
    
    def _is_greeting_needed(self, co: Dict[str, Any]) -> bool:
        """判断是否需要问候"""
        # 如果用户直接说想要什么，跳过问候直接处理订单
        intent = co.get('intent', '')
        objects = co.get('objects', [])
        
        return intent not in ['order'] or not objects
    
    def _call_claude_for_response(self, context: str, menu_candidates: str, language: str) -> str:
        """调用Claude生成响应"""
        system_prompt = f"""
You are KongFood AI ordering assistant. Follow these rules:

1. LANGUAGE: Respond in {language} (es=Spanish, en=English, zh=Chinese)
2. MENU CONSTRAINT: Only suggest items from the provided menu candidates
3. CONVERSATION FLOW: Follow the restaurant ordering flow
4. NO HALLUCINATION: Never mention items not in the menu candidates

Menu candidates available:
{menu_candidates}

Context:
{context}

Respond naturally as a restaurant ordering assistant.
"""
        
        try:
            response = ask_claude(system_prompt)
            return response if response else self._get_fallback_response(language)
        except Exception as e:
            logger.error(f"Error calling Claude: {e}")
            return self._get_fallback_response(language)
    
    def _get_fallback_response(self, language: str) -> str:
        """获取回退响应"""
        if language == 'zh':
            return "抱歉，请问您需要什么帮助？"
        elif language == 'en':
            return "Sorry, how can I help you?"
        else:
            return "Disculpa, ¿en qué puedo ayudarte?"
    
    def _extract_menu_candidates(self, path_data: Dict[str, Any]) -> str:
        """从路径数据中提取菜单候选项"""
        candidates = []
        
        if path_data and path_data.get('path'):
            for item in path_data['path']:
                candidate = f"- {item.get('item_name', '')} (${item.get('price', 0.0)})"
                candidates.append(candidate)
        
        if path_data and path_data.get('alternative_paths'):
            for alt_path in path_data['alternative_paths'][:3]:  # 最多3个备选
                for match in alt_path.get('matches', []):
                    candidate = f"- {match.get('item_name', '')} (${match.get('price', 0.0)})"
                    if candidate not in candidates:
                        candidates.append(candidate)
        
        return '\n'.join(candidates) if candidates else "No specific menu items found"

def reply(co_struct: Dict[str, Any], path_data: Dict[str, Any] = None, 
          session_id: str = None) -> str:
    """
    主决策函数 - 决定如何响应
    
    Args:
        co_struct: 对话对象结构
        path_data: 路径数据
        session_id: 会话ID
        
    Returns:
        响应消息
    """
    director = OutputDirector()
    
    # 生成会话ID
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    
    language = co_struct.get('language', 'es')
    intent = co_struct.get('intent', 'inquiry')
    confidence = co_struct.get('confidence', 0.0)
    
    logger.info(f"Processing response - Intent: {intent}, Confidence: {confidence}, Session: {session_id}")
    
    # 1. 处理特殊意图
    if intent == 'cancel':
        return director._get_template('closing', language)
    
    # 2. 问候处理
    if director._is_greeting_needed(co_struct):
        greeting = director.flow_rules['greeting']
        if language == 'zh':
            return greeting['chinese']
        elif language == 'en':
            return greeting['english']
        else:
            return greeting['spanish']
    
    # 3. 订单处理流程
    if intent in ['order', 'modify'] and path_data:
        # 计算张力分数
        tension_score = score(path_data)
        action = get_action_recommendation(tension_score)
        
        logger.info(f"Tension score: {tension_score}, Action: {action}")
        
        # 开始会话跟踪
        start_session_tracking(session_id, confidence, path_data.get('score', 0.0))
        
        if action == 'clarify':
            # 需要澄清
            add_clarification(session_id)
            clarification = build_clarification(co_struct, path_data)
            return clarification
            
        elif action == 'execute':
            # 直接执行订单
            try:
                # 处理订单
                order_result = place_order({
                    'path_data': path_data,
                    'customer_phone': co_struct.get('customer_phone'),
                    'customer_name': co_struct.get('customer_name', 'Cliente')
                })
                
                if order_result.get('success'):
                    # 订单成功
                    complete_session(session_id, True, 1.0)
                    
                    # 格式化成功响应
                    order_items = path_data.get('path', [])
                    subtotal = sum(item.get('price', 0) * item.get('quantity', 1) for item in order_items)
                    total_with_tax = director._calculate_total_with_tax(subtotal)
                    prep_time = director._determine_prep_time(order_items)
                    order_summary = director._format_order_summary(order_items, language)
                    
                    return director._get_template(
                        'order_success',
                        language,
                        name=co_struct.get('customer_name', 'Cliente'),
                        order_summary=order_summary,
                        total=total_with_tax,
                        prep_time=prep_time
                    )
                else:
                    # 订单失败
                    complete_session(session_id, False, 0.0)
                    return director._get_template('order_error', language)
                    
            except Exception as e:
                logger.error(f"Error executing order: {e}")
                complete_session(session_id, False, 0.0)
                return director._get_template('order_error', language)
        
        else:  # action == 'analyze'
            # 需要更多分析，调用Claude
            menu_candidates = director._extract_menu_candidates(path_data)
            context = f"""
User input: {co_struct.get('raw_text', '')}
Intent: {intent}
Confidence: {confidence}
Path matches found: {len(path_data.get('path', []))}
Tension score: {tension_score}
"""
            
            response = director._call_claude_for_response(context, menu_candidates, language)
            return response
    
    # 4. 确认意图处理
    elif intent == 'confirm':
        # 处理用户确认
        if language == 'zh':
            return "好的，正在为您处理订单..."
        elif language == 'en':
            return "Okay, processing your order..."
        else:
            return "Muy bien, procesando su orden..."
    
    # 5. 默认响应 - 调用Claude
    else:
        menu_candidates = director._extract_menu_candidates(path_data) if path_data else "Full menu available"
        context = f"""
User input: {co_struct.get('raw_text', '')}
Intent: {intent}
Confidence: {confidence}
"""
        
        response = director._call_claude_for_response(context, menu_candidates, language)
        return response


# 便捷函数
def reply_with_clarification(co: Dict[str, Any], path_data: Dict[str, Any]) -> str:
    """生成澄清响应"""
    return build_clarification(co, path_data)

def reply_with_confirmation(co: Dict[str, Any], order_items: List[Dict[str, Any]], total: float) -> str:
    """生成确认响应"""
    return build_order_confirmation(co, order_items, total)

def reply_simple(message: str, language: str = 'es') -> str:
    """生成简单响应"""
    director = OutputDirector()
    return director._get_fallback_response(language)


# 测试函数
if __name__ == "__main__":
    # 测试输出决策
    test_co = {
        'objects': [
            {
                'item_type': 'main_dish',
                'content': 'Pollo Teriyaki',
                'quantity': 1,
                'confidence': 0.8
            }
        ],
        'intent': 'order',
        'language': 'es',
        'confidence': 0.8,
        'raw_text': 'quiero Pollo Teriyaki'
    }
    
    test_path = {
        'path': [
            {
                'item_name': 'Pollo Teriyaki',
                'price': 11.99,
                'quantity': 1
            }
        ],
        'score': 0.9,
        'confidence': 0.8,
        'requires_clarification': False
    }
    
    response = reply(test_co, test_path)
    print("响应:")
    print(response)
