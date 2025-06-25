#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Output Director (优化版)
智能输出决策引擎 - 简化决策流程、A/B测试、动态优化、策略多样化
"""

import json
import uuid
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path

from ..llm.claude_client import ask_claude
from .tension_eval import score, should_clarify, should_execute, get_action_recommendation
from .tension_eval import start_session_tracking, add_clarification, complete_session
from .clarify_engine import build as build_clarification
from .clarify_engine import build_order_confirmation, generate_enhanced_clarification
from ..pos.loyverse_client import place_order
from ..logger import logger

class ResponseStrategy(Enum):
    """响应策略枚举"""
    CONCISE = "concise"
    DETAILED = "detailed"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    EFFICIENT = "efficient"

class DecisionType(Enum):
    """决策类型枚举"""
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    EXECUTION = "execution"
    CONFIRMATION = "confirmation"
    ERROR_HANDLING = "error_handling"
    FALLBACK = "fallback"
    UPSELL = "upsell"
    CLOSING = "closing"

@dataclass
class DecisionContext:
    """决策上下文"""
    session_id: str
    user_id: str = "anonymous"
    conversation_turn: int = 0
    previous_decisions: List[DecisionType] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    ab_test_group: Optional[str] = None
    response_strategy: ResponseStrategy = ResponseStrategy.CONVERSATIONAL

@dataclass
class DecisionResult:
    """决策结果"""
    response: str
    decision_type: DecisionType
    strategy_used: ResponseStrategy
    confidence: float
    execution_time: float
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DecisionLogger:
    """决策日志记录器"""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.decision_log = deque(maxlen=max_entries)
        self.performance_metrics = defaultdict(list)
        self.ab_test_results = defaultdict(lambda: defaultdict(list))
    
    def log_decision(self, context: DecisionContext, result: DecisionResult,
                    co_struct: Dict[str, Any], path_data: Dict[str, Any] = None):
        """记录决策"""
        log_entry = {
            'timestamp': time.time(),
            'session_id': context.session_id,
            'user_id': context.user_id,
            'decision_type': result.decision_type.value,
            'strategy': result.strategy_used.value,
            'confidence': result.confidence,
            'execution_time': result.execution_time,
            'input_intent': co_struct.get('intent'),
            'input_confidence': co_struct.get('confidence'),
            'path_score': path_data.get('score') if path_data else None,
            'ab_test_group': context.ab_test_group,
            'response_length': len(result.response),
            'alternatives_count': len(result.alternatives)
        }
        
        self.decision_log.append(log_entry)
        
        # 更新性能指标
        self.performance_metrics[result.decision_type.value].append(result.confidence)
        
        # A/B测试数据
        if context.ab_test_group:
            self.ab_test_results[context.ab_test_group][result.decision_type.value].append(result.confidence)
    
    def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """获取分析数据"""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_logs = [log for log in self.decision_log if log['timestamp'] > cutoff_time]
        
        if not recent_logs:
            return {'error': 'No recent data'}
        
        # 决策类型分布
        decision_distribution = defaultdict(int)
        strategy_distribution = defaultdict(int)
        avg_confidence_by_type = defaultdict(list)
        avg_execution_time = []
        
        for log in recent_logs:
            decision_distribution[log['decision_type']] += 1
            strategy_distribution[log['strategy']] += 1
            avg_confidence_by_type[log['decision_type']].append(log['confidence'])
            avg_execution_time.append(log['execution_time'])
        
        # 计算平均值
        confidence_stats = {}
        for decision_type, confidences in avg_confidence_by_type.items():
            confidence_stats[decision_type] = {
                'avg': sum(confidences) / len(confidences),
                'count': len(confidences)
            }
        
        return {
            'period_days': days,
            'total_decisions': len(recent_logs),
            'decision_distribution': dict(decision_distribution),
            'strategy_distribution': dict(strategy_distribution),
            'confidence_stats': confidence_stats,
            'avg_execution_time': sum(avg_execution_time) / len(avg_execution_time),
            'ab_test_results': dict(self.ab_test_results)
        }

class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self):
        self.active_tests = {
            'response_strategy_test': {
                'variants': ['concise', 'detailed', 'friendly'],
                'allocation': [0.33, 0.33, 0.34],
                'metric': 'confidence'
            },
            'clarification_approach_test': {
                'variants': ['direct', 'guided', 'progressive'],
                'allocation': [0.33, 0.33, 0.34],
                'metric': 'resolution_speed'
            }
        }
    
    def assign_variant(self, user_id: str, test_name: str) -> Optional[str]:
        """为用户分配A/B测试变体"""
        if test_name not in self.active_tests:
            return None
        
        # 基于用户ID的一致性哈希
        hash_value = int(hashlib.md5(f"{user_id}_{test_name}".encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0
        
        test_config = self.active_tests[test_name]
        variants = test_config['variants']
        allocations = test_config['allocation']
        
        cumulative = 0
        for variant, allocation in zip(variants, allocations):
            cumulative += allocation
            if normalized_hash <= cumulative:
                return variant
        
        return variants[-1]  # 回退到最后一个变体
    
    def get_optimal_strategy(self, user_id: str, context: DecisionContext) -> ResponseStrategy:
        """获取最佳响应策略"""
        # A/B测试分配
        strategy_variant = self.assign_variant(user_id, 'response_strategy_test')
        
        if strategy_variant == 'concise':
            return ResponseStrategy.CONCISE
        elif strategy_variant == 'detailed':
            return ResponseStrategy.DETAILED
        elif strategy_variant == 'friendly':
            return ResponseStrategy.FRIENDLY
        else:
            # 基于用户历史表现选择
            if context.performance_history:
                avg_performance = sum(context.performance_history) / len(context.performance_history)
                if avg_performance > 0.8:
                    return ResponseStrategy.EFFICIENT
                elif avg_performance < 0.5:
                    return ResponseStrategy.DETAILED
            
            return ResponseStrategy.CONVERSATIONAL

class DynamicTemplateManager:
    """动态模板管理器"""
    
    def __init__(self):
        self.base_templates = self._load_base_templates()
        self.template_performance = defaultdict(lambda: defaultdict(list))
        self.template_cache = {}
    
    def _load_base_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载基础模板"""
        return {
            'greeting': {
                ResponseStrategy.CONCISE: {
                    'es': '¡Hola! ¿Su orden?',
                    'en': 'Hello! Your order?',
                    'zh': '您好！要点什么？'
                },
                ResponseStrategy.DETAILED: {
                    'es': 'Bienvenido a Kong Food. Somos especialistas en comida asiática. ¿Qué le gustaría ordenar hoy?',
                    'en': 'Welcome to Kong Food. We specialize in Asian cuisine. What would you like to order today?',
                    'zh': '欢迎来到Kong Food。我们专门做亚洲料理。今天想点什么？'
                },
                ResponseStrategy.FRIENDLY: {
                    'es': '¡Hola! ¡Qué bueno verte! ¿Qué te provoca comer hoy?',
                    'en': 'Hello! Great to see you! What are you in the mood for today?',
                    'zh': '您好！很高兴见到您！今天想吃什么？'
                },
                ResponseStrategy.PROFESSIONAL: {
                    'es': 'Buenos días. Kong Food a su servicio. ¿En qué podemos asistirle?',
                    'en': 'Good day. Kong Food at your service. How may we assist you?',
                    'zh': '您好。Kong Food为您服务。需要什么帮助？'
                }
            },
            'order_success': {
                ResponseStrategy.CONCISE: {
                    'es': 'Confirmado: {order_summary}\nTotal: ${total}\n{prep_time} min.',
                    'en': 'Confirmed: {order_summary}\nTotal: ${total}\n{prep_time} min.',
                    'zh': '已确认：{order_summary}\n总计：${total}\n{prep_time}分钟'
                },
                ResponseStrategy.DETAILED: {
                    'es': 'Excelente elección, {name}. He registrado su orden:\n{order_summary}\n\nSubtotal: ${subtotal}\nImpuesto (11%): ${tax}\nTotal con impuesto: ${total}\n\nTiempo estimado de preparación: {prep_time} minutos.\nLe avisaremos cuando esté listo.',
                    'en': 'Excellent choice, {name}. I\'ve registered your order:\n{order_summary}\n\nSubtotal: ${subtotal}\nTax (11%): ${tax}\nTotal with tax: ${total}\n\nEstimated preparation time: {prep_time} minutes.\nWe\'ll notify you when ready.',
                    'zh': '很好的选择，{name}。已登记您的订单：\n{order_summary}\n\n小计：${subtotal}\n税费(11%)：${tax}\n含税总计：${total}\n\n预计准备时间：{prep_time}分钟\n准备好后会通知您。'
                },
                ResponseStrategy.FRIENDLY: {
                    'es': '¡Perfecto, {name}! Tu orden se ve deliciosa:\n{order_summary}\n\nTotal: ${total} (con impuesto incluido)\n\n¡Estará lista en {prep_time} minutos! 😊',
                    'en': 'Perfect, {name}! Your order looks delicious:\n{order_summary}\n\nTotal: ${total} (tax included)\n\nIt\'ll be ready in {prep_time} minutes! 😊',
                    'zh': '太好了，{name}！您的订单看起来很棒：\n{order_summary}\n\n总计：${total}（含税）\n\n{prep_time}分钟就好！😊'
                }
            },
            'upsell': {
                ResponseStrategy.CONVERSATIONAL: {
                    'es': '¡Excelente elección! ¿Te gustaría agregar {suggestion} por solo ${price} más?',
                    'en': 'Excellent choice! Would you like to add {suggestion} for just ${price} more?',
                    'zh': '很好的选择！要不要加个{suggestion}，只要${price}？'
                },
                ResponseStrategy.CONCISE: {
                    'es': '¿Agregar {suggestion}? +${price}',
                    'en': 'Add {suggestion}? +${price}',
                    'zh': '加{suggestion}？+${price}'
                }
            },
            'error_recovery': {
                ResponseStrategy.DETAILED: {
                    'es': 'Disculpe las molestias. Hubo un inconveniente técnico al procesar su orden. Por favor, permítame intentar nuevamente. ¿Podría repetir su pedido?',
                    'en': 'Sorry for the inconvenience. There was a technical issue processing your order. Please allow me to try again. Could you repeat your order?',
                    'zh': '很抱歉给您带来不便。处理您的订单时出现技术问题。请允许我重新尝试。能否重复一下您的订单？'
                },
                ResponseStrategy.CONCISE: {
                    'es': 'Error procesando. ¿Repetir orden?',
                    'en': 'Processing error. Repeat order?',
                    'zh': '处理错误。重复订单？'
                }
            }
        }
    
    def get_template(self, template_type: str, strategy: ResponseStrategy, 
                    language: str, **kwargs) -> str:
        """获取动态优化的模板"""
        
        # 检查缓存
        cache_key = f"{template_type}_{strategy.value}_{language}"
        
        # 获取基础模板
        templates = self.base_templates.get(template_type, {})
        strategy_templates = templates.get(strategy, templates.get(ResponseStrategy.CONVERSATIONAL, {}))
        
        if not strategy_templates:
            return self._get_fallback_template(language)
        
        template = strategy_templates.get(language, strategy_templates.get('es', ''))
        
        # 应用参数
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Template formatting error: {e}")
            return template
    
    def _get_fallback_template(self, language: str) -> str:
        """获取回退模板"""
        fallbacks = {
            'es': '¿En qué puedo ayudarte?',
            'en': 'How can I help you?',
            'zh': '需要什么帮助？'
        }
        return fallbacks.get(language, fallbacks['es'])
    
    def record_template_performance(self, template_type: str, strategy: ResponseStrategy,
                                  language: str, performance_score: float):
        """记录模板性能"""
        key = f"{template_type}_{strategy.value}_{language}"
        self.template_performance[key]['scores'].append(performance_score)
        self.template_performance[key]['timestamps'].append(time.time())
    
    def get_best_strategy_for_template(self, template_type: str, language: str) -> ResponseStrategy:
        """获取最佳策略"""
        best_strategy = ResponseStrategy.CONVERSATIONAL
        best_score = 0.0
        
        for strategy in ResponseStrategy:
            key = f"{template_type}_{strategy.value}_{language}"
            if key in self.template_performance:
                scores = self.template_performance[key]['scores']
                if scores:
                    avg_score = sum(scores[-10:]) / len(scores[-10:])  # 最近10次的平均分
                    if avg_score > best_score:
                        best_score = avg_score
                        best_strategy = strategy
        
        return best_strategy

class IntelligentOutputDirector:
    """智能输出决策引擎（优化版）"""
    
    def __init__(self):
        self.decision_logger = DecisionLogger()
        self.ab_test_manager = ABTestManager()
        self.template_manager = DynamicTemplateManager()
        self.context_memory = {}  # session_id -> DecisionContext
        
        # 决策规则简化
        self.decision_rules = self._build_simplified_rules()
        
        # 性能监控
        self.performance_tracker = {
            'total_decisions': 0,
            'successful_executions': 0,
            'clarification_rate': 0.0,
            'avg_decision_time': 0.0
        }
    
    def _build_simplified_rules(self) -> Dict[str, Any]:
        """构建简化的决策规则"""
        return {
            'greeting_threshold': 0.3,  # 低于此置信度需要问候
            'clarification_threshold': 0.6,  # 低于此置信度需要澄清
            'execution_threshold': 0.8,  # 高于此置信度可直接执行
            'upsell_probability': 0.3,  # 30%概率提供追加销售
            'max_clarifications': 3,  # 最大澄清次数
            'auto_execution_intents': ['confirm', 'yes']  # 自动执行的意图
        }
    
    def _get_or_create_context(self, session_id: str, user_id: str = "anonymous") -> DecisionContext:
        """获取或创建决策上下文"""
        if session_id not in self.context_memory:
            # 分配A/B测试组
            strategy = self.ab_test_manager.get_optimal_strategy(user_id, DecisionContext(session_id, user_id))
            ab_group = self.ab_test_manager.assign_variant(user_id, 'response_strategy_test')
            
            self.context_memory[session_id] = DecisionContext(
                session_id=session_id,
                user_id=user_id,
                response_strategy=strategy,
                ab_test_group=ab_group
            )
        
        return self.context_memory[session_id]
    
    def make_decision(self, co_struct: Dict[str, Any], path_data: Dict[str, Any] = None,
                     session_id: str = None, user_id: str = "anonymous") -> DecisionResult:
        """
        主决策函数（简化版）
        
        Args:
            co_struct: 对话对象结构
            path_data: 路径数据
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            决策结果
        """
        start_time = time.time()
        
        # 生成会话ID
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        
        # 获取上下文
        context = self._get_or_create_context(session_id, user_id)
        context.conversation_turn += 1
        
        # 更新性能追踪
        self.performance_tracker['total_decisions'] += 1
        
        language = co_struct.get('language', 'es')
        intent = co_struct.get('intent', 'inquiry')
        confidence = co_struct.get('confidence', 0.0)
        
        logger.info(f"Making decision - Intent: {intent}, Confidence: {confidence}, "
                   f"Session: {session_id}, Strategy: {context.response_strategy.value}")
        
        try:
            # 简化的决策流程
            decision_type, response = self._execute_simplified_decision_flow(
                co_struct, path_data, context, language, intent, confidence
            )
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 创建决策结果
            result = DecisionResult(
                response=response,
                decision_type=decision_type,
                strategy_used=context.response_strategy,
                confidence=self._calculate_decision_confidence(decision_type, confidence, path_data),
                execution_time=execution_time,
                metadata={
                    'session_id': session_id,
                    'conversation_turn': context.conversation_turn,
                    'ab_test_group': context.ab_test_group
                }
            )
            
            # 记录决策
            self.decision_logger.log_decision(context, result, co_struct, path_data)
            
            # 更新上下文
            context.previous_decisions.append(decision_type)
            context.performance_history.append(result.confidence)
            
            # 限制历史记录长度
            if len(context.previous_decisions) > 10:
                context.previous_decisions = context.previous_decisions[-10:]
            if len(context.performance_history) > 20:
                context.performance_history = context.performance_history[-20:]
            
            return result
            
        except Exception as e:
            logger.error(f"Decision making error: {e}")
            execution_time = time.time() - start_time
            
            # 错误回退
            fallback_response = self._get_error_fallback(language, context.response_strategy)
            
            return DecisionResult(
                response=fallback_response,
                decision_type=DecisionType.ERROR_HANDLING,
                strategy_used=context.response_strategy,
                confidence=0.3,
                execution_time=execution_time,
                metadata={'error': str(e)}
            )
    
    def _execute_simplified_decision_flow(self, co_struct: Dict[str, Any], path_data: Dict[str, Any],
                                        context: DecisionContext, language: str, intent: str,
                                        confidence: float) -> Tuple[DecisionType, str]:
        """执行简化的决策流程"""
        
        # 1. 特殊意图处理
        if intent == 'cancel':
            return DecisionType.CLOSING, self._get_closing_message(language, context.response_strategy)
        
        # 2. 问候检查
        if self._needs_greeting(co_struct, context):
            return DecisionType.GREETING, self._get_greeting_message(language, context.response_strategy)
        
        # 3. 订单处理
        if intent in ['order', 'modify'] and path_data:
            return self._handle_order_intent(co_struct, path_data, context, language)
        
        # 4. 确认处理
        if intent in self.decision_rules['auto_execution_intents']:
            return DecisionType.CONFIRMATION, self._get_confirmation_message(language, context.response_strategy)
        
        # 5. 默认处理 - 智能响应
        return DecisionType.FALLBACK, self._get_intelligent_fallback(co_struct, path_data, context, language)
    
    def _handle_order_intent(self, co_struct: Dict[str, Any], path_data: Dict[str, Any],
                           context: DecisionContext, language: str) -> Tuple[DecisionType, str]:
        """处理订单意图"""
        
        # 计算张力分数
        tension_score = score(path_data, context.user_id)
        action = get_action_recommendation(tension_score, context.user_id)
        
        logger.info(f"Order handling - Tension: {tension_score}, Action: {action}")
        
        # 开始会话跟踪
        start_session_tracking(context.session_id, co_struct.get('confidence', 0.0), 
                             path_data.get('score', 0.0))
        
        if action == 'clarify':
            # 澄清处理
            add_clarification(context.session_id)
            
            # 检查澄清次数限制
            clarification_count = sum(1 for d in context.previous_decisions if d == DecisionType.CLARIFICATION)
            
            if clarification_count >= self.decision_rules['max_clarifications']:
                # 强制执行或提供简化选择
                return DecisionType.EXECUTION, self._force_simple_execution(co_struct, path_data, context, language)
            
            # 使用增强澄清引擎
            clarification = generate_enhanced_clarification(
                co_struct, path_data, context.session_id, context.user_id
            )
            
            return DecisionType.CLARIFICATION, clarification['message']
            
        elif action == 'execute':
            # 直接执行
            return self._execute_order(co_struct, path_data, context, language)
        
        else:  # analyze
            # 需要更多分析
            return DecisionType.FALLBACK, self._get_analysis_response(co_struct, path_data, context, language)
    
    def _execute_order(self, co_struct: Dict[str, Any], path_data: Dict[str, Any],
                      context: DecisionContext, language: str) -> Tuple[DecisionType, str]:
        """执行订单"""
        try:
            # 处理订单
            order_result = place_order({
                'path_data': path_data,
                'customer_phone': co_struct.get('customer_phone'),
                'customer_name': co_struct.get('customer_name', 'Cliente'),
                'session_id': context.session_id
            })
            
            if order_result.get('success'):
                # 订单成功
                complete_session(context.session_id, True, 1.0, context.user_id)
                self.performance_tracker['successful_executions'] += 1
                
                # 生成成功响应
                success_message = self._generate_order_success_message(
                    co_struct, path_data, context, language, order_result
                )
                
                # 检查是否添加追加销售
                if self._should_upsell(context):
                    upsell_message = self._generate_upsell_message(path_data, context, language)
                    if upsell_message:
                        success_message += f"\n\n{upsell_message}"
                
                return DecisionType.EXECUTION, success_message
            else:
                # 订单失败
                complete_session(context.session_id, False, 0.0, context.user_id)
                error_message = self.template_manager.get_template(
                    'error_recovery', context.response_strategy, language
                )
                return DecisionType.ERROR_HANDLING, error_message
                
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            complete_session(context.session_id, False, 0.0, context.user_id)
            error_message = self.template_manager.get_template(
                'error_recovery', context.response_strategy, language
            )
            return DecisionType.ERROR_HANDLING, error_message
    
    def _generate_order_success_message(self, co_struct: Dict[str, Any], path_data: Dict[str, Any],
                                      context: DecisionContext, language: str,
                                      order_result: Dict[str, Any]) -> str:
        """生成订单成功消息"""
        
        order_items = path_data.get('path', [])
        subtotal = sum(item.get('price', 0) * item.get('quantity', 1) for item in order_items)
        tax_rate = 0.11
        tax_amount = subtotal * tax_rate
        total_with_tax = subtotal + tax_amount
        prep_time = self._calculate_prep_time(order_items)
        
        order_summary = self._format_order_summary(order_items, language, context.response_strategy)
        customer_name = co_struct.get('customer_name', 'Cliente')
        
        template_vars = {
            'name': customer_name,
            'order_summary': order_summary,
            'subtotal': f"{subtotal:.2f}",
            'tax': f"{tax_amount:.2f}",
            'total': f"{total_with_tax:.2f}",
            'prep_time': prep_time
        }
        
        return self.template_manager.get_template(
            'order_success', context.response_strategy, language, **template_vars
        )
    
    def _should_upsell(self, context: DecisionContext) -> bool:
        """判断是否应该追加销售"""
        # 基于A/B测试和用户历史
        if context.ab_test_group == 'detailed':
            return True
        elif context.response_strategy == ResponseStrategy.CONCISE:
            return False
        else:
            import random
            return random.random() < self.decision_rules['upsell_probability']
    
    def _generate_upsell_message(self, path_data: Dict[str, Any], context: DecisionContext,
                               language: str) -> Optional[str]:
        """生成追加销售消息"""
        # 简单的追加销售逻辑
        suggestions = {
            'es': ('bebida', '2.99'),
            'en': ('drink', '2.99'),
            'zh': ('饮料', '2.99')
        }
        
        suggestion, price = suggestions.get(language, suggestions['es'])
        
        return self.template_manager.get_template(
            'upsell', context.response_strategy, language,
            suggestion=suggestion, price=price
        )
    
    def _needs_greeting(self, co_struct: Dict[str, Any], context: DecisionContext) -> bool:
        """判断是否需要问候"""
        # 简化的问候逻辑
        if context.conversation_turn == 1:
            intent = co_struct.get('intent', '')
            confidence = co_struct.get('confidence', 0.0)
            
            # 如果用户意图不明确或置信度过低
            if intent not in ['order'] or confidence < self.decision_rules['greeting_threshold']:
                return True
        
        return False
    
    def _get_greeting_message(self, language: str, strategy: ResponseStrategy) -> str:
        """获取问候消息"""
        return self.template_manager.get_template('greeting', strategy, language)
    
    def _get_confirmation_message(self, language: str, strategy: ResponseStrategy) -> str:
        """获取确认消息"""
        messages = {
            ResponseStrategy.CONCISE: {
                'es': 'Procesando...',
                'en': 'Processing...',
                'zh': '处理中...'
            },
            ResponseStrategy.DETAILED: {
                'es': 'Perfecto, estoy procesando su orden ahora mismo.',
                'en': 'Perfect, I\'m processing your order right now.',
                'zh': '好的，我现在就为您处理订单。'
            },
            ResponseStrategy.FRIENDLY: {
                'es': '¡Excelente! Procesando tu orden 😊',
                'en': 'Excellent! Processing your order 😊',
                'zh': '太好了！正在处理您的订单 😊'
            }
        }
        
        strategy_messages = messages.get(strategy, messages[ResponseStrategy.CONVERSATIONAL])
        return strategy_messages.get(language, strategy_messages['es'])
    
    def _get_closing_message(self, language: str, strategy: ResponseStrategy) -> str:
        """获取结束消息"""
        messages = {
            ResponseStrategy.CONCISE: {
                'es': '¡Gracias!',
                'en': 'Thank you!',
                'zh': '谢谢！'
            },
            ResponseStrategy.DETAILED: {
                'es': 'Muchas gracias por elegir Kong Food. ¡Que tenga un excelente día!',
                'en': 'Thank you very much for choosing Kong Food. Have an excellent day!',
                'zh': '非常感谢您选择Kong Food。祝您有美好的一天！'
            },
            ResponseStrategy.FRIENDLY: {
                'es': '¡Muchas gracias! ¡Hasta la próxima! 👋',
                'en': 'Thank you so much! See you next time! 👋',
                'zh': '非常感谢！下次见！👋'
            }
        }
        
        strategy_messages = messages.get(strategy, messages[ResponseStrategy.CONVERSATIONAL])
        return strategy_messages.get(language, strategy_messages['es'])
    
    def _get_intelligent_fallback(self, co_struct: Dict[str, Any], path_data: Dict[str, Any],
                                context: DecisionContext, language: str) -> str:
        """获取智能回退响应"""
        
        # 尝试使用Claude生成智能响应
        if hasattr(self, '_call_claude_for_response'):
            try:
                menu_candidates = self._extract_menu_candidates(path_data) if path_data else "Full menu available"
                context_info = f"""
User input: {co_struct.get('raw_text', '')}
Intent: {co_struct.get('intent', 'unknown')}
Confidence: {co_struct.get('confidence', 0.0)}
Conversation turn: {context.conversation_turn}
Strategy: {context.response_strategy.value}
"""
                
                response = self._call_claude_for_response(context_info, menu_candidates, language)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Claude fallback failed: {e}")
        
        # 基础回退
        return self._get_basic_fallback(language, context.response_strategy)
    
    def _get_basic_fallback(self, language: str, strategy: ResponseStrategy) -> str:
        """获取基础回退响应"""
        fallbacks = {
            ResponseStrategy.CONCISE: {
                'es': '¿Su orden?',
                'en': 'Your order?',
                'zh': '您的订单？'
            },
            ResponseStrategy.DETAILED: {
                'es': 'Disculpe, no estoy seguro de entender completamente. ¿Podría decirme qué le gustaría ordenar?',
                'en': 'Sorry, I\'m not sure I understand completely. Could you tell me what you\'d like to order?',
                'zh': '抱歉，我不太确定理解。您能告诉我想要点什么吗？'
            },
            ResponseStrategy.FRIENDLY: {
                'es': '¡Hola! ¿En qué te puedo ayudar hoy? 😊',
                'en': 'Hi there! How can I help you today? 😊',
                'zh': '您好！今天我能帮您什么？😊'
            }
        }
        
        strategy_fallbacks = fallbacks.get(strategy, fallbacks[ResponseStrategy.CONVERSATIONAL])
        return strategy_fallbacks.get(language, strategy_fallbacks['es'])
    
    def _get_error_fallback(self, language: str, strategy: ResponseStrategy) -> str:
        """获取错误回退响应"""
        return self.template_manager.get_template('error_recovery', strategy, language)
    
    def _get_analysis_response(self, co_struct: Dict[str, Any], path_data: Dict[str, Any],
                             context: DecisionContext, language: str) -> str:
        """获取分析响应"""
        # 基于策略返回不同的分析响应
        if context.response_strategy == ResponseStrategy.CONCISE:
            return self._get_basic_fallback(language, context.response_strategy)
        else:
            # 提供更详细的分析
            if language == 'zh':
                return "让我为您分析一下选项..."
            elif language == 'en':
                return "Let me analyze the options for you..."
            else:
                return "Déjeme analizar las opciones..."
    
    def _force_simple_execution(self, co_struct: Dict[str, Any], path_data: Dict[str, Any],
                               context: DecisionContext, language: str) -> str:
        """强制简单执行"""
        # 当澄清次数过多时，提供简化的选择
        path = path_data.get('path', [])
        if path:
            item = path[0]  # 选择第一个选项
            item_name = item.get('item_name', '')
            
            if language == 'zh':
                return f"我为您选择了{item_name}，如果需要修改请告诉我。"
            elif language == 'en':
                return f"I've selected {item_name} for you. Let me know if you'd like to change it."
            else:
                return f"He seleccionado {item_name} para usted. Avíseme si quiere cambiarlo."
        
        return self._get_basic_fallback(language, context.response_strategy)
    
    def _calculate_prep_time(self, order_items: List[Dict[str, Any]]) -> int:
        """计算准备时间"""
        base_time = 8  # 基础时间
        item_count = sum(item.get('quantity', 1) for item in order_items)
        
        if item_count <= 2:
            return base_time
        elif item_count <= 5:
            return base_time + 3
        else:
            return base_time + 7
    
    def _format_order_summary(self, order_items: List[Dict[str, Any]], 
                            language: str, strategy: ResponseStrategy) -> str:
        """格式化订单摘要"""
        if not order_items:
            return ""
        
        summary_lines = []
        for item in order_items:
            quantity = item.get('quantity', 1)
            name = item.get('item_name', '')
            price = item.get('price', 0.0)
            
            if strategy == ResponseStrategy.CONCISE:
                # 简洁格式
                if language == 'zh':
                    line = f"{quantity}x {name}"
                elif language == 'en':
                    line = f"{quantity}x {name}"
                else:
                    line = f"{quantity}x {name}"
            else:
                # 详细格式
                if language == 'zh':
                    line = f"• {quantity}份 {name} - ${price * quantity:.2f}"
                elif language == 'en':
                    line = f"• {quantity}x {name} - ${price * quantity:.2f}"
                else:
                    line = f"• {quantity}x {name} - ${price * quantity:.2f}"
            
            summary_lines.append(line)
        
        return '\n'.join(summary_lines)
    
    def _calculate_decision_confidence(self, decision_type: DecisionType, 
                                     input_confidence: float, path_data: Dict[str, Any]) -> float:
        """计算决策置信度"""
        base_confidence = input_confidence
        
        # 基于决策类型调整
        type_adjustments = {
            DecisionType.EXECUTION: 0.1,
            DecisionType.CLARIFICATION: -0.1,
            DecisionType.ERROR_HANDLING: -0.3,
            DecisionType.GREETING: 0.0,
            DecisionType.FALLBACK: -0.2
        }
        
        adjustment = type_adjustments.get(decision_type, 0.0)
        
        # 基于路径数据调整
        if path_data:
            path_confidence = path_data.get('confidence', 0.5)
            base_confidence = (base_confidence + path_confidence) / 2
        
        final_confidence = base_confidence + adjustment
        return max(0.1, min(1.0, final_confidence))
    
    def _extract_menu_candidates(self, path_data: Dict[str, Any]) -> str:
        """从路径数据中提取菜单候选项"""
        candidates = []
        
        if path_data and path_data.get('path'):
            for item in path_data['path'][:3]:  # 最多3个主要选项
                candidate = f"- {item.get('item_name', '')} (${item.get('price', 0.0)})"
                candidates.append(candidate)
        
        if path_data and path_data.get('alternative_paths'):
            for alt_path in path_data['alternative_paths'][:2]:  # 最多2个备选路径
                for match in alt_path.get('matches', [])[:2]:  # 每个路径最多2个匹配
                    candidate = f"- {match.get('item_name', '')} (${match.get('price', 0.0)})"
                    if candidate not in candidates:
                        candidates.append(candidate)
        
        return '\n'.join(candidates) if candidates else "No specific menu items found"
    
    def _call_claude_for_response(self, context: str, menu_candidates: str, language: str) -> Optional[str]:
        """调用Claude生成响应"""
        try:
            language_names = {'es': 'Spanish', 'en': 'English', 'zh': 'Chinese'}
            lang_name = language_names.get(language, 'Spanish')
            
            system_prompt = f"""
You are Kong Food AI ordering assistant. Follow these rules:

1. LANGUAGE: Respond ONLY in {lang_name}
2. MENU CONSTRAINT: Only suggest items from: {menu_candidates}
3. BE CONCISE: Keep responses under 50 words
4. BE HELPFUL: Guide the customer to place an order
5. NO HALLUCINATION: Never mention items not in the menu candidates

Context: {context}

Respond naturally as a restaurant ordering assistant.
"""
            
            response = ask_claude(system_prompt)
            return response if response and len(response) < 200 else None
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """获取性能分析"""
        decision_analytics = self.decision_logger.get_analytics()
        
        # 添加性能追踪数据
        decision_analytics.update({
            'performance_tracker': self.performance_tracker,
            'active_sessions': len(self.context_memory),
            'ab_test_groups': self.ab_test_manager.active_tests
        })
        
        return decision_analytics
    
    def reset_session(self, session_id: str) -> bool:
        """重置会话"""
        if session_id in self.context_memory:
            del self.context_memory[session_id]
            return True
        return False
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """更新用户偏好"""
        # 更新所有该用户的活跃会话
        for context in self.context_memory.values():
            if context.user_id == user_id:
                context.user_preferences.update(preferences)
                
                # 根据偏好调整策略
                if preferences.get('preferred_style') == 'concise':
                    context.response_strategy = ResponseStrategy.CONCISE
                elif preferences.get('preferred_style') == 'detailed':
                    context.response_strategy = ResponseStrategy.DETAILED

# 全局导演实例
_global_director = IntelligentOutputDirector()

# 保持向后兼容的接口
def reply(co_struct: Dict[str, Any], path_data: Dict[str, Any] = None, 
          session_id: str = None) -> str:
    """
    主决策函数 - 决定如何响应（保持向后兼容）
    
    Args:
        co_struct: 对话对象结构
        path_data: 路径数据
        session_id: 会话ID
        
    Returns:
        响应消息
    """
    result = _global_director.make_decision(co_struct, path_data, session_id)
    return result.response

# 新增的增强接口
def make_intelligent_decision(co_struct: Dict[str, Any], path_data: Dict[str, Any] = None,
                            session_id: str = None, user_id: str = "anonymous") -> Dict[str, Any]:
    """
    智能决策函数（新接口）
    
    Returns:
        完整的决策结果字典
    """
    result = _global_director.make_decision(co_struct, path_data, session_id, user_id)
    
    return {
        'response': result.response,
        'decision_type': result.decision_type.value,
        'strategy_used': result.strategy_used.value,
        'confidence': result.confidence,
        'execution_time': result.execution_time,
        'alternatives': result.alternatives,
        'metadata': result.metadata
    }

def get_decision_analytics() -> Dict[str, Any]:
    """获取决策分析数据（新接口）"""
    return _global_director.get_performance_analytics()

def reset_decision_session(session_id: str) -> bool:
    """重置决策会话（新接口）"""
    return _global_director.reset_session(session_id)

def update_user_decision_preferences(user_id: str, preferences: Dict[str, Any]):
    """更新用户决策偏好（新接口）"""
    _global_director.update_user_preferences(user_id, preferences)

def get_optimal_strategy_for_user(user_id: str) -> str:
    """获取用户最佳策略（新接口）"""
    context = DecisionContext(session_id="temp", user_id=user_id)
    strategy = _global_director.ab_test_manager.get_optimal_strategy(user_id, context)
    return strategy.value

# 保持向后兼容的便捷函数
def reply_with_clarification(co: Dict[str, Any], path_data: Dict[str, Any]) -> str:
    """生成澄清响应（保持兼容）"""
    return build_clarification(co, path_data)

def reply_with_confirmation(co: Dict[str, Any], order_items: List[Dict[str, Any]], total: float) -> str:
    """生成确认响应（保持兼容）"""
    return build_order_confirmation(co, order_items, total)

def reply_simple(message: str, language: str = 'es') -> str:
    """生成简单响应（保持兼容）"""
    director = IntelligentOutputDirector()
    return director._get_basic_fallback(language, ResponseStrategy.CONVERSATIONAL)

# 测试和基准函数
def run_comprehensive_tests():
    """运行综合测试"""
    print("=== 智能输出决策引擎综合测试 ===\n")
    
    # 1. 向后兼容性测试
    print("1. 向后兼容性测试:")
    
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
    
    # 原有接口测试
    old_response = reply(test_co, test_path)
    print(f"   原有接口响应: {old_response[:80]}...")
    
    # 2. 智能决策测试
    print("\n2. 智能决策测试:")
    
    intelligent_result = make_intelligent_decision(test_co, test_path, "test_session", "test_user")
    
    print(f"   决策类型: {intelligent_result['decision_type']}")
    print(f"   使用策略: {intelligent_result['strategy_used']}")
    print(f"   置信度: {intelligent_result['confidence']}")
    print(f"   执行时间: {intelligent_result['execution_time']:.4f}s")
    print(f"   响应: {intelligent_result['response'][:80]}...")
    
    # 3. A/B测试演示
    print("\n3. A/B测试演示:")
    
    users = ['user_concise', 'user_detailed', 'user_friendly']
    for user in users:
        strategy = get_optimal_strategy_for_user(user)
        result = make_intelligent_decision(test_co, test_path, f"session_{user}", user)
        print(f"   {user}: 策略={strategy}, 响应={result['response'][:50]}...")
    
    # 4. 不同决策场景测试
    print("\n4. 不同决策场景测试:")
    
    scenarios = [
        {
            'name': '问候场景',
            'co': {'intent': 'greeting', 'confidence': 0.3, 'language': 'es', 'raw_text': 'hola'},
            'path': None
        },
        {
            'name': '澄清场景',
            'co': {'intent': 'order', 'confidence': 0.4, 'language': 'es', 'raw_text': 'pollo'},
            'path': {'requires_clarification': True, 'confidence': 0.4, 'clarification_reason': 'multiple_matches'}
        },
        {
            'name': '取消场景',
            'co': {'intent': 'cancel', 'confidence': 0.9, 'language': 'es', 'raw_text': 'cancelar'},
            'path': None
        },
        {
            'name': '错误场景',
            'co': {'intent': 'unknown', 'confidence': 0.1, 'language': 'es', 'raw_text': 'asdfgh'},
            'path': None
        }
    ]
    
    for scenario in scenarios:
        result = make_intelligent_decision(scenario['co'], scenario['path'])
        print(f"   {scenario['name']}: {result['decision_type']} - {result['response'][:50]}...")
    
    # 5. 多语言测试
    print("\n5. 多语言测试:")
    
    languages = ['es', 'en', 'zh']
    for lang in languages:
        test_co_lang = test_co.copy()
        test_co_lang['language'] = lang
        
        result = make_intelligent_decision(test_co_lang, test_path)
        print(f"   {lang}: {result['response'][:60]}...")
    
    # 6. 性能分析
    print("\n6. 性能分析:")
    
    analytics = get_decision_analytics()
    print(f"   总决策数: {analytics.get('total_decisions', 0)}")
    print(f"   活跃会话: {analytics.get('active_sessions', 0)}")
    
    if 'decision_distribution' in analytics:
        print("   决策类型分布:")
        for decision_type, count in analytics['decision_distribution'].items():
            print(f"     {decision_type}: {count}")
    
    if 'performance_tracker' in analytics:
        tracker = analytics['performance_tracker']
        print(f"   成功执行率: {tracker.get('successful_executions', 0) / max(tracker.get('total_decisions', 1), 1):.2%}")
    
    # 7. 用户偏好测试
    print("\n7. 用户偏好测试:")
    
    # 设置用户偏好
    update_user_decision_preferences("pref_user", {"preferred_style": "concise"})
    
    result_before = make_intelligent_decision(test_co, test_path, "pref_session", "pref_user")
    print(f"   偏好设置后: {result_before['strategy_used']} - {result_before['response'][:50]}...")
    
    print("\n=== 测试完成 ===")

# 主程序兼容性测试
if __name__ == "__main__":
    # 保持原有测试的向后兼容性
    print("=== 向后兼容性验证 ===")
    
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
    
    print("\n=== 智能功能演示 ===")
    run_comprehensive_tests()
