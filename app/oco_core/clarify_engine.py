#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Clarify Engine (优化版)
智能多语言澄清生成引擎 - 动态模板、上下文感知、多轮对话、LLM集成
"""

import json
import random
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
from functools import lru_cache

from ..config import settings

class ClarificationType(Enum):
    """澄清类型枚举"""
    MULTIPLE_MATCHES = "multiple_matches"
    LOW_CONFIDENCE = "low_confidence"
    QUANTITY_AMBIGUOUS = "quantity_ambiguous"
    SIZE_SELECTION = "size_selection"
    MODIFICATION_UNCLEAR = "modification_unclear"
    COMBO_CHOICE = "combo_choice"
    NO_MATCHES = "no_matches"
    CONTEXT_MISSING = "context_missing"
    LANGUAGE_SWITCH = "language_switch"
    ORDER_CONFIRMATION = "order_confirmation"
    PROGRESSIVE_DISAMBIGUATION = "progressive_disambiguation"

class ClarificationStrategy(Enum):
    """澄清策略枚举"""
    DIRECT_QUESTION = "direct_question"
    MULTIPLE_CHOICE = "multiple_choice"
    GUIDED_NARROWING = "guided_narrowing"
    CONTEXTUAL_SUGGESTION = "contextual_suggestion"
    PROGRESSIVE_REFINEMENT = "progressive_refinement"
    SMART_DEFAULT = "smart_default"
    CONVERSATIONAL_FLOW = "conversational_flow"

@dataclass
class ClarificationTemplate:
    """增强的澄清模板"""
    id: str
    type: ClarificationType
    strategy: ClarificationStrategy
    spanish: str
    english: str
    chinese: str
    context: str
    priority: int = 1  # 优先级，数字越小优先级越高
    conditions: Dict[str, Any] = field(default_factory=dict)  # 使用条件
    variables: List[str] = field(default_factory=list)  # 模板变量
    follow_up_templates: List[str] = field(default_factory=list)  # 后续模板ID

@dataclass
class ClarificationContext:
    """澄清上下文"""
    session_id: str
    user_id: str = "anonymous"
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    clarification_count: int = 0
    last_clarification_type: Optional[ClarificationType] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    failed_attempts: List[str] = field(default_factory=list)
    current_disambiguation_path: List[str] = field(default_factory=list)
    context_memory: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneratedClarification:
    """生成的澄清结果"""
    message: str
    type: ClarificationType
    strategy: ClarificationStrategy
    confidence: float
    expected_response_types: List[str]
    timeout_seconds: int = 30
    fallback_message: Optional[str] = None
    context_updates: Dict[str, Any] = field(default_factory=dict)

class DynamicTemplateGenerator:
    """动态模板生成器"""
    
    def __init__(self):
        self.pattern_library = self._build_pattern_library()
        self.context_patterns = self._build_context_patterns()
        
    def _build_pattern_library(self) -> Dict[str, Dict[str, List[str]]]:
        """构建模式库"""
        return {
            'question_starters': {
                'es': ['¿Quieres', '¿Te refieres a', '¿Prefieres', '¿Cuál', '¿Podrías'],
                'en': ['Do you want', 'Do you mean', 'Would you prefer', 'Which', 'Could you'],
                'zh': ['您想要', '您是指', '您比较喜欢', '哪个', '您能否']
            },
            'option_connectors': {
                'es': [' o ', ', ', ' - también tenemos '],
                'en': [' or ', ', ', ' - we also have '],
                'zh': ['或者', '，', ' - 我们还有']
            },
            'politeness_phrases': {
                'es': ['por favor', 'disculpa', 'perdón', 'si no te molesta'],
                'en': ['please', 'sorry', 'excuse me', 'if you don\'t mind'],
                'zh': ['请', '抱歉', '不好意思', '如果您不介意的话']
            },
            'uncertainty_expressions': {
                'es': ['no estoy seguro', 'no entiendo bien', 'podrías aclarar'],
                'en': ['I\'m not sure', 'I don\'t quite understand', 'could you clarify'],
                'zh': ['我不确定', '我不太明白', '您能澄清一下吗']
            }
        }
    
    def _build_context_patterns(self) -> Dict[str, Dict[str, str]]:
        """构建上下文模式"""
        return {
            'time_based': {
                'morning': {'es': 'buenos días', 'en': 'good morning', 'zh': '早上好'},
                'afternoon': {'es': 'buenas tardes', 'en': 'good afternoon', 'zh': '下午好'},
                'evening': {'es': 'buenas noches', 'en': 'good evening', 'zh': '晚上好'}
            },
            'frequency_based': {
                'first_time': {'es': 'Como es tu primera vez', 'en': 'Since it\'s your first time', 'zh': '由于您是第一次'},
                'returning': {'es': 'Como siempre', 'en': 'As usual', 'zh': '像往常一样'},
                'frequent': {'es': 'Como de costumbre', 'en': 'As customary', 'zh': '按照习惯'}
            }
        }
    
    def generate_template(self, clarification_type: ClarificationType, 
                         language: str, context: ClarificationContext,
                         variables: Dict[str, Any]) -> str:
        """动态生成模板"""
        
        # 基于类型选择核心结构
        core_structure = self._get_core_structure(clarification_type, language)
        
        # 添加上下文适应
        context_prefix = self._generate_context_prefix(context, language)
        
        # 添加个性化元素
        personalization = self._add_personalization(context, language)
        
        # 组合最终模板
        template_parts = [
            context_prefix,
            core_structure,
            personalization
        ]
        
        template = ' '.join(filter(None, template_parts))
        
        # 应用变量替换
        return self._apply_variables(template, variables)
    
    def _get_core_structure(self, clarification_type: ClarificationType, language: str) -> str:
        """获取核心结构"""
        structures = {
            ClarificationType.MULTIPLE_MATCHES: {
                'es': 'Encontré varias opciones: {options}. ¿Cuál prefieres?',
                'en': 'I found several options: {options}. Which do you prefer?',
                'zh': '我找到了几个选项：{options}。您比较喜欢哪个？'
            },
            ClarificationType.LOW_CONFIDENCE: {
                'es': 'No estoy completamente seguro de lo que buscas. ¿Podrías ser más específico sobre "{query}"?',
                'en': 'I\'m not completely sure what you\'re looking for. Could you be more specific about "{query}"?',
                'zh': '我不太确定您要找什么。您能更具体地说明一下"{query}"吗？'
            },
            ClarificationType.QUANTITY_AMBIGUOUS: {
                'es': '¿Cuántos {item} te gustaría?',
                'en': 'How many {item} would you like?',
                'zh': '您要几个{item}？'
            }
        }
        
        return structures.get(clarification_type, {}).get(language, structures[clarification_type]['es'])
    
    def _generate_context_prefix(self, context: ClarificationContext, language: str) -> str:
        """生成上下文前缀"""
        if context.clarification_count == 0:
            return ""
        elif context.clarification_count == 1:
            return self._get_phrase('clarification_follow_up_1', language)
        else:
            return self._get_phrase('clarification_follow_up_multiple', language)
    
    def _add_personalization(self, context: ClarificationContext, language: str) -> str:
        """添加个性化元素"""
        personalization = ""
        
        # 基于用户偏好添加建议
        if context.user_preferences.get('interaction_style') == 'helpful':
            personalization = self._get_phrase('helpful_suggestion', language)
        elif context.user_preferences.get('interaction_style') == 'concise':
            return ""  # 简洁模式不添加额外内容
        
        return personalization
    
    def _get_phrase(self, phrase_type: str, language: str) -> str:
        """获取短语"""
        phrases = {
            'clarification_follow_up_1': {
                'es': 'Déjame intentar de nuevo.',
                'en': 'Let me try again.',
                'zh': '让我再试一次。'
            },
            'clarification_follow_up_multiple': {
                'es': 'Para asegurarme de entender bien,',
                'en': 'To make sure I understand correctly,',
                'zh': '为了确保我理解正确，'
            },
            'helpful_suggestion': {
                'es': '¿Te ayudo con algo más?',
                'en': 'Can I help you with anything else?',
                'zh': '还有什么我可以帮您的吗？'
            }
        }
        
        return phrases.get(phrase_type, {}).get(language, "")
    
    def _apply_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """应用变量替换"""
        for key, value in variables.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template

class ContextAwareClarificationEngine:
    """上下文感知澄清引擎"""
    
    def __init__(self):
        self.context_memory = {}  # session_id -> ClarificationContext
        self.template_generator = DynamicTemplateGenerator()
        self.strategy_selector = ClarificationStrategySelector()
        
    def process_clarification_request(self, co: Dict[str, Any], path_data: Dict[str, Any],
                                    session_id: str, user_id: str = "anonymous") -> GeneratedClarification:
        """处理澄清请求"""
        
        # 获取或创建上下文
        context = self._get_or_create_context(session_id, user_id)
        context.clarification_count += 1
        
        # 分析当前情况
        clarification_type = self._analyze_clarification_need(co, path_data, context)
        
        # 选择策略
        strategy = self.strategy_selector.select_strategy(clarification_type, context, co, path_data)
        
        # 生成澄清消息
        clarification = self._generate_clarification_message(
            clarification_type, strategy, co, path_data, context
        )
        
        # 更新上下文
        self._update_context(context, clarification_type, clarification)
        
        return clarification
    
    def _get_or_create_context(self, session_id: str, user_id: str) -> ClarificationContext:
        """获取或创建澄清上下文"""
        if session_id not in self.context_memory:
            self.context_memory[session_id] = ClarificationContext(
                session_id=session_id,
                user_id=user_id
            )
        return self.context_memory[session_id]
    
    def _analyze_clarification_need(self, co: Dict[str, Any], path_data: Dict[str, Any],
                                  context: ClarificationContext) -> ClarificationType:
        """分析澄清需求类型"""
        
        # 基于路径数据的分析
        clarification_reason = path_data.get('clarification_reason', '')
        alternative_paths = path_data.get('alternative_paths', [])
        confidence = path_data.get('confidence', 0.0)
        
        # 考虑历史上下文
        if context.clarification_count > 2:
            return ClarificationType.PROGRESSIVE_DISAMBIGUATION
        
        # 分析逻辑
        if clarification_reason == 'multiple_matches' and len(alternative_paths) > 1:
            return ClarificationType.MULTIPLE_MATCHES
        elif clarification_reason == 'low_confidence' or confidence < 0.4:
            return ClarificationType.LOW_CONFIDENCE
        elif clarification_reason == 'no_menu_matches':
            return ClarificationType.NO_MATCHES
        elif self._has_quantity_ambiguity(co):
            return ClarificationType.QUANTITY_AMBIGUOUS
        elif self._has_size_ambiguity(path_data):
            return ClarificationType.SIZE_SELECTION
        else:
            return ClarificationType.CONTEXT_MISSING
    
    def _has_quantity_ambiguity(self, co: Dict[str, Any]) -> bool:
        """检查是否有数量歧义"""
        objects = co.get('objects', [])
        for obj in objects:
            if obj.get('quantity', 1) > 10 or obj.get('quantity', 1) <= 0:
                return True
        return False
    
    def _has_size_ambiguity(self, path_data: Dict[str, Any]) -> bool:
        """检查是否有尺寸歧义"""
        path = path_data.get('path', [])
        for item in path:
            if 'size_options' in item or 'variants' in item:
                return True
        return False
    
    def _generate_clarification_message(self, clarification_type: ClarificationType,
                                      strategy: ClarificationStrategy, co: Dict[str, Any],
                                      path_data: Dict[str, Any], context: ClarificationContext) -> GeneratedClarification:
        """生成澄清消息"""
        
        language = co.get('language', 'es')
        
        # 准备变量
        variables = self._prepare_template_variables(co, path_data, context)
        
        # 使用策略生成消息
        if strategy == ClarificationStrategy.MULTIPLE_CHOICE:
            message = self._generate_multiple_choice_message(clarification_type, language, variables, context)
        elif strategy == ClarificationStrategy.GUIDED_NARROWING:
            message = self._generate_guided_narrowing_message(clarification_type, language, variables, context)
        elif strategy == ClarificationStrategy.PROGRESSIVE_REFINEMENT:
            message = self._generate_progressive_refinement_message(clarification_type, language, variables, context)
        else:
            message = self.template_generator.generate_template(clarification_type, language, context, variables)
        
        # 生成预期响应类型
        expected_response_types = self._get_expected_response_types(clarification_type, strategy)
        
        # 计算置信度
        confidence = self._calculate_clarification_confidence(clarification_type, strategy, context)
        
        return GeneratedClarification(
            message=message,
            type=clarification_type,
            strategy=strategy,
            confidence=confidence,
            expected_response_types=expected_response_types,
            timeout_seconds=self._get_timeout_for_type(clarification_type),
            fallback_message=self._generate_fallback_message(language),
            context_updates={"last_clarification_timestamp": time.time()}
        )
    
    def _generate_multiple_choice_message(self, clarification_type: ClarificationType,
                                        language: str, variables: Dict[str, Any],
                                        context: ClarificationContext) -> str:
        """生成多选消息"""
        options = variables.get('options', [])
        if not options:
            return self._generate_fallback_message(language)
        
        # 限制选项数量
        max_options = 3 if context.clarification_count > 1 else 5
        limited_options = options[:max_options]
        
        # 格式化选项
        formatted_options = []
        for i, option in enumerate(limited_options, 1):
            name = option.get('item_name', '')
            price = option.get('price', 0.0)
            
            if language == 'zh':
                formatted_options.append(f"{i}. {name} (${price:.2f})")
            elif language == 'en':
                formatted_options.append(f"{i}. {name} - ${price:.2f}")
            else:  # Spanish
                formatted_options.append(f"{i}. {name} - ${price:.2f}")
        
        options_text = '\n'.join(formatted_options)
        
        # 生成介绍文本
        if language == 'zh':
            intro = "我找到了这些选项：" if context.clarification_count == 1 else "让我们缩小范围："
        elif language == 'en':
            intro = "I found these options:" if context.clarification_count == 1 else "Let's narrow it down:"
        else:
            intro = "Encontré estas opciones:" if context.clarification_count == 1 else "Vamos a precisar:"
        
        return f"{intro}\n{options_text}\n\n" + self._get_selection_prompt(language)
    
    def _generate_guided_narrowing_message(self, clarification_type: ClarificationType,
                                         language: str, variables: Dict[str, Any],
                                         context: ClarificationContext) -> str:
        """生成引导缩小范围的消息"""
        
        # 基于上下文选择缩小策略
        if context.clarification_count == 1:
            return self._ask_category_preference(language, variables)
        elif context.clarification_count == 2:
            return self._ask_price_preference(language, variables)
        else:
            return self._ask_specific_feature(language, variables)
    
    def _generate_progressive_refinement_message(self, clarification_type: ClarificationType,
                                               language: str, variables: Dict[str, Any],
                                               context: ClarificationContext) -> str:
        """生成渐进式细化消息"""
        
        failed_attempts = context.failed_attempts
        
        if language == 'zh':
            intro = f"我们试了{len(failed_attempts)}次了。让我换个方式："
        elif language == 'en':
            intro = f"We've tried {len(failed_attempts)} times. Let me try a different approach:"
        else:
            intro = f"Hemos intentado {len(failed_attempts)} veces. Déjame intentar de otra manera:"
        
        # 提供更简单的选择
        simple_question = self._generate_simple_binary_question(language, variables)
        
        return f"{intro}\n{simple_question}"
    
    def _prepare_template_variables(self, co: Dict[str, Any], path_data: Dict[str, Any],
                                  context: ClarificationContext) -> Dict[str, Any]:
        """准备模板变量"""
        variables = {
            'query': co.get('raw_text', ''),
            'language': co.get('language', 'es'),
            'clarification_count': context.clarification_count,
            'user_id': context.user_id
        }
        
        # 添加路径数据
        if path_data:
            variables.update({
                'confidence': path_data.get('confidence', 0.0),
                'options': self._extract_options_from_paths(path_data),
                'alternatives': path_data.get('alternative_paths', [])
            })
        
        # 添加上下文信息
        if context.user_preferences:
            variables.update({
                'user_preferences': context.user_preferences,
                'interaction_style': context.user_preferences.get('interaction_style', 'standard')
            })
        
        return variables
    
    def _extract_options_from_paths(self, path_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从路径数据中提取选项"""
        options = []
        
        # 主路径
        main_path = path_data.get('path', [])
        options.extend(main_path)
        
        # 备选路径
        alternative_paths = path_data.get('alternative_paths', [])
        for alt_path in alternative_paths:
            if isinstance(alt_path, dict) and 'matches' in alt_path:
                options.extend(alt_path['matches'])
        
        # 去重
        seen_ids = set()
        unique_options = []
        for option in options:
            item_id = option.get('item_id')
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_options.append(option)
        
        return unique_options
    
    def _get_expected_response_types(self, clarification_type: ClarificationType,
                                   strategy: ClarificationStrategy) -> List[str]:
        """获取预期响应类型"""
        type_mapping = {
            ClarificationType.MULTIPLE_MATCHES: ['number', 'item_name', 'selection'],
            ClarificationType.QUANTITY_AMBIGUOUS: ['number', 'quantity'],
            ClarificationType.LOW_CONFIDENCE: ['clarification', 'rephrasing'],
            ClarificationType.SIZE_SELECTION: ['size', 'option_selection'],
            ClarificationType.NO_MATCHES: ['alternative', 'new_request']
        }
        
        return type_mapping.get(clarification_type, ['text', 'clarification'])
    
    def _calculate_clarification_confidence(self, clarification_type: ClarificationType,
                                          strategy: ClarificationStrategy,
                                          context: ClarificationContext) -> float:
        """计算澄清置信度"""
        base_confidence = 0.8
        
        # 基于类型调整
        type_adjustments = {
            ClarificationType.MULTIPLE_MATCHES: 0.1,
            ClarificationType.LOW_CONFIDENCE: -0.2,
            ClarificationType.PROGRESSIVE_DISAMBIGUATION: -0.3
        }
        
        # 基于策略调整
        strategy_adjustments = {
            ClarificationStrategy.MULTIPLE_CHOICE: 0.1,
            ClarificationStrategy.PROGRESSIVE_REFINEMENT: -0.1
        }
        
        # 基于上下文调整
        context_penalty = min(0.3, context.clarification_count * 0.1)
        
        final_confidence = (base_confidence + 
                           type_adjustments.get(clarification_type, 0) +
                           strategy_adjustments.get(strategy, 0) - 
                           context_penalty)
        
        return max(0.1, min(1.0, final_confidence))
    
    def _get_timeout_for_type(self, clarification_type: ClarificationType) -> int:
        """获取超时时间"""
        timeouts = {
            ClarificationType.MULTIPLE_MATCHES: 45,
            ClarificationType.QUANTITY_AMBIGUOUS: 30,
            ClarificationType.PROGRESSIVE_DISAMBIGUATION: 60,
            ClarificationType.ORDER_CONFIRMATION: 30
        }
        
        return timeouts.get(clarification_type, 30)
    
    def _generate_fallback_message(self, language: str) -> str:
        """生成回退消息"""
        fallbacks = {
            'es': 'Disculpa, ¿podrías reformular tu solicitud?',
            'en': 'Sorry, could you rephrase your request?',
            'zh': '抱歉，您能重新表述一下您的要求吗？'
        }
        return fallbacks.get(language, fallbacks['es'])
    
    def _get_selection_prompt(self, language: str) -> str:
        """获取选择提示"""
        prompts = {
            'es': '¿Cuál número eliges?',
            'en': 'Which number do you choose?',
            'zh': '您选择哪个数字？'
        }
        return prompts.get(language, prompts['es'])
    
    def _ask_category_preference(self, language: str, variables: Dict[str, Any]) -> str:
        """询问类别偏好"""
        questions = {
            'es': '¿Prefieres algo con pollo, carne, o vegetariano?',
            'en': 'Do you prefer something with chicken, beef, or vegetarian?',
            'zh': '您比较喜欢鸡肉、牛肉还是素食？'
        }
        return questions.get(language, questions['es'])
    
    def _ask_price_preference(self, language: str, variables: Dict[str, Any]) -> str:
        """询问价格偏好"""
        questions = {
            'es': '¿Buscas algo económico o estás bien con opciones premium?',
            'en': 'Are you looking for something budget-friendly or are you okay with premium options?',
            'zh': '您想要经济实惠的还是高档一些的选项？'
        }
        return questions.get(language, questions['es'])
    
    def _ask_specific_feature(self, language: str, variables: Dict[str, Any]) -> str:
        """询问具体特征"""
        questions = {
            'es': '¿Hay algún ingrediente específico que te guste o quieras evitar?',
            'en': 'Is there any specific ingredient you like or want to avoid?',
            'zh': '有什么特定的配料您喜欢或者想要避免的吗？'
        }
        return questions.get(language, questions['es'])
    
    def _generate_simple_binary_question(self, language: str, variables: Dict[str, Any]) -> str:
        """生成简单的二元问题"""
        questions = {
            'es': '¿Quieres pollo o algo diferente?',
            'en': 'Do you want chicken or something different?',
            'zh': '您要鸡肉还是其他的？'
        }
        return questions.get(language, questions['es'])
    
    def _update_context(self, context: ClarificationContext, clarification_type: ClarificationType,
                       clarification: GeneratedClarification):
        """更新上下文"""
        context.last_clarification_type = clarification_type
        context.conversation_history.append({
            'timestamp': time.time(),
            'type': clarification_type.value,
            'strategy': clarification.strategy.value,
            'message': clarification.message,
            'confidence': clarification.confidence
        })
        
        # 限制历史记录长度
        if len(context.conversation_history) > 10:
            context.conversation_history = context.conversation_history[-10:]
        
        # 更新上下文内存
        context.context_memory.update(clarification.context_updates)

class ClarificationStrategySelector:
    """澄清策略选择器"""
    
    def __init__(self):
        self.strategy_rules = self._build_strategy_rules()
    
    def _build_strategy_rules(self) -> Dict[ClarificationType, List[Tuple[ClarificationStrategy, Dict[str, Any]]]]:
        """构建策略规则"""
        return {
            ClarificationType.MULTIPLE_MATCHES: [
                (ClarificationStrategy.MULTIPLE_CHOICE, {'max_clarifications': 2}),
                (ClarificationStrategy.GUIDED_NARROWING, {'min_clarifications': 2})
            ],
            ClarificationType.LOW_CONFIDENCE: [
                (ClarificationStrategy.CONTEXTUAL_SUGGESTION, {'confidence_threshold': 0.3}),
                (ClarificationStrategy.DIRECT_QUESTION, {})
            ],
            ClarificationType.PROGRESSIVE_DISAMBIGUATION: [
                (ClarificationStrategy.PROGRESSIVE_REFINEMENT, {}),
                (ClarificationStrategy.SMART_DEFAULT, {'fallback': True})
            ]
        }
    
    def select_strategy(self, clarification_type: ClarificationType, context: ClarificationContext,
                       co: Dict[str, Any], path_data: Dict[str, Any]) -> ClarificationStrategy:
        """选择澄清策略"""
        
        rules = self.strategy_rules.get(clarification_type, [])
        
        for strategy, conditions in rules:
            if self._evaluate_conditions(conditions, context, co, path_data):
                return strategy
        
        # 默认策略
        return ClarificationStrategy.DIRECT_QUESTION
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], context: ClarificationContext,
                           co: Dict[str, Any], path_data: Dict[str, Any]) -> bool:
        """评估策略条件"""
        
        # 最大澄清次数检查
        if 'max_clarifications' in conditions:
            if context.clarification_count >= conditions['max_clarifications']:
                return False
        
        # 最小澄清次数检查
        if 'min_clarifications' in conditions:
            if context.clarification_count < conditions['min_clarifications']:
                return False
        
        # 置信度阈值检查
        if 'confidence_threshold' in conditions:
            confidence = path_data.get('confidence', 0.0)
            if confidence >= conditions['confidence_threshold']:
                return False
        
        # 回退条件
        if conditions.get('fallback'):
            return True
        
        return True

class EnhancedMultilingualClarifyEngine:
    """增强的多语言澄清引擎"""
    
    def __init__(self):
        self.static_templates = self._load_static_templates()
        self.knowledge_base = self._load_knowledge_base()
        self.context_engine = ContextAwareClarificationEngine()
        self.llm_integration = LLMClarificationGenerator() if self._has_llm_support() else None
        
        # 性能监控
        self.performance_metrics = {
            'total_clarifications': 0,
            'successful_resolutions': 0,
            'avg_clarification_rounds': 0.0,
            'strategy_effectiveness': defaultdict(list)
        }
    
    def _has_llm_support(self) -> bool:
        """检查是否支持LLM集成"""
        # 这里可以检查是否配置了LLM API密钥
        return hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """加载菜单知识库（保持兼容）"""
        try:
            kb_path = Path(settings.MENU_KB_FILE)
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load knowledge base: {e}")
        return {}
    
    def _load_static_templates(self) -> Dict[str, List[ClarificationTemplate]]:
        """加载静态模板（保持向后兼容）"""
        templates = {
            'multiple_matches': [
                ClarificationTemplate(
                    id='multi_match_1',
                    type=ClarificationType.MULTIPLE_MATCHES,
                    strategy=ClarificationStrategy.MULTIPLE_CHOICE,
                    spanish='Encontré varias opciones para "{item}". ¿Cuál te gustaría?\n{options}',
                    english='I found several options for "{item}". Which one would you like?\n{options}',
                    chinese='我找到了几个"{item}"的选项，您要哪一个？\n{options}',
                    context='当有多个菜品匹配时',
                    priority=1,
                    variables=['item', 'options']
                ),
                ClarificationTemplate(
                    id='multi_match_2',
                    type=ClarificationType.MULTIPLE_MATCHES,
                    strategy=ClarificationStrategy.GUIDED_NARROWING,
                    spanish='Tenemos varias opciones de {item}. ¿Prefieres algo específico como {category_hint}?',
                    english='We have several {item} options. Do you prefer something specific like {category_hint}?',
                    chinese='我们有几种{item}选项。您比较喜欢{category_hint}这样的吗？',
                    context='引导式缩小范围',
                    priority=2,
                    variables=['item', 'category_hint']
                )
            ],
            'low_confidence': [
                ClarificationTemplate(
                    id='low_conf_1',
                    type=ClarificationType.LOW_CONFIDENCE,
                    strategy=ClarificationStrategy.CONTEXTUAL_SUGGESTION,
                    spanish='No estoy completamente seguro. ¿Te refieres a {suggestions}?',
                    english='I\'m not completely sure. Do you mean {suggestions}?',
                    chinese='我不太确定。您是指{suggestions}吗？',
                    context='提供上下文建议',
                    priority=1,
                    variables=['suggestions']
                ),
                ClarificationTemplate(
                    id='low_conf_2',
                    type=ClarificationType.LOW_CONFIDENCE,
                    strategy=ClarificationStrategy.DIRECT_QUESTION,
                    spanish='Disculpa, ¿podrías ser más específico sobre "{query}"?',
                    english='Sorry, could you be more specific about "{query}"?',
                    chinese='抱歉，您能更具体地说明一下"{query}"吗？',
                    context='直接询问澄清',
                    priority=2,
                    variables=['query']
                )
            ],
            'progressive_disambiguation': [
                ClarificationTemplate(
                    id='prog_disamb_1',
                    type=ClarificationType.PROGRESSIVE_DISAMBIGUATION,
                    strategy=ClarificationStrategy.PROGRESSIVE_REFINEMENT,
                    spanish='Intentemos paso a paso. ¿Buscas {category}?',
                    english='Let\'s try step by step. Are you looking for {category}?',
                    chinese='我们一步步来。您在找{category}吗？',
                    context='逐步细化',
                    priority=1,
                    variables=['category']
                )
            ]
        }
        
        return templates
    
    def generate_clarification(self, co: Dict[str, Any], path_data: Dict[str, Any] = None,
                             session_id: str = None, user_id: str = "anonymous",
                             use_llm: bool = True) -> Dict[str, Any]:
        """
        增强的澄清生成入口
        
        Args:
            co: 对话对象
            path_data: 路径数据
            session_id: 会话ID
            user_id: 用户ID
            use_llm: 是否使用LLM增强
            
        Returns:
            澄清结果字典
        """
        
        # 更新性能指标
        self.performance_metrics['total_clarifications'] += 1
        
        # 如果没有会话ID，生成一个
        if session_id is None:
            session_id = f"session_{int(time.time())}_{hash(str(co))}"
        
        try:
            # 使用上下文感知引擎生成澄清
            clarification = self.context_engine.process_clarification_request(
                co, path_data or {}, session_id, user_id
            )
            
            # LLM增强（如果可用且请求）
            if use_llm and self.llm_integration and clarification.confidence < 0.7:
                enhanced_clarification = self.llm_integration.enhance_clarification(
                    clarification, co, path_data
                )
                if enhanced_clarification:
                    clarification = enhanced_clarification
            
            # 记录策略效果
            self.performance_metrics['strategy_effectiveness'][clarification.strategy.value].append(
                clarification.confidence
            )
            
            # 转换为返回格式
            result = {
                'message': clarification.message,
                'type': clarification.type.value,
                'strategy': clarification.strategy.value,
                'confidence': clarification.confidence,
                'expected_response_types': clarification.expected_response_types,
                'timeout_seconds': clarification.timeout_seconds,
                'fallback_message': clarification.fallback_message,
                'session_id': session_id,
                'context_updates': clarification.context_updates
            }
            
            return result
            
        except Exception as e:
            print(f"Error in clarification generation: {e}")
            
            # 回退到简单澄清
            language = co.get('language', 'es')
            fallback_message = self._get_simple_fallback(language)
            
            return {
                'message': fallback_message,
                'type': 'fallback',
                'strategy': 'simple',
                'confidence': 0.3,
                'expected_response_types': ['text'],
                'timeout_seconds': 30,
                'session_id': session_id
            }
    
    def _get_simple_fallback(self, language: str) -> str:
        """获取简单回退消息"""
        fallbacks = {
            'es': '¿Podrías decirme qué necesitas?',
            'en': 'Could you tell me what you need?',
            'zh': '您能告诉我您需要什么吗？'
        }
        return fallbacks.get(language, fallbacks['es'])
    
    def process_clarification_response(self, response: str, session_id: str,
                                     original_co: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理澄清响应
        
        Args:
            response: 用户的澄清响应
            session_id: 会话ID
            original_co: 原始对话对象
            
        Returns:
            处理结果
        """
        
        context = self.context_engine.context_memory.get(session_id)
        if not context:
            return {'error': 'Session not found'}
        
        # 解析响应
        parsed_response = self._parse_clarification_response(response, context)
        
        # 更新上下文
        context.conversation_history.append({
            'timestamp': time.time(),
            'user_response': response,
            'parsed_response': parsed_response,
            'type': 'user_clarification_response'
        })
        
        # 判断是否需要进一步澄清
        if parsed_response.get('needs_further_clarification'):
            # 生成后续澄清
            return self.generate_clarification(
                original_co, 
                parsed_response.get('updated_path_data'),
                session_id
            )
        else:
            # 澄清完成，更新成功指标
            self.performance_metrics['successful_resolutions'] += 1
            avg_rounds = (self.performance_metrics['avg_clarification_rounds'] * 
                         (self.performance_metrics['successful_resolutions'] - 1) + 
                         context.clarification_count) / self.performance_metrics['successful_resolutions']
            self.performance_metrics['avg_clarification_rounds'] = avg_rounds
            
            return {
                'clarification_complete': True,
                'final_selection': parsed_response.get('selected_item'),
                'updated_co': parsed_response.get('updated_co'),
                'session_summary': self._generate_session_summary(context)
            }
    
    def _parse_clarification_response(self, response: str, context: ClarificationContext) -> Dict[str, Any]:
        """解析澄清响应"""
        
        response_lower = response.lower().strip()
        
        # 数字选择检测
        number_match = re.search(r'\b(\d+)\b', response)
        if number_match:
            selected_number = int(number_match.group(1))
            return {
                'type': 'number_selection',
                'selected_number': selected_number,
                'needs_further_clarification': False
            }
        
        # 是/否检测
        yes_patterns = ['sí', 'si', 'yes', '是', '对', 'correcto', 'exacto']
        no_patterns = ['no', '不', '不是', 'incorrect', 'wrong']
        
        if any(pattern in response_lower for pattern in yes_patterns):
            return {
                'type': 'confirmation',
                'confirmed': True,
                'needs_further_clarification': False
            }
        elif any(pattern in response_lower for pattern in no_patterns):
            return {
                'type': 'confirmation',
                'confirmed': False,
                'needs_further_clarification': True
            }
        
        # 物品名称检测
        # 这里可以添加更复杂的NLP解析
        
        return {
            'type': 'text_response',
            'content': response,
            'needs_further_clarification': True
        }
    
    def _generate_session_summary(self, context: ClarificationContext) -> Dict[str, Any]:
        """生成会话摘要"""
        return {
            'total_clarifications': context.clarification_count,
            'conversation_length': len(context.conversation_history),
            'final_resolution_type': context.last_clarification_type.value if context.last_clarification_type else None,
            'user_id': context.user_id,
            'session_duration': time.time() - context.conversation_history[0].get('timestamp', time.time()) if context.conversation_history else 0
        }
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """获取性能分析"""
        analytics = self.performance_metrics.copy()
        
        # 计算策略效果统计
        strategy_stats = {}
        for strategy, confidences in analytics['strategy_effectiveness'].items():
            if confidences:
                strategy_stats[strategy] = {
                    'avg_confidence': sum(confidences) / len(confidences),
                    'usage_count': len(confidences),
                    'success_rate': sum(1 for c in confidences if c > 0.7) / len(confidences)
                }
        
        analytics['strategy_stats'] = strategy_stats
        analytics['overall_success_rate'] = (analytics['successful_resolutions'] / 
                                           max(analytics['total_clarifications'], 1))
        
        return analytics
    
    def reset_session(self, session_id: str) -> bool:
        """重置会话"""
        if session_id in self.context_engine.context_memory:
            del self.context_engine.context_memory[session_id]
            return True
        return False
    
    def export_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """导出会话数据"""
        context = self.context_engine.context_memory.get(session_id)
        if context:
            return {
                'session_id': session_id,
                'user_id': context.user_id,
                'clarification_count': context.clarification_count,
                'conversation_history': context.conversation_history,
                'user_preferences': context.user_preferences,
                'failed_attempts': context.failed_attempts,
                'export_timestamp': time.time()
            }
        return None

class LLMClarificationGenerator:
    """LLM澄清生成器"""
    
    def __init__(self):
        self.api_available = self._check_api_availability()
        
    def _check_api_availability(self) -> bool:
        """检查API可用性"""
        return (hasattr(settings, 'ANTHROPIC_API_KEY') and 
                settings.ANTHROPIC_API_KEY and 
                len(settings.ANTHROPIC_API_KEY) > 10)
    
    def enhance_clarification(self, clarification: GeneratedClarification,
                            co: Dict[str, Any], path_data: Dict[str, Any]) -> Optional[GeneratedClarification]:
        """使用LLM增强澄清"""
        
        if not self.api_available:
            return None
        
        try:
            # 构建LLM提示
            prompt = self._build_enhancement_prompt(clarification, co, path_data)
            
            # 调用LLM（这里需要实际的API调用）
            enhanced_message = self._call_llm_api(prompt)
            
            if enhanced_message:
                # 创建增强的澄清对象
                enhanced_clarification = GeneratedClarification(
                    message=enhanced_message,
                    type=clarification.type,
                    strategy=ClarificationStrategy.CONVERSATIONAL_FLOW,
                    confidence=min(1.0, clarification.confidence + 0.2),
                    expected_response_types=clarification.expected_response_types,
                    timeout_seconds=clarification.timeout_seconds,
                    fallback_message=clarification.message,  # 原消息作为回退
                    context_updates=clarification.context_updates
                )
                
                return enhanced_clarification
                
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
        
        return None
    
    def _build_enhancement_prompt(self, clarification: GeneratedClarification,
                                co: Dict[str, Any], path_data: Dict[str, Any]) -> str:
        """构建LLM增强提示"""
        
        language = co.get('language', 'es')
        language_names = {'es': 'Spanish', 'en': 'English', 'zh': 'Chinese'}
        
        prompt = f"""
You are a helpful restaurant ordering assistant. Please improve this clarification message to be more natural and conversational.

Current situation:
- User's original request: "{co.get('raw_text', '')}"
- Language: {language_names.get(language, 'Spanish')}
- Clarification type: {clarification.type.value}
- Current confidence: {clarification.confidence}

Current clarification message:
"{clarification.message}"

Please provide an improved version that is:
1. More natural and conversational
2. Maintains the same information
3. Is culturally appropriate for {language_names.get(language, 'Spanish')} speakers
4. Keeps the same structure if it's a multiple choice question

Improved message:
"""
        
        return prompt
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """调用LLM API"""
        # 这里应该实际调用Anthropic Claude API
        # 由于这是示例代码，返回None表示未实现
        
        # 示例实现结构：
        # try:
        #     import anthropic
        #     client = anthropic.Client(api_key=settings.ANTHROPIC_API_KEY)
        #     response = client.messages.create(
        #         model="claude-3-sonnet-20240229",
        #         max_tokens=200,
        #         messages=[{"role": "user", "content": prompt}]
        #     )
        #     return response.content[0].text
        # except Exception as e:
        #     print(f"LLM API call failed: {e}")
        #     return None
        
        return None

# 全局引擎实例
_global_engine = EnhancedMultilingualClarifyEngine()

# 保持向后兼容的接口
def build(co: Dict[str, Any], path_data: Dict[str, Any] = None) -> str:
    """
    外部调用接口 - 构建澄清消息（保持向后兼容）
    
    Args:
        co: 对话对象
        path_data: 路径数据
        
    Returns:
        澄清消息字符串
    """
    result = _global_engine.generate_clarification(co, path_data)
    return result['message']

def build_order_confirmation(co: Dict[str, Any], order_items: List[Dict[str, Any]], total_price: float) -> str:
    """构建订单确认消息（保持向后兼容）"""
    language = co.get('language', 'es')
    
    # 格式化订单摘要
    order_lines = []
    for item in order_items:
        quantity = item.get('quantity', 1)
        name = item.get('item_name', '')
        price = item.get('price', 0.0)
        total_item_price = price * quantity
        
        if language == 'zh':
            line = f"• {quantity}份 {name} - ${total_item_price:.2f}"
        elif language == 'en':
            line = f"• {quantity}x {name} - ${total_item_price:.2f}"
        else:  # Spanish
            line = f"• {quantity}x {name} - ${total_item_price:.2f}"
        
        order_lines.append(line)
    
    order_summary = '\n'.join(order_lines)
    
    # 确认模板
    templates = {
        'es': f'Perfecto, confirmo tu orden:\n{order_summary}\nTotal: ${total_price:.2f}\n¿Es correcto?',
        'en': f'Perfect, I confirm your order:\n{order_summary}\nTotal: ${total_price:.2f}\nIs this correct?',
        'zh': f'好的，确认您的订单：\n{order_summary}\n总计：${total_price:.2f}\n正确吗？'
    }
    
    return templates.get(language, templates['es'])

def build_quantity_clarification(co: Dict[str, Any], item_name: str, quantity: int = 1) -> str:
    """构建数量澄清消息（保持向后兼容）"""
    language = co.get('language', 'es')
    
    templates = {
        'es': f'¿Quieres {quantity} de {item_name}?',
        'en': f'Do you want {quantity} {item_name}?',
        'zh': f'您要{quantity}份{item_name}吗？'
    }
    
    return templates.get(language, templates['es'])

def build_modification_clarification(co: Dict[str, Any], item_name: str, modification: str) -> str:
    """构建修改澄清消息（保持向后兼容）"""
    language = co.get('language', 'es')
    
    templates = {
        'es': f'¿Quieres {item_name} {modification}?',
        'en': f'Do you want {item_name} {modification}?',
        'zh': f'您要{item_name}{modification}吗？'
    }
    
    return templates.get(language, templates['es'])

# 新增的增强接口
def generate_enhanced_clarification(co: Dict[str, Any], path_data: Dict[str, Any] = None,
                                   session_id: str = None, user_id: str = "anonymous",
                                   use_llm: bool = True) -> Dict[str, Any]:
    """生成增强澄清（新接口）"""
    return _global_engine.generate_clarification(co, path_data, session_id, user_id, use_llm)

def process_clarification_response(response: str, session_id: str, original_co: Dict[str, Any]) -> Dict[str, Any]:
    """处理澄清响应（新接口）"""
    return _global_engine.process_clarification_response(response, session_id, original_co)

def get_clarification_analytics() -> Dict[str, Any]:
    """获取澄清分析数据（新接口）"""
    return _global_engine.get_performance_analytics()

def reset_clarification_session(session_id: str) -> bool:
    """重置澄清会话（新接口）"""
    return _global_engine.reset_session(session_id)

def export_clarification_session(session_id: str) -> Optional[Dict[str, Any]]:
    """导出澄清会话数据（新接口）"""
    return _global_engine.export_session_data(session_id)

# 综合测试函数
def run_comprehensive_tests():
    """运行综合测试"""
    print("=== 增强版 Clarify Engine 综合测试 ===\n")
    
    # 1. 向后兼容性测试
    print("1. 向后兼容性测试:")
    
    test_co = {
        'language': 'es',
        'raw_text': 'pollo',
        'intent': 'order'
    }
    
    test_path_data = {
        'requires_clarification': True,
        'clarification_reason': 'multiple_matches',
        'alternative_paths': [
            {
                'matches': [
                    {'item_id': '1', 'item_name': 'Pollo Teriyaki', 'price': 11.99},
                    {'item_id': '2', 'item_name': 'Pollo Naranja', 'price': 11.89}
                ]
            }
        ]
    }
    
    # 原有接口测试
    old_clarification = build(test_co, test_path_data)
    print(f"   原有接口澄清: {old_clarification[:100]}...")
    
    # 2. 增强接口测试
    print("\n2. 增强接口测试:")
    
    enhanced_result = generate_enhanced_clarification(
        test_co, test_path_data, "test_session_001", "test_user_001"
    )
    
    print(f"   增强澄清类型: {enhanced_result['type']}")
    print(f"   使用策略: {enhanced_result['strategy']}")
    print(f"   置信度: {enhanced_result['confidence']}")
    print(f"   消息: {enhanced_result['message'][:100]}...")
    
    # 3. 多轮对话测试
    print("\n3. 多轮对话测试:")
    
    session_id = "multi_round_session"
    
    # 第一轮澄清
    round1 = generate_enhanced_clarification(test_co, test_path_data, session_id)
    print(f"   第1轮: {round1['message'][:80]}...")
    
    # 模拟用户响应
    response1 = process_clarification_response("不确定", session_id, test_co)
    if not response1.get('clarification_complete'):
        print(f"   第2轮: {response1.get('message', '无后续澄清')[:80]}...")
    
    # 4. 不同语言测试
    print("\n4. 多语言测试:")
    
    languages = ['es', 'en', 'zh']
    for lang in languages:
        test_co_lang = test_co.copy()
        test_co_lang['language'] = lang
        
        clarification = generate_enhanced_clarification(test_co_lang, test_path_data)
        print(f"   {lang}: {clarification['message'][:60]}...")
    
    # 5. 特殊场景测试
    print("\n5. 特殊场景测试:")
    
    # 低置信度场景
    low_conf_path = {
        'requires_clarification': True,
        'clarification_reason': 'low_confidence',
        'confidence': 0.2
    }
    
    low_conf_result = generate_enhanced_clarification(test_co, low_conf_path)
    print(f"   低置信度: {low_conf_result['message'][:60]}...")
    
    # 无匹配场景
    no_match_path = {
        'requires_clarification': True,
        'clarification_reason': 'no_menu_matches',
        'path': []
    }
    
    no_match_result = generate_enhanced_clarification(test_co, no_match_path)
    print(f"   无匹配: {no_match_result['message'][:60]}...")
    
    # 6. 性能分析
    print("\n6. 性能分析:")
    
    analytics = get_clarification_analytics()
    print(f"   总澄清次数: {analytics['total_clarifications']}")
    print(f"   成功解决率: {analytics['overall_success_rate']:.2%}")
    print(f"   平均澄清轮数: {analytics['avg_clarification_rounds']:.1f}")
    
    # 7. 订单确认测试
    print("\n7. 订单确认测试:")
    
    order_items = [
        {'item_name': 'Pollo Teriyaki', 'quantity': 2, 'price': 11.99},
        {'item_name': 'Arroz Frito', 'quantity': 1, 'price': 5.99}
    ]
    
    confirmation = build_order_confirmation(test_co, order_items, 29.97)
    print(f"   确认消息: {confirmation[:100]}...")
    
    print("\n=== 测试完成 ===")

# 主程序兼容性测试
if __name__ == "__main__":
    # 保持原有测试的向后兼容性
    print("=== 向后兼容性验证 ===")
    
    test_co = {
        'language': 'es',
        'raw_text': 'pollo',
        'intent': 'order'
    }
    
    test_path_data = {
        'requires_clarification': True,
        'clarification_reason': 'multiple_matches',
        'alternative_paths': [
            {
                'matches': [
                    {'item_id': '1', 'item_name': 'Pollo Teriyaki', 'price': 11.99},
                    {'item_id': '2', 'item_name': 'Pollo Naranja', 'price': 11.89}
                ]
            }
        ]
    }
    
    clarification = build(test_co, test_path_data)
    print("澄清消息:")
    print(clarification)
    
    print("\n=== 增强功能演示 ===")
    run_comprehensive_tests()
