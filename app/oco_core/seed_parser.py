#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Seed Parser (优化版)
多语言自然语言订单解析器 - 增强性能、上下文理解、缓存机制
"""

import re
import json
import hashlib
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import time

from ..config import settings

@dataclass
class ParsedObject:
    """解析出的订单对象"""
    item_type: str          # 'main_dish', 'modification', 'quantity'
    content: str            # 原始文本
    quantity: int = 1       # 数量
    modifiers: List[str] = field(default_factory=list)  # 修饰词
    confidence: float = 0.0 # 置信度
    start_pos: int = -1     # 在原文中的起始位置
    end_pos: int = -1       # 在原文中的结束位置
    normalized_content: str = ""  # 标准化后的内容

@dataclass
class ParseContext:
    """解析上下文"""
    previous_orders: List[Dict] = field(default_factory=list)
    session_language: Optional[str] = None
    preferred_items: Set[str] = field(default_factory=set)
    conversation_flow: str = "initial"  # initial, ordering, modifying, confirming

@dataclass
class ConversationObject:
    """对话对象 - CO结构（增强版）"""
    objects: List[ParsedObject]
    intent: str                    # 'order', 'modify', 'confirm', 'cancel', 'inquiry'
    conditions: List[str]          # 约束条件
    language: str = 'es'          # 检测到的主要语言
    confidence: float = 0.0       # 整体解析置信度
    raw_text: str = ''            # 原始输入
    context_used: bool = False    # 是否使用了上下文
    parse_time: float = 0.0       # 解析耗时（秒）
    alternatives: List[Dict] = field(default_factory=list)  # 可选解析结果

class AdvancedLanguageDetector:
    """高级语言检测器"""
    
    def __init__(self):
        self.language_patterns = {
            'zh': {
                'chars': r'[\u4e00-\u9fff]',
                'keywords': ['要', '点', '个', '份', '块', '件', '换', '改', '不要', '加', '来'],
                'grammar': [r'\d+\s*[个份块件]', r'[要点来]\s*\w+']
            },
            'es': {
                'chars': r'[ñáéíóúü]',
                'keywords': ['pollo', 'carne', 'arroz', 'papa', 'quiero', 'con', 'sin', 'más', 'poco', 
                           'combo', 'presa', 'cambio', 'dame', 'necesito'],
                'grammar': [r'\b(?:quiero|dame|necesito)\b', r'\b\d+\s*(?:presas?|combos?)\b']
            },
            'en': {
                'chars': r'[a-zA-Z]',
                'keywords': ['chicken', 'beef', 'rice', 'potato', 'want', 'with', 'without', 'more', 
                           'less', 'order', 'get', 'combo', 'piece'],
                'grammar': [r'\b(?:want|order|get)\b', r'\b\d+\s*(?:pieces?|combos?)\b']
            }
        }
    
    @lru_cache(maxsize=1000)
    def detect(self, text: str, context_lang: Optional[str] = None) -> Tuple[str, float]:
        """
        检测语言及置信度
        
        Args:
            text: 输入文本
            context_lang: 上下文语言
            
        Returns:
            (语言代码, 置信度)
        """
        text_lower = text.lower()
        scores = defaultdict(float)
        
        for lang, patterns in self.language_patterns.items():
            # 字符特征得分
            char_matches = len(re.findall(patterns['chars'], text))
            scores[lang] += char_matches * 2
            
            # 关键词得分
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    scores[lang] += 3
            
            # 语法模式得分
            for pattern in patterns['grammar']:
                matches = re.findall(pattern, text_lower)
                scores[lang] += len(matches) * 4
            
            # 上下文语言加成
            if context_lang == lang:
                scores[lang] += 5
        
        if not scores:
            return ('es', 0.5)  # 默认西班牙语
        
        max_lang = max(scores, key=scores.get)
        max_score = scores[max_lang]
        total_score = sum(scores.values())
        
        confidence = min(1.0, max_score / max(total_score, 1))
        return (max_lang, confidence)

class OptimizedPatternMatcher:
    """优化的模式匹配器"""
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
        self.synonym_groups = self._build_synonym_groups()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """编译正则表达式模式"""
        pattern_strings = {
            # 订单意图模式（优化后）
            "order_intent": [
                r"(?:我?要|点|来|给我|quiero|dame|want|order|get|necesito)\s*(.+?)(?:\s|$)",
                r"(.+?)\s*(?:个|份|块|件|piezas?|presas?|combos?)",
                r"(\d+)\s*(.+)"
            ],
            
            # 数量提取模式（改进）
            "quantity": [
                r"(\d+)\s*(?:个|份|块|件|piezas?|presas?|combos?|pieces?)\s*(?:de\s+)?(.+)",
                r"(?:uno?|dos|tres|cuatro|cinco|一|二|三|四|五|1|2|3|4|5)\s*(.+)",
                r"^(\d+)\s+(.+?)(?:\s|$)"
            ],
            
            # 修改意图模式（精确化）
            "modification": [
                r"(?:换|改|cambio|change)\s*(.+?)(?:\s|$)",
                r"(?:不要|no|sin|without)\s*(.+?)(?:\s|$)",
                r"(?:加|extra|más|more)\s*(.+?)(?:\s|$)",
                r"(?:少|poco|less)\s*(.+?)(?:\s|$)"
            ],
            
            # 确认模式
            "confirmation": [
                r"^(?:对|是|好|确认|sí|yes|correcto|ok|确定)$",
                r"^(?:不对|不是|错|no|incorrecto|wrong|不对)$"
            ],
            
            # 取消模式
            "cancel": [
                r"(?:取消|cancel|cancelar|不要了|算了)"
            ]
        }
        
        compiled = {}
        for category, patterns in pattern_strings.items():
            compiled[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
        
        return compiled
    
    def _build_synonym_groups(self) -> Dict[str, Set[str]]:
        """构建同义词组"""
        return {
            'chicken': {'pollo', 'chicken', '鸡', '鸡肉'},
            'beef': {'carne', 'beef', 'meat', '牛肉', '肉'},
            'rice': {'arroz', 'rice', '米饭', '饭'},
            'potato': {'papa', 'papas', 'potato', 'potatoes', '土豆', '马铃薯'},
            'combo': {'combo', 'combinacion', 'combination', '套餐'},
            'piece': {'presa', 'piece', '块', '件'}
        }
    
    def find_matches(self, pattern_type: str, text: str) -> List[re.Match]:
        """查找匹配项"""
        matches = []
        for pattern in self.compiled_patterns.get(pattern_type, []):
            matches.extend(pattern.finditer(text))
        return matches

class SmartQuantityParser:
    """智能数量解析器"""
    
    def __init__(self):
        self.number_mappings = self._build_number_mappings()
        self.fraction_patterns = re.compile(r'(\d+)?\s*(?:/|分之)\s*(\d+)')
        self.range_patterns = re.compile(r'(\d+)\s*[-到至]\s*(\d+)')
    
    def _build_number_mappings(self) -> Dict[str, int]:
        """构建数字映射表"""
        mappings = {}
        
        # 中文数字
        chinese_basic = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, 
                        '七': 7, '八': 8, '九': 9, '十': 10, '两': 2, '双': 2}
        mappings.update(chinese_basic)
        
        # 中文复合数字
        for i in range(11, 21):
            mappings[f'十{list(chinese_basic.keys())[i-11]}'] = i
        
        # 西班牙语数字
        spanish_nums = {
            'uno': 1, 'una': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
            'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
            'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15,
            'veinte': 20, 'veintiuno': 21, 'treinta': 30
        }
        mappings.update(spanish_nums)
        
        # 英文数字
        english_nums = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20
        }
        mappings.update(english_nums)
        
        return mappings
    
    @lru_cache(maxsize=500)
    def parse_quantity(self, text: str) -> Tuple[int, float]:
        """
        解析数量表达
        
        Returns:
            (数量, 置信度)
        """
        text = text.strip().lower()
        
        # 直接数字
        if text.isdigit():
            qty = int(text)
            return (qty if 1 <= qty <= 50 else 1, 1.0)
        
        # 分数处理
        fraction_match = self.fraction_patterns.search(text)
        if fraction_match:
            numerator = int(fraction_match.group(1) or 1)
            denominator = int(fraction_match.group(2))
            return (max(1, numerator // denominator), 0.8)
        
        # 范围处理（取中间值）
        range_match = self.range_patterns.search(text)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            return ((start + end) // 2, 0.7)
        
        # 词语映射
        for word, number in self.number_mappings.items():
            if word in text:
                return (number, 0.9)
        
        # 提取数字
        numbers = re.findall(r'\d+', text)
        if numbers:
            qty = int(numbers[0])
            return (qty if 1 <= qty <= 50 else 1, 0.8)
        
        return (1, 0.5)

class ContextAwareParser:
    """上下文感知解析器"""
    
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_key(self, text: str, context: Optional[ParseContext] = None) -> str:
        """生成缓存键"""
        context_str = ""
        if context:
            context_str = f"{context.session_language}_{context.conversation_flow}"
        
        return hashlib.md5(f"{text}_{context_str}".encode()).hexdigest()
    
    def enhance_parsing_with_context(self, objects: List[ParsedObject], 
                                   context: Optional[ParseContext] = None) -> List[ParsedObject]:
        """使用上下文增强解析结果"""
        if not context:
            return objects
        
        enhanced_objects = []
        
        for obj in objects:
            enhanced_obj = ParsedObject(
                item_type=obj.item_type,
                content=obj.content,
                quantity=obj.quantity,
                modifiers=obj.modifiers.copy(),
                confidence=obj.confidence,
                start_pos=obj.start_pos,
                end_pos=obj.end_pos,
                normalized_content=obj.normalized_content
            )
            
            # 上下文增强置信度
            if obj.content.lower() in context.preferred_items:
                enhanced_obj.confidence = min(1.0, enhanced_obj.confidence + 0.1)
            
            # 根据对话流程调整
            if context.conversation_flow == "modifying" and obj.item_type == "main_dish":
                enhanced_obj.item_type = "modification"
                enhanced_obj.confidence = min(1.0, enhanced_obj.confidence + 0.05)
            
            enhanced_objects.append(enhanced_obj)
        
        return enhanced_objects

class MultiLanguageSeedParser:
    """多语言种子解析器（优化版）"""
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.lang_detector = AdvancedLanguageDetector()
        self.pattern_matcher = OptimizedPatternMatcher()
        self.quantity_parser = SmartQuantityParser()
        self.context_parser = ContextAwareParser()
        self.performance_stats = {
            'total_parses': 0,
            'avg_parse_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """加载菜单知识库（缓存优化）"""
        try:
            kb_path = Path(settings.MENU_KB_FILE)
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load knowledge base: {e}")
        
        return {
            "ai_parsing_rules": {
                "intent_detection": {
                    "order_intent_keywords": ["要", "点", "来", "给我", "我要", "订", "买", "quiero", "dame", "necesito", "want", "order", "get"],
                    "modification_keywords": ["换", "改", "不要", "少", "多", "加", "cambio", "extra", "poco", "no", "sin", "without", "less", "more", "change"],
                    "quantity_keywords": ["个", "份", "块", "件", "piezas", "presas", "combos", "pieces", "orders", "servings"]
                },
                "menu_items": {
                    "main_dishes": ["pollo teriyaki", "carne con broccoli", "arroz con pollo", "combo familiar"],
                    "sides": ["arroz", "papa", "tostones", "yuca", "ensalada"],
                    "sauces": ["salsa", "ajo", "picante", "dulce"]
                },
                "synonym_mapping": {
                    "food_items": {
                        "pollo": ["chicken", "鸡", "鸡肉"],
                        "carne": ["beef", "meat", "牛肉", "肉"],
                        "arroz": ["rice", "米饭", "饭"],
                        "papa": ["papas", "potato", "potatoes", "土豆", "马铃薯"]
                    }
                }
            }
        }
    
    def _detect_intent_advanced(self, text: str, context: Optional[ParseContext] = None) -> Tuple[str, float]:
        """高级意图检测"""
        text_lower = text.lower().strip()
        
        # 使用上下文信息
        if context and context.conversation_flow == "confirming":
            if re.search(r'(?:对|是|好|确认|sí|yes|correcto|ok|确定)', text_lower):
                return ("confirm", 0.95)
            if re.search(r'(?:不对|不是|错|no|incorrecto|wrong)', text_lower):
                return ("modify", 0.95)
        
        # 意图权重计算
        intent_scores = defaultdict(float)
        
        # 取消意图
        cancel_matches = self.pattern_matcher.find_matches("cancel", text)
        intent_scores["cancel"] = len(cancel_matches) * 3
        
        # 确认意图
        confirm_matches = self.pattern_matcher.find_matches("confirmation", text)
        intent_scores["confirm"] = len(confirm_matches) * 2.5
        
        # 修改意图
        modify_matches = self.pattern_matcher.find_matches("modification", text)
        intent_scores["modify"] = len(modify_matches) * 2
        
        # 订单意图
        order_matches = self.pattern_matcher.find_matches("order_intent", text)
        intent_scores["order"] = len(order_matches) * 1.5
        
        # 数量暗示订单意图
        quantity_matches = self.pattern_matcher.find_matches("quantity", text)
        intent_scores["order"] += len(quantity_matches) * 1
        
        if not intent_scores:
            return ("inquiry", 0.3)
        
        max_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[max_intent]
        total_score = sum(intent_scores.values())
        
        confidence = min(1.0, max_score / max(total_score, 1))
        return (max_intent, confidence)
    
    def _extract_food_items_advanced(self, text: str, language: str) -> List[ParsedObject]:
        """高级食物项目提取"""
        objects = []
        processed_positions = set()
        
        # 菜单项目匹配
        menu_items = self.knowledge_base.get("ai_parsing_rules", {}).get("menu_items", {})
        all_menu_items = []
        for category in menu_items.values():
            if isinstance(category, list):
                all_menu_items.extend(category)
        
        # 精确菜单匹配
        text_lower = text.lower()
        for item in all_menu_items:
            if item.lower() in text_lower:
                start_pos = text_lower.find(item.lower())
                end_pos = start_pos + len(item)
                
                if not any(pos in processed_positions for pos in range(start_pos, end_pos)):
                    obj = ParsedObject(
                        item_type="main_dish",
                        content=item,
                        quantity=1,
                        confidence=0.9,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        normalized_content=item.lower()
                    )
                    objects.append(obj)
                    processed_positions.update(range(start_pos, end_pos))
        
        # 数量+项目匹配
        quantity_matches = self.pattern_matcher.find_matches("quantity", text)
        for match in quantity_matches:
            if len(match.groups()) >= 2:
                qty_str, item_str = match.groups()[:2]
                quantity, qty_confidence = self.quantity_parser.parse_quantity(qty_str)
                
                start_pos, end_pos = match.span()
                if not any(pos in processed_positions for pos in range(start_pos, end_pos)):
                    obj = ParsedObject(
                        item_type="main_dish",
                        content=item_str.strip(),
                        quantity=quantity,
                        confidence=min(0.8, qty_confidence),
                        start_pos=start_pos,
                        end_pos=end_pos,
                        normalized_content=item_str.strip().lower()
                    )
                    objects.append(obj)
                    processed_positions.update(range(start_pos, end_pos))
        
        # 订单意图匹配
        order_matches = self.pattern_matcher.find_matches("order_intent", text)
        for match in order_matches:
            if match.groups():
                item_text = match.group(1).strip()
                start_pos, end_pos = match.span(1)
                
                if (item_text and len(item_text) > 1 and 
                    not any(pos in processed_positions for pos in range(start_pos, end_pos))):
                    obj = ParsedObject(
                        item_type="main_dish",
                        content=item_text,
                        quantity=1,
                        confidence=0.6,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        normalized_content=item_text.lower()
                    )
                    objects.append(obj)
                    processed_positions.update(range(start_pos, end_pos))
        
        return objects
    
    def _extract_modifications_advanced(self, text: str) -> List[ParsedObject]:
        """高级修改要求提取"""
        modifications = []
        
        modification_types = {
            r"(?:不要|no|sin|without)": "remove",
            r"(?:加|extra|más|more)": "add",
            r"(?:换|cambio|change)": "change",
            r"(?:少|poco|less)": "reduce"
        }
        
        for type_pattern, mod_type in modification_types.items():
            matches = re.finditer(f"{type_pattern}\\s*(.+?)(?:\\s|$)", text, re.IGNORECASE)
            for match in matches:
                mod_text = match.group(1).strip()
                if mod_text and len(mod_text) > 1:
                    start_pos, end_pos = match.span()
                    obj = ParsedObject(
                        item_type=mod_type,
                        content=mod_text,
                        confidence=0.75,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        normalized_content=mod_text.lower()
                    )
                    modifications.append(obj)
        
        return modifications
    
    def _extract_conditions_advanced(self, text: str) -> List[str]:
        """高级约束条件提取"""
        conditions = []
        condition_patterns = {
            "separate_serving": r"(?:分开|aparte|separate)",
            "takeaway": r"(?:打包|para llevar|takeaway|to go)",
            "dine_in": r"(?:堂食|para comer aquí|dine in|eat here)",
            "urgent": r"(?:急|快|急单|rápido|urgente|quickly|asap)",
            "hot": r"(?:热|caliente|hot)",
            "cold": r"(?:冷|frío|cold|iced)"
        }
        
        text_lower = text.lower()
        for condition, pattern in condition_patterns.items():
            if re.search(pattern, text_lower):
                conditions.append(condition)
        
        return conditions
    
    def _calculate_confidence_advanced(self, objects: List[ParsedObject], intent: str, 
                                     text: str, language: str, lang_confidence: float) -> float:
        """高级置信度计算"""
        if not objects:
            return max(0.1, lang_confidence * 0.3)
        
        # 对象置信度
        obj_confidence = sum(obj.confidence for obj in objects) / len(objects)
        
        # 语言检测置信度影响
        lang_factor = lang_confidence * 0.2
        
        # 意图明确性
        intent_weights = {
            "order": 0.3, "modify": 0.25, "confirm": 0.2,
            "cancel": 0.15, "inquiry": 0.1
        }
        intent_factor = intent_weights.get(intent, 0.1)
        
        # 文本长度合理性
        text_length = len(text.strip())
        length_factor = min(0.15, text_length / 50 * 0.15)
        
        # 菜单项匹配加成
        menu_match_bonus = 0.0
        menu_items = self.knowledge_base.get("ai_parsing_rules", {}).get("menu_items", {})
        all_items = []
        for category in menu_items.values():
            if isinstance(category, list):
                all_items.extend([item.lower() for item in category])
        
        text_lower = text.lower()
        matched_items = sum(1 for item in all_items if item in text_lower)
        menu_match_bonus = min(0.2, matched_items * 0.1)
        
        # 数量准确性加成
        quantity_bonus = 0.0
        for obj in objects:
            if obj.quantity > 1 and obj.item_type == "main_dish":
                quantity_bonus += 0.05
        
        final_confidence = min(1.0, 
            obj_confidence + lang_factor + intent_factor + 
            length_factor + menu_match_bonus + quantity_bonus
        )
        
        return round(final_confidence, 3)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        cache_total = self.context_parser.cache_hits + self.context_parser.cache_misses
        cache_hit_rate = (self.context_parser.cache_hits / max(cache_total, 1)) * 100
        
        return {
            **self.performance_stats,
            'cache_hit_rate': round(cache_hit_rate, 2),
            'cache_size': len(self.context_parser.cache)
        }

def parse(text: str, context: Optional[ParseContext] = None) -> Dict[str, Any]:
    """
    主解析函数 - 外部调用接口（优化版）
    
    Args:
        text: 用户输入的自然语言文本
        context: 解析上下文
        
    Returns:
        CO对象的字典表示
    """
    start_time = time.time()
    parser = MultiLanguageSeedParser()
    
    # 统计更新
    parser.performance_stats['total_parses'] += 1
    
    # 预处理文本
    cleaned_text = text.strip()
    if not cleaned_text:
        return {
            'objects': [],
            'intent': 'inquiry',
            'conditions': [],
            'language': context.session_language if context and context.session_language else 'es',
            'confidence': 0.0,
            'raw_text': text,
            'context_used': False,
            'parse_time': 0.0,
            'alternatives': []
        }
    
    # 检查缓存
    cache_key = parser.context_parser.get_cache_key(cleaned_text, context)
    if cache_key in parser.context_parser.cache:
        parser.context_parser.cache_hits += 1
        cached_result = parser.context_parser.cache[cache_key].copy()
        cached_result['parse_time'] = time.time() - start_time
        return cached_result
    else:
        parser.context_parser.cache_misses += 1
    
    # 检测语言
    context_lang = context.session_language if context else None
    language, lang_confidence = parser.lang_detector.detect(cleaned_text, context_lang)
    
    # 检测意图
    intent, intent_confidence = parser._detect_intent_advanced(cleaned_text, context)
    
    # 提取对象
    objects = []
    objects.extend(parser._extract_food_items_advanced(cleaned_text, language))
    objects.extend(parser._extract_modifications_advanced(cleaned_text))
    
    # 上下文增强
    if context:
        objects = parser.context_parser.enhance_parsing_with_context(objects, context)
    
    # 提取条件
    conditions = parser._extract_conditions_advanced(cleaned_text)
    
    # 计算置信度
    confidence = parser._calculate_confidence_advanced(
        objects, intent, cleaned_text, language, lang_confidence
    )
    
    # 生成可选解析结果
    alternatives = []
    if confidence < 0.7:  # 如果置信度不高，提供备选方案
        # 可以在这里添加备选解析逻辑
        pass
    
    parse_time = time.time() - start_time
    
    # 更新性能统计
    parser.performance_stats['avg_parse_time'] = (
        (parser.performance_stats['avg_parse_time'] * (parser.performance_stats['total_parses'] - 1) + parse_time) /
        parser.performance_stats['total_parses']
    )
    
    # 构建结果
    result = {
        'objects': [
            {
                'item_type': obj.item_type,
                'content': obj.content,
                'quantity': obj.quantity,
                'modifiers': obj.modifiers,
                'confidence': obj.confidence,
                'normalized_content': obj.normalized_content,
                'position': {'start': obj.start_pos, 'end': obj.end_pos} if obj.start_pos >= 0 else None
            }
            for obj in objects
        ],
        'intent': intent,
        'intent_confidence': intent_confidence,
        'conditions': conditions,
        'language': language,
        'language_confidence': lang_confidence,
        'confidence': confidence,
        'raw_text': cleaned_text,
        'context_used': context is not None,
        'parse_time': round(parse_time, 4),
        'alternatives': alternatives,
        'performance_stats': parser.get_performance_stats() if len(parser.context_parser.cache) % 100 == 0 else None
    }
    
    # 缓存结果（限制缓存大小）
    if len(parser.context_parser.cache) < 1000:
        parser.context_parser.cache[cache_key] = result.copy()
    elif len(parser.context_parser.cache) >= 1000:
        # 清理最老的缓存项
        oldest_key = next(iter(parser.context_parser.cache))
        del parser.context_parser.cache[oldest_key]
        parser.context_parser.cache[cache_key] = result.copy()
    
    return result

def parse_with_context(text: str, session_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    带上下文的解析函数
    
    Args:
        text: 用户输入文本
        session_data: 会话数据
        
    Returns:
        解析结果
    """
    context = None
    if session_data:
        context = ParseContext(
            previous_orders=session_data.get('previous_orders', []),
            session_language=session_data.get('language'),
            preferred_items=set(session_data.get('preferred_items', [])),
            conversation_flow=session_data.get('conversation_flow', 'initial')
        )
    
    return parse(text, context)

def batch_parse(texts: List[str], context: Optional[ParseContext] = None) -> List[Dict[str, Any]]:
    """
    批量解析函数
    
    Args:
        texts: 文本列表
        context: 共享上下文
        
    Returns:
        解析结果列表
    """
    results = []
    for text in texts:
        result = parse(text, context)
        results.append(result)
        
        # 更新上下文中的语言信息
        if context and result['confidence'] > 0.8:
            context.session_language = result['language']
    
    return results

def analyze_parsing_performance(test_cases: List[str]) -> Dict[str, Any]:
    """
    分析解析性能
    
    Args:
        test_cases: 测试用例列表
        
    Returns:
        性能分析报告
    """
    parser = MultiLanguageSeedParser()
    results = []
    
    start_time = time.time()
    for text in test_cases:
        result = parse(text)
        results.append(result)
    
    total_time = time.time() - start_time
    
    # 统计分析
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    high_confidence_count = sum(1 for r in results if r['confidence'] > 0.8)
    low_confidence_count = sum(1 for r in results if r['confidence'] < 0.5)
    
    language_distribution = {}
    for result in results:
        lang = result['language']
        language_distribution[lang] = language_distribution.get(lang, 0) + 1
    
    intent_distribution = {}
    for result in results:
        intent = result['intent']
        intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
    
    return {
        'total_cases': len(test_cases),
        'total_time': round(total_time, 3),
        'avg_time_per_case': round(total_time / len(test_cases), 4),
        'avg_confidence': round(avg_confidence, 3),
        'high_confidence_rate': round(high_confidence_count / len(test_cases) * 100, 1),
        'low_confidence_rate': round(low_confidence_count / len(test_cases) * 100, 1),
        'language_distribution': language_distribution,
        'intent_distribution': intent_distribution,
        'performance_stats': parser.get_performance_stats()
    }

def validate_menu_coverage(menu_items: List[str], test_texts: List[str]) -> Dict[str, Any]:
    """
    验证菜单项覆盖率
    
    Args:
        menu_items: 菜单项列表
        test_texts: 测试文本列表
        
    Returns:
        覆盖率分析
    """
    parser = MultiLanguageSeedParser()
    detected_items = set()
    
    for text in test_texts:
        result = parse(text)
        for obj in result['objects']:
            if obj['item_type'] == 'main_dish':
                detected_items.add(obj['normalized_content'])
    
    menu_items_lower = [item.lower() for item in menu_items]
    covered_items = []
    missed_items = []
    
    for item in menu_items_lower:
        if any(item in detected for detected in detected_items):
            covered_items.append(item)
        else:
            missed_items.append(item)
    
    coverage_rate = len(covered_items) / len(menu_items) * 100 if menu_items else 0
    
    return {
        'total_menu_items': len(menu_items),
        'covered_items': len(covered_items),
        'missed_items': len(missed_items),
        'coverage_rate': round(coverage_rate, 1),
        'covered_list': covered_items,
        'missed_list': missed_items,
        'extra_detected': [item for item in detected_items if item not in menu_items_lower]
    }

# 优化的测试函数
def run_comprehensive_tests():
    """运行综合测试"""
    
    test_cases = [
        # 中文测试
        "我要一个Pollo Teriyaki",
        "两份鸡肉套餐，不要洋葱",
        "来三个牛肉combo，打包",
        
        # 西班牙语测试
        "Quiero 2 Combinaciones Brócoli con Carne, papa cambio tostones",
        "Dame 3 presas de pollo, no salsa",
        "Necesito un combo familiar para llevar",
        
        # 英文测试
        "I want chicken with rice, no onions",
        "Order 2 beef combos, extra sauce",
        "Get me 5 pieces of chicken, dine in",
        
        # 混合语言测试
        "我要 2 pollo teriyaki, sin cebolla",
        "Quiero 一个 beef combo",
        
        # 修改意图测试
        "换成牛肉",
        "不要辣椒",
        "加多点饭",
        
        # 确认/取消测试
        "对，确认",
        "不对，取消",
        "算了，不要了",
        
        # 复杂订单测试
        "我要两个鸡肉套餐，一个不要洋葱，一个加辣椒，都要打包",
        "Quiero 3 combos: dos con pollo y uno con carne, todos para llevar"
    ]
    
    print("=== 优化版 Seed Parser 综合测试 ===\n")
    
    # 性能测试
    print("1. 性能分析:")
    perf_report = analyze_parsing_performance(test_cases)
    for key, value in perf_report.items():
        print(f"   {key}: {value}")
    
    print("\n2. 解析结果示例:")
    for i, test in enumerate(test_cases[:5]):  # 只显示前5个
        print(f"\n   测试 {i+1}: {test}")
        result = parse(test)
        print(f"   语言: {result['language']} (置信度: {result['language_confidence']})")
        print(f"   意图: {result['intent']} (置信度: {result['intent_confidence']})")
        print(f"   整体置信度: {result['confidence']}")
        print(f"   解析时间: {result['parse_time']}秒")
        if result['objects']:
            print("   提取对象:")
            for obj in result['objects']:
                print(f"     - {obj['item_type']}: {obj['content']} (数量: {obj['quantity']}, 置信度: {obj['confidence']})")
        if result['conditions']:
            print(f"   条件: {result['conditions']}")
    
    # 菜单覆盖率测试
    print("\n3. 菜单覆盖率:")
    sample_menu = [
        "Pollo Teriyaki", "Carne con Brócoli", "Combo Familiar",
        "Arroz con Pollo", "Papa Rellena", "Tostones"
    ]
    coverage_report = validate_menu_coverage(sample_menu, test_cases)
    for key, value in coverage_report.items():
        if not isinstance(value, list) or len(value) < 10:  # 避免打印过长的列表
            print(f"   {key}: {value}")
    
    # 上下文测试
    print("\n4. 上下文感知测试:")
    context = ParseContext(
        previous_orders=[{"item": "pollo teriyaki", "quantity": 1}],
        session_language="es",
        preferred_items={"pollo", "arroz"},
        conversation_flow="ordering"
    )
    
    context_test = "quiero lo mismo pero sin cebolla"
    result_with_context = parse(context_test, context)
    result_without_context = parse(context_test)
    
    print(f"   输入: {context_test}")
    print(f"   有上下文置信度: {result_with_context['confidence']}")
    print(f"   无上下文置信度: {result_without_context['confidence']}")
    print(f"   上下文提升: {result_with_context['confidence'] - result_without_context['confidence']:.3f}")

# 测试函数
if __name__ == "__main__":
    run_comprehensive_tests()
