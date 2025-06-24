#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Seed Parser
多语言自然语言订单解析器
"""

import re
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from ..config import settings

@dataclass
class ParsedObject:
    """解析出的订单对象"""
    item_type: str          # 'main_dish', 'modification', 'quantity'
    content: str            # 原始文本
    quantity: int = 1       # 数量
    modifiers: List[str] = None  # 修饰词
    confidence: float = 0.0 # 置信度
    
    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = []

@dataclass
class ConversationObject:
    """对话对象 - CO结构"""
    objects: List[ParsedObject]
    intent: str                    # 'order', 'modify', 'confirm', 'cancel', 'inquiry'
    conditions: List[str]          # 约束条件
    language: str = 'es'          # 检测到的主要语言
    confidence: float = 0.0       # 整体解析置信度
    raw_text: str = ''            # 原始输入

class MultiLanguageSeedParser:
    """多语言种子解析器"""
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.patterns = self._build_patterns()
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """加载菜单知识库"""
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
    
    def _build_patterns(self) -> Dict[str, List[str]]:
        """构建多语言匹配模式"""
        rules = self.knowledge_base.get("ai_parsing_rules", {})
        
        return {
            # 订单意图模式
            "order_intent": [
                r"(?:我?要|点|来|给我|quiero|dame|want|order|get|necesito)\s*(.+)",
                r"(.+)\s*(?:个|份|块|件|piezas?|presas?|combos?)",
                r"(\d+)\s*(.+)"
            ],
            
            # 数量提取模式
            "quantity": [
                r"(\d+)\s*(?:个|份|块|件|piezas?|presas?|combos?|pieces?)",
                r"(?:uno?|dos|tres|cuatro|cinco|六|七|八|九|十|one|two|three|four|five|1|2|3|4|5|6|7|8|9|10|15|20)\s*(.+)",
                r"^(\d+)\s+(.+)"
            ],
            
            # 修改意图模式
            "modification": [
                r"(?:换|改|cambio|change)\s*(.+)",
                r"(?:不要|no|sin|without)\s*(.+)",
                r"(?:加|extra|más|more)\s*(.+)",
                r"(?:少|poco|less)\s*(.+)"
            ],
            
            # 确认模式
            "confirmation": [
                r"(?:对|是|好|确认|sí|yes|correcto|ok|确定)",
                r"(?:不对|不是|错|no|incorrecto|wrong|不对)"
            ],
            
            # 取消模式
            "cancel": [
                r"(?:取消|cancel|cancelar|不要了|算了)"
            ]
        }
    
    def _detect_language(self, text: str) -> str:
        """检测主要语言"""
        text_lower = text.lower()
        
        # 中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 西班牙语关键词
        spanish_words = len(re.findall(r'\b(?:pollo|carne|arroz|papa|quiero|con|sin|más|poco)\b', text_lower))
        # 英文单词
        english_words = len(re.findall(r'\b(?:chicken|beef|rice|potato|want|with|without|more|less)\b', text_lower))
        
        if chinese_chars > 0:
            return 'zh'
        elif spanish_words >= english_words:
            return 'es'
        else:
            return 'en'
    
    def _extract_quantities(self, text: str) -> List[Tuple[int, str]]:
        """提取数量和对应的物品"""
        results = []
        
        for pattern in self.patterns["quantity"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match.groups()) == 2:
                        qty_str, item = match.groups()
                        quantity = self._parse_number(qty_str)
                        if quantity > 0:
                            results.append((quantity, item.strip()))
                    elif len(match.groups()) == 1:
                        # 处理数字在前面的情况
                        full_match = match.group(0)
                        qty_match = re.search(r'(\d+)', full_match)
                        if qty_match:
                            quantity = int(qty_match.group(1))
                            item = full_match.replace(qty_match.group(0), '').strip()
                            if item:
                                results.append((quantity, item))
                except (ValueError, AttributeError):
                    continue
        
        return results
    
    def _parse_number(self, text: str) -> int:
        """解析各种语言的数字表达"""
        text = text.strip().lower()
        
        # 直接数字
        if text.isdigit():
            return int(text)
        
        # 中文数字映射
        chinese_numbers = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '两': 2, '双': 2
        }
        
        # 西班牙语数字映射
        spanish_numbers = {
            'uno': 1, 'una': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
            'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
            'quince': 15, 'veinte': 20
        }
        
        # 英文数字映射
        english_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'fifteen': 15, 'twenty': 20
        }
        
        for num_dict in [chinese_numbers, spanish_numbers, english_numbers]:
            if text in num_dict:
                return num_dict[text]
        
        return 1
    
    def _detect_intent(self, text: str) -> str:
        """检测用户意图"""
        text_lower = text.lower()
        
        # 取消意图
        for pattern in self.patterns["cancel"]:
            if re.search(pattern, text_lower):
                return "cancel"
        
        # 确认意图
        for pattern in self.patterns["confirmation"]:
            if re.search(pattern, text_lower):
                return "confirm"
        
        # 修改意图
        for pattern in self.patterns["modification"]:
            if re.search(pattern, text_lower):
                return "modify"
        
        # 订单意图
        for pattern in self.patterns["order_intent"]:
            if re.search(pattern, text_lower):
                return "order"
        
        # 默认为询问
        return "inquiry"
    
    def _extract_food_items(self, text: str) -> List[ParsedObject]:
        """提取食物项目"""
        objects = []
        
        # 首先提取带数量的项目
        quantity_items = self._extract_quantities(text)
        processed_text = text
        
        for quantity, item in quantity_items:
            obj = ParsedObject(
                item_type="main_dish",
                content=item,
                quantity=quantity,
                confidence=0.8
            )
            objects.append(obj)
            # 从文本中移除已处理的部分
            processed_text = processed_text.replace(f"{quantity} {item}", "").replace(f"{quantity}{item}", "")
        
        # 处理剩余的订单意图
        for pattern in self.patterns["order_intent"]:
            matches = re.finditer(pattern, processed_text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    item_text = match.group(1).strip()
                    if item_text and len(item_text) > 1:
                        obj = ParsedObject(
                            item_type="main_dish",
                            content=item_text,
                            quantity=1,
                            confidence=0.6
                        )
                        objects.append(obj)
        
        return objects
    
    def _extract_modifications(self, text: str) -> List[ParsedObject]:
        """提取修改要求"""
        modifications = []
        
        for pattern in self.patterns["modification"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    mod_text = match.group(1).strip()
                    if mod_text:
                        mod_type = "modification"
                        if re.search(r"(?:不要|no|sin|without)", match.group(0).lower()):
                            mod_type = "remove"
                        elif re.search(r"(?:加|extra|más|more)", match.group(0).lower()):
                            mod_type = "add"
                        elif re.search(r"(?:换|cambio|change)", match.group(0).lower()):
                            mod_type = "change"
                        elif re.search(r"(?:少|poco|less)", match.group(0).lower()):
                            mod_type = "reduce"
                        
                        obj = ParsedObject(
                            item_type=mod_type,
                            content=mod_text,
                            confidence=0.7
                        )
                        modifications.append(obj)
        
        return modifications
    
    def _extract_conditions(self, text: str) -> List[str]:
        """提取约束条件"""
        conditions = []
        
        # 分开装盘
        if re.search(r"(?:分开|aparte|separate)", text, re.IGNORECASE):
            conditions.append("separate_serving")
        
        # 打包
        if re.search(r"(?:打包|para llevar|takeaway|to go)", text, re.IGNORECASE):
            conditions.append("takeaway")
        
        # 堂食
        if re.search(r"(?:堂食|para comer aquí|dine in)", text, re.IGNORECASE):
            conditions.append("dine_in")
        
        # 急单
        if re.search(r"(?:急|快|急单|rápido|urgente|quickly)", text, re.IGNORECASE):
            conditions.append("urgent")
        
        return conditions
    
    def _calculate_confidence(self, objects: List[ParsedObject], intent: str, text: str) -> float:
        """计算整体解析置信度"""
        if not objects:
            return 0.1
        
        # 基础置信度
        base_confidence = sum(obj.confidence for obj in objects) / len(objects)
        
        # 意图明确性加成
        intent_bonus = 0.2 if intent in ["order", "modify", "confirm"] else 0.0
        
        # 文本长度合理性
        length_factor = min(1.0, len(text.strip()) / 10) * 0.1
        
        # 关键词匹配加成
        keyword_bonus = 0.0
        rules = self.knowledge_base.get("ai_parsing_rules", {})
        all_keywords = []
        for category in rules.get("intent_detection", {}).values():
            if isinstance(category, list):
                all_keywords.extend(category)
        
        text_lower = text.lower()
        matched_keywords = sum(1 for kw in all_keywords if kw in text_lower)
        keyword_bonus = min(0.2, matched_keywords * 0.05)
        
        final_confidence = min(1.0, base_confidence + intent_bonus + length_factor + keyword_bonus)
        return round(final_confidence, 2)

def parse(text: str) -> Dict[str, Any]:
    """
    主解析函数 - 外部调用接口
    
    Args:
        text: 用户输入的自然语言文本
        
    Returns:
        CO对象的字典表示
    """
    parser = MultiLanguageSeedParser()
    
    # 预处理文本
    cleaned_text = text.strip()
    if not cleaned_text:
        return {
            'objects': [],
            'intent': 'inquiry',
            'conditions': [],
            'language': 'es',
            'confidence': 0.0,
            'raw_text': text
        }
    
    # 检测语言
    language = parser._detect_language(cleaned_text)
    
    # 检测意图
    intent = parser._detect_intent(cleaned_text)
    
    # 提取对象
    objects = []
    objects.extend(parser._extract_food_items(cleaned_text))
    objects.extend(parser._extract_modifications(cleaned_text))
    
    # 提取条件
    conditions = parser._extract_conditions(cleaned_text)
    
    # 计算置信度
    confidence = parser._calculate_confidence(objects, intent, cleaned_text)
    
    # 构建CO对象
    co = ConversationObject(
        objects=objects,
        intent=intent,
        conditions=conditions,
        language=language,
        confidence=confidence,
        raw_text=cleaned_text
    )
    
    # 转换为字典返回
    return {
        'objects': [
            {
                'item_type': obj.item_type,
                'content': obj.content,
                'quantity': obj.quantity,
                'modifiers': obj.modifiers,
                'confidence': obj.confidence
            }
            for obj in co.objects
        ],
        'intent': co.intent,
        'conditions': co.conditions,
        'language': co.language,
        'confidence': co.confidence,
        'raw_text': co.raw_text
    }


# 测试函数
if __name__ == "__main__":
    test_cases = [
        "我要一个Pollo Teriyaki",
        "2 Combinaciones Brocoli con Carne, papa cambio tostones",
        "quiero 3 presas de pollo, no salsa",
        "I want chicken with rice, no onions",
        "取消订单",
        "对，确认"
    ]
    
    for test in test_cases:
        print(f"\n输入: {test}")
        result = parse(test)
        print(f"解析结果: {result}")
