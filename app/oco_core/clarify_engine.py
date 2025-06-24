#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Clarify Engine
多语言澄清生成引擎
"""

import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from ..config import settings

@dataclass
class ClarificationTemplate:
    """澄清模板"""
    type: str              # 'multiple_choice', 'confirmation', 'specification'
    spanish: str
    english: str
    chinese: str
    context: str           # 使用场景

class MultilingualClarifyEngine:
    """多语言澄清引擎"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """加载菜单知识库"""
        try:
            kb_path = Path(settings.MENU_KB_FILE)
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load knowledge base: {e}")
        return {}
    
    def _load_templates(self) -> Dict[str, List[ClarificationTemplate]]:
        """加载澄清模板"""
        templates = {
            'multiple_matches': [
                ClarificationTemplate(
                    type='multiple_choice',
                    spanish='Encontré varias opciones para "{item}". ¿Cuál te gustaría?\n{options}',
                    english='I found several options for "{item}". Which one would you like?\n{options}',
                    chinese='我找到了几个"{item}"的选项，您要哪一个？\n{options}',
                    context='当有多个菜品匹配时'
                ),
                ClarificationTemplate(
                    type='multiple_choice',
                    spanish='Tenemos estas opciones de {item}:\n{options}\n¿Cuál prefieres?',
                    english='We have these {item} options:\n{options}\nWhich do you prefer?',
                    chinese='我们有这些{item}选项：\n{options}\n您比较喜欢哪个？',
                    context='列出选项让用户选择'
                )
            ],
            'low_confidence': [
                ClarificationTemplate(
                    type='specification',
                    spanish='Disculpa, no estoy seguro de entender "{query}". ¿Podrías ser más específico?',
                    english='Sorry, I\'m not sure I understand "{query}". Could you be more specific?',
                    chinese='抱歉，我不太确定您说的"{query}"是什么意思，能具体说明一下吗？',
                    context='当理解置信度低时'
                ),
                ClarificationTemplate(
                    type='specification',
                    spanish='¿Te refieres a alguno de estos platos? {suggestions}',
                    english='Do you mean any of these dishes? {suggestions}',
                    chinese='您是指这些菜品中的哪一个吗？{suggestions}',
                    context='提供建议选项'
                )
            ],
            'quantity_clarification': [
                ClarificationTemplate(
                    type='confirmation',
                    spanish='¿Quieres {quantity} de {item}?',
                    english='Do you want {quantity} {item}?',
                    chinese='您要{quantity}份{item}吗？',
                    context='确认数量'
                ),
                ClarificationTemplate(
                    type='specification',
                    spanish='¿Cuántos {item} te gustaría?',
                    english='How many {item} would you like?',
                    chinese='您要几份{item}？',
                    context='询问数量'
                )
            ],
            'modification_clarification': [
                ClarificationTemplate(
                    type='confirmation',
                    spanish='¿Quieres {item} {modification}?',
                    english='Do you want {item} {modification}?',
                    chinese='您要{item}{modification}吗？',
                    context='确认修改'
                ),
                ClarificationTemplate(
                    type='specification',
                    spanish='Para el {item}, ¿qué cambios te gustaría hacer?',
                    english='For the {item}, what changes would you like to make?',
                    chinese='对于{item}，您想要做什么修改？',
                    context='询问具体修改'
                )
            ],
            'size_clarification': [
                ClarificationTemplate(
                    type='multiple_choice',
                    spanish='Tenemos {item} en tamaño pequeño (${small_price}) y grande (${large_price}). ¿Cuál prefieres?',
                    english='We have {item} in small (${small_price}) and large (${large_price}) sizes. Which do you prefer?',
                    chinese='我们有小份{item}(${small_price})和大份{item}(${large_price})，您要哪种？',
                    context='询问尺寸大小'
                )
            ],
            'combo_clarification': [
                ClarificationTemplate(
                    type='multiple_choice',
                    spanish='Para {item}, ¿quieres la combinación completa (${combo_price}) o solo el plato principal (${main_price})?',
                    english='For {item}, do you want the full combo (${combo_price}) or just the main dish (${main_price})?',
                    chinese='对于{item}，您要套餐(${combo_price})还是只要主菜(${main_price})？',
                    context='询问是否要套餐'
                )
            ],
            'no_matches': [
                ClarificationTemplate(
                    type='specification',
                    spanish='Lo siento, no encontré "{item}" en nuestro menú. ¿Podrías revisar el nombre o te gustaría ver nuestras opciones populares?',
                    english='Sorry, I couldn\'t find "{item}" on our menu. Could you check the name or would you like to see our popular options?',
                    chinese='抱歉，我在菜单中没有找到"{item}"。您能检查一下名称，或者想看看我们的热门选项吗？',
                    context='找不到匹配项'
                )
            ],
            'order_confirmation': [
                ClarificationTemplate(
                    type='confirmation',
                    spanish='Perfecto, confirmo tu orden:\n{order_summary}\nTotal: ${total}\n¿Es correcto?',
                    english='Perfect, I confirm your order:\n{order_summary}\nTotal: ${total}\nIs this correct?',
                    chinese='好的，确认您的订单：\n{order_summary}\n总计：${total}\n正确吗？',
                    context='最终确认订单'
                )
            ]
        }
        
        return templates
    
    def _format_options_list(self, matches: List[Dict[str, Any]], language: str) -> str:
        """格式化选项列表"""
        if not matches:
            return ""
        
        options = []
        for i, match in enumerate(matches[:5], 1):  # 最多显示5个选项
            item_name = match.get('item_name', '')
            price = match.get('price', 0.0)
            
            if language == 'zh':
                option = f"{i}. {item_name} - ${price:.2f}"
            elif language == 'en':
                option = f"{i}. {item_name} - ${price:.2f}"
            else:  # Spanish
                option = f"{i}. {item_name} - ${price:.2f}"
            
            options.append(option)
        
        return '\n'.join(options)
    
    def _get_template(self, template_type: str, language: str = 'es') -> Optional[str]:
        """获取指定类型和语言的模板"""
        if template_type not in self.templates:
            return None
        
        template_list = self.templates[template_type]
        if not template_list:
            return None
        
        # 随机选择一个模板以增加变化
        template = random.choice(template_list)
        
        if language == 'zh' or language == 'zh-CN':
            return template.chinese
        elif language == 'en' or language == 'en-US':
            return template.english
        else:  # 默认西班牙语
            return template.spanish
    
    def build_multiple_matches_clarification(self, co: Dict[str, Any], path_data: Dict[str, Any]) -> str:
        """构建多重匹配澄清"""
        language = co.get('language', 'es')
        original_query = co.get('raw_text', '')
        alternative_paths = path_data.get('alternative_paths', [])
        
        if not alternative_paths:
            return self._get_template('no_matches', language).format(item=original_query)
        
        # 构建选项列表
        matches = []
        for path in alternative_paths:
            if path.get('matches'):
                matches.extend(path['matches'])
        
        # 去重并限制数量
        seen_items = set()
        unique_matches = []
        for match in matches:
            item_id = match.get('item_id')
            if item_id not in seen_items:
                seen_items.add(item_id)
                unique_matches.append(match)
        
        if not unique_matches:
            return self._get_template('no_matches', language).format(item=original_query)
        
        template = self._get_template('multiple_matches', language)
        options_list = self._format_options_list(unique_matches, language)
        
        return template.format(
            item=original_query,
            options=options_list
        )
    
    def build_low_confidence_clarification(self, co: Dict[str, Any], path_data: Dict[str, Any]) -> str:
        """构建低置信度澄清"""
        language = co.get('language', 'es')
        original_query = co.get('raw_text', '')
        
        # 尝试从知识库中找到相似的建议
        suggestions = self._get_menu_suggestions(original_query)
        
        template = self._get_template('low_confidence', language)
        
        if suggestions:
            suggestion_names = [s['item_name'] for s in suggestions[:3]]
            if language == 'zh':
                suggestions_text = '、'.join(suggestion_names)
            else:
                suggestions_text = ', '.join(suggestion_names)
            
            return template.format(
                query=original_query,
                suggestions=suggestions_text
            )
        else:
            return template.format(query=original_query)
    
    def build_quantity_clarification(self, co: Dict[str, Any], item_name: str, suggested_quantity: int = 1) -> str:
        """构建数量澄清"""
        language = co.get('language', 'es')
        template = self._get_template('quantity_clarification', language)
        
        return template.format(
            quantity=suggested_quantity,
            item=item_name
        )
    
    def build_size_clarification(self, co: Dict[str, Any], item_name: str, size_options: List[Dict[str, Any]]) -> str:
        """构建尺寸澄清"""
        language = co.get('language', 'es')
        template = self._get_template('size_clarification', language)
        
        if len(size_options) >= 2:
            small_option = min(size_options, key=lambda x: x.get('price', 0))
            large_option = max(size_options, key=lambda x: x.get('price', 0))
            
            return template.format(
                item=item_name,
                small_price=small_option.get('price', 0),
                large_price=large_option.get('price', 0)
            )
        
        return self._get_template('low_confidence', language).format(query=item_name)
    
    def build_order_confirmation(self, co: Dict[str, Any], order_items: List[Dict[str, Any]], total_price: float) -> str:
        """构建订单确认"""
        language = co.get('language', 'es')
        template = self._get_template('order_confirmation', language)
        
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
        
        return template.format(
            order_summary=order_summary,
            total=total_price
        )
    
    def build_modification_clarification(self, co: Dict[str, Any], item_name: str, modification: str) -> str:
        """构建修改澄清"""
        language = co.get('language', 'es')
        template = self._get_template('modification_clarification', language)
        
        return template.format(
            item=item_name,
            modification=modification
        )
    
    def _get_menu_suggestions(self, query: str, max_suggestions: int = 3) -> List[Dict[str, Any]]:
        """从菜单中获取建议"""
        suggestions = []
        
        # 从知识库中搜索相似项目
        categories = self.knowledge_base.get('menu_categories', {})
        query_lower = query.lower()
        
        for category_data in categories.values():
            items = category_data.get('items', [])
            for item in items:
                item_name = item.get('item_name', '').lower()
                aliases = [alias.lower() for alias in item.get('aliases', [])]
                keywords = [kw.lower() for kw in item.get('keywords', [])]
                
                # 检查是否有部分匹配
                if (query_lower in item_name or 
                    any(query_lower in alias for alias in aliases) or
                    any(query_lower in keyword for keyword in keywords) or
                    any(keyword in query_lower for keyword in keywords)):
                    
                    suggestions.append({
                        'item_name': item.get('item_name'),
                        'price': item.get('price', 0.0),
                        'match_score': 0.5  # 部分匹配分数
                    })
        
        # 按匹配分数排序并限制数量
        suggestions.sort(key=lambda x: x['match_score'], reverse=True)
        return suggestions[:max_suggestions]
    
    def detect_clarification_type(self, co: Dict[str, Any], path_data: Dict[str, Any]) -> str:
        """检测需要的澄清类型"""
        clarification_reason = path_data.get('clarification_reason', '')
        alternative_paths = path_data.get('alternative_paths', [])
        requires_clarification = path_data.get('requires_clarification', False)
        
        if not requires_clarification:
            return 'none'
        
        if clarification_reason == 'multiple_matches' and alternative_paths:
            return 'multiple_matches'
        elif clarification_reason == 'low_confidence':
            return 'low_confidence'
        elif clarification_reason == 'no_menu_matches':
            return 'no_matches'
        else:
            return 'low_confidence'  # 默认

def build(co: Dict[str, Any], path_data: Dict[str, Any] = None) -> str:
    """
    外部调用接口 - 构建澄清消息
    
    Args:
        co: 对话对象
        path_data: 路径数据
        
    Returns:
        澄清消息字符串
    """
    engine = MultilingualClarifyEngine()
    
    if not path_data:
        # 简单的澄清请求
        language = co.get('language', 'es')
        if language == 'zh':
            return "抱歉，我没有理解您的意思，能再说一遍吗？"
        elif language == 'en':
            return "Sorry, I didn't understand. Could you please repeat that?"
        else:
            return "Disculpa, no entendí. ¿Podrías repetir?"
    
    # 检测澄清类型并生成相应消息
    clarification_type = engine.detect_clarification_type(co, path_data)
    
    if clarification_type == 'multiple_matches':
        return engine.build_multiple_matches_clarification(co, path_data)
    elif clarification_type == 'low_confidence':
        return engine.build_low_confidence_clarification(co, path_data)
    elif clarification_type == 'no_matches':
        language = co.get('language', 'es')
        template = engine._get_template('no_matches', language)
        return template.format(item=co.get('raw_text', ''))
    else:
        # 默认澄清
        language = co.get('language', 'es')
        if language == 'zh':
            return "请问您需要什么帮助？"
        elif language == 'en':
            return "How can I help you?"
        else:
            return "¿En qué puedo ayudarte?"

def build_order_confirmation(co: Dict[str, Any], order_items: List[Dict[str, Any]], total_price: float) -> str:
    """构建订单确认消息"""
    engine = MultilingualClarifyEngine()
    return engine.build_order_confirmation(co, order_items, total_price)

def build_quantity_clarification(co: Dict[str, Any], item_name: str, quantity: int = 1) -> str:
    """构建数量澄清消息"""
    engine = MultilingualClarifyEngine()
    return engine.build_quantity_clarification(co, item_name, quantity)

def build_modification_clarification(co: Dict[str, Any], item_name: str, modification: str) -> str:
    """构建修改澄清消息"""
    engine = MultilingualClarifyEngine()
    return engine.build_modification_clarification(co, item_name, modification)


# 测试函数
if __name__ == "__main__":
    # 测试澄清引擎
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
