#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Jump Planner
菜单项匹配与跳跃路径规划器
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from ..config import settings

@dataclass
class MenuMatch:
    """菜单匹配结果"""
    item_id: str
    item_name: str
    variant_id: str
    variant_name: str
    category_name: str
    price: float
    sku: str
    match_score: float
    match_type: str  # 'exact', 'alias', 'keyword', 'fuzzy'
    original_query: str

@dataclass
class PlanPath:
    """规划路径"""
    matches: List[MenuMatch]
    total_score: float
    confidence: float
    requires_clarification: bool
    clarification_reason: str = ""

class MenuFuzzyMatcher:
    """菜单模糊匹配器"""
    
    def __init__(self):
        self.menu_data = self._load_menu_data()
        self.search_index = self._build_search_index()
    
    def _load_menu_data(self) -> Dict[str, Any]:
        """加载菜单数据"""
        try:
            kb_path = Path(settings.MENU_KB_FILE)
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load menu data: {e}")
        
        # 返回默认的菜单结构
        return {
            "menu_categories": {
                "combinaciones": {
                    "name": "Combinaciones",
                    "items": []
                },
                "adicionales": {
                    "name": "Adicionales", 
                    "items": []
                }
            }
        }
    
    def _build_search_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """构建搜索索引"""
        index = {
            'exact_names': {},
            'aliases': {},
            'keywords': {},
            'all_items': []
        }
        
        categories = self.menu_data.get("menu_categories", {})
        
        for category_key, category_data in categories.items():
            items = category_data.get("items", [])
            
            for item in items:
                item_info = {
                    'item_id': item.get('item_id', ''),
                    'item_name': item.get('item_name', ''),
                    'variant_id': item.get('variant_id', ''),
                    'variant_name': item.get('variant_name', '默认'),
                    'category_name': category_data.get('name', category_key),
                    'category_key': category_key,
                    'price': item.get('price', 0.0),
                    'sku': item.get('sku', ''),
                    'aliases': item.get('aliases', []),
                    'keywords': item.get('keywords', [])
                }
                
                index['all_items'].append(item_info)
                
                # 精确名称索引
                exact_name = item['item_name'].lower()
                if exact_name not in index['exact_names']:
                    index['exact_names'][exact_name] = []
                index['exact_names'][exact_name].append(item_info)
                
                # 别名索引
                for alias in item.get('aliases', []):
                    alias_key = alias.lower()
                    if alias_key not in index['aliases']:
                        index['aliases'][alias_key] = []
                    index['aliases'][alias_key].append(item_info)
                
                # 关键词索引
                for keyword in item.get('keywords', []):
                    keyword_key = keyword.lower()
                    if keyword_key not in index['keywords']:
                        index['keywords'][keyword_key] = []
                    index['keywords'][keyword_key].append(item_info)
        
        return index
    
    def _fuzzy_similarity(self, text1: str, text2: str) -> float:
        """计算模糊相似度"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询文本"""
        # 移除多余空格
        query = re.sub(r'\s+', ' ', query.strip())
        
        # 移除常见的数量词
        query = re.sub(r'^\d+\s*(?:个|份|块|件|piezas?|presas?|combos?|pieces?)\s*', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\s*\d+\s*(?:个|份|块|件|piezas?|presas?|combos?|pieces?)\s*$', '', query, flags=re.IGNORECASE)
        
        # 移除常见修饰词
        modifiers = ['我要', '给我', '来', '点', 'quiero', 'dame', 'want', 'order', 'get']
        for modifier in modifiers:
            query = re.sub(f'^{re.escape(modifier)}\\s*', '', query, flags=re.IGNORECASE)
        
        return query.strip()
    
    def search_menu_items(self, query: str, max_results: int = 5) -> List[MenuMatch]:
        """搜索菜单项"""
        if not query or not query.strip():
            return []
        
        normalized_query = self._normalize_query(query)
        query_lower = normalized_query.lower()
        matches = []
        
        # 1. 精确匹配
        if query_lower in self.search_index['exact_names']:
            for item in self.search_index['exact_names'][query_lower]:
                match = MenuMatch(
                    item_id=item['item_id'],
                    item_name=item['item_name'],
                    variant_id=item['variant_id'],
                    variant_name=item['variant_name'],
                    category_name=item['category_name'],
                    price=item['price'],
                    sku=item['sku'],
                    match_score=1.0,
                    match_type='exact',
                    original_query=query
                )
                matches.append(match)
        
        # 2. 别名匹配
        if query_lower in self.search_index['aliases']:
            for item in self.search_index['aliases'][query_lower]:
                if not any(m.item_id == item['item_id'] for m in matches):
                    match = MenuMatch(
                        item_id=item['item_id'],
                        item_name=item['item_name'],
                        variant_id=item['variant_id'],
                        variant_name=item['variant_name'],
                        category_name=item['category_name'],
                        price=item['price'],
                        sku=item['sku'],
                        match_score=0.95,
                        match_type='alias',
                        original_query=query
                    )
                    matches.append(match)
        
        # 3. 关键词匹配
        for keyword, items in self.search_index['keywords'].items():
            if keyword in query_lower:
                for item in items:
                    if not any(m.item_id == item['item_id'] for m in matches):
                        # 计算关键词覆盖度
                        keyword_coverage = len(keyword) / len(query_lower)
                        match_score = 0.8 * keyword_coverage
                        
                        match = MenuMatch(
                            item_id=item['item_id'],
                            item_name=item['item_name'],
                            variant_id=item['variant_id'],
                            variant_name=item['variant_name'],
                            category_name=item['category_name'],
                            price=item['price'],
                            sku=item['sku'],
                            match_score=match_score,
                            match_type='keyword',
                            original_query=query
                        )
                        matches.append(match)
        
        # 4. 模糊匹配
        if len(matches) < max_results:
            for item in self.search_index['all_items']:
                if any(m.item_id == item['item_id'] for m in matches):
                    continue
                
                # 对项目名称进行模糊匹配
                name_similarity = self._fuzzy_similarity(query_lower, item['item_name'])
                
                # 对别名进行模糊匹配
                alias_similarity = 0.0
                for alias in item['aliases']:
                    alias_sim = self._fuzzy_similarity(query_lower, alias)
                    alias_similarity = max(alias_similarity, alias_sim)
                
                # 对关键词进行模糊匹配
                keyword_similarity = 0.0
                for keyword in item['keywords']:
                    keyword_sim = self._fuzzy_similarity(query_lower, keyword)
                    keyword_similarity = max(keyword_similarity, keyword_sim)
                
                best_similarity = max(name_similarity, alias_similarity, keyword_similarity)
                
                # 只有相似度超过阈值才加入结果
                if best_similarity >= 0.6:
                    match = MenuMatch(
                        item_id=item['item_id'],
                        item_name=item['item_name'],
                        variant_id=item['variant_id'],
                        variant_name=item['variant_name'],
                        category_name=item['category_name'],
                        price=item['price'],
                        sku=item['sku'],
                        match_score=best_similarity,
                        match_type='fuzzy',
                        original_query=query
                    )
                    matches.append(match)
        
        # 按匹配分数排序并限制结果数量
        matches.sort(key=lambda x: x.match_score, reverse=True)
        return matches[:max_results]

class JumpPlanner:
    """跳跃路径规划器"""
    
    def __init__(self):
        self.matcher = MenuFuzzyMatcher()
        self.clarification_threshold = 0.7
        self.multiple_match_threshold = 3
    
    def plan(self, co: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据CO对象规划跳跃路径
        
        Args:
            co: 对话对象字典
            
        Returns:
            规划结果字典
        """
        objects = co.get('objects', [])
        intent = co.get('intent', 'inquiry')
        confidence = co.get('confidence', 0.0)
        
        if not objects or intent not in ['order', 'modify']:
            return {
                'path': [],
                'score': 0.0,
                'confidence': 0.0,
                'requires_clarification': True,
                'clarification_reason': 'no_valid_items'
            }
        
        paths = []
        
        # 为每个提取的对象寻找匹配
        for obj in objects:
            if obj['item_type'] in ['main_dish']:
                content = obj['content']
                quantity = obj.get('quantity', 1)
                
                # 搜索匹配的菜单项
                matches = self.matcher.search_menu_items(content)
                
                if matches:
                    # 为每个匹配项创建路径
                    for match in matches:
                        # 复制匹配项并设置数量
                        path_match = MenuMatch(
                            item_id=match.item_id,
                            item_name=match.item_name,
                            variant_id=match.variant_id,
                            variant_name=match.variant_name,
                            category_name=match.category_name,
                            price=match.price,
                            sku=match.sku,
                            match_score=match.match_score,
                            match_type=match.match_type,
                            original_query=match.original_query
                        )
                        
                        # 创建路径
                        path = PlanPath(
                            matches=[path_match],
                            total_score=match.match_score,
                            confidence=match.match_score * confidence,
                            requires_clarification=False
                        )
                        paths.append(path)
        
        if not paths:
            return {
                'path': [],
                'score': 0.0,
                'confidence': 0.0,
                'requires_clarification': True,
                'clarification_reason': 'no_menu_matches'
            }
        
        # 选择最佳路径
        best_path = max(paths, key=lambda p: p.confidence)
        
        # 检查是否需要澄清
        requires_clarification = False
        clarification_reason = ""
        
        # 如果有多个高分匹配，需要澄清
        high_score_paths = [p for p in paths if p.total_score >= self.clarification_threshold]
        if len(high_score_paths) >= self.multiple_match_threshold:
            requires_clarification = True
            clarification_reason = "multiple_matches"
        
        # 如果最佳匹配分数过低，需要澄清
        elif best_path.total_score < self.clarification_threshold:
            requires_clarification = True
            clarification_reason = "low_confidence"
        
        best_path.requires_clarification = requires_clarification
        best_path.clarification_reason = clarification_reason
        
        # 转换为返回格式
        return {
            'path': [
                {
                    'item_id': match.item_id,
                    'item_name': match.item_name,
                    'variant_id': match.variant_id,
                    'variant_name': match.variant_name,
                    'category_name': match.category_name,
                    'price': match.price,
                    'sku': match.sku,
                    'match_score': match.match_score,
                    'match_type': match.match_type,
                    'original_query': match.original_query
                }
                for match in best_path.matches
            ],
            'score': best_path.total_score,
            'confidence': best_path.confidence,
            'requires_clarification': requires_clarification,
            'clarification_reason': clarification_reason,
            'alternative_paths': [
                {
                    'matches': [
                        {
                            'item_id': match.item_id,
                            'item_name': match.item_name,
                            'price': match.price,
                            'match_score': match.match_score
                        }
                        for match in path.matches
                    ],
                    'score': path.total_score
                }
                for path in sorted(paths, key=lambda p: p.confidence, reverse=True)[:3]
                if path != best_path
            ]
        }


def plan(co: Dict[str, Any]) -> Dict[str, Any]:
    """
    外部调用接口 - 规划跳跃路径
    
    Args:
        co: 对话对象字典
        
    Returns:
        规划结果字典
    """
    planner = JumpPlanner()
    return planner.plan(co)


# 测试函数
if __name__ == "__main__":
    # 测试用例
    test_co = {
        'objects': [
            {
                'item_type': 'main_dish',
                'content': 'Pollo Teriyaki',
                'quantity': 1,
                'modifiers': [],
                'confidence': 0.8
            }
        ],
        'intent': 'order',
        'conditions': [],
        'language': 'es',
        'confidence': 0.8,
        'raw_text': 'quiero Pollo Teriyaki'
    }
    
    result = plan(test_co)
    print("规划结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
