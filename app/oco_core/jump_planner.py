#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Jump Planner (优化版)
菜单项匹配与跳跃路径规划器 - 集成向量搜索、学习机制、性能监控
"""

import json
import re
import time
import hashlib
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict, deque
from functools import lru_cache
import math

from ..config import settings

@dataclass
class MenuMatch:
    """菜单匹配结果（增强版）"""
    item_id: str
    item_name: str
    variant_id: str
    variant_name: str
    category_name: str
    price: float
    sku: str
    match_score: float
    match_type: str  # 'exact', 'alias', 'keyword', 'fuzzy', 'semantic', 'learned'
    original_query: str
    semantic_score: float = 0.0
    learned_score: float = 0.0
    popularity_score: float = 0.0
    final_score: float = 0.0
    explanation: str = ""

@dataclass
class SearchIndex:
    """搜索索引结构"""
    exact_names: Dict[str, List[Dict]] = field(default_factory=dict)
    aliases: Dict[str, List[Dict]] = field(default_factory=dict)
    keywords: Dict[str, List[Dict]] = field(default_factory=dict)
    n_grams: Dict[str, List[Dict]] = field(default_factory=dict)
    all_items: List[Dict] = field(default_factory=list)
    item_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    category_embeddings: Dict[str, List[float]] = field(default_factory=dict)

@dataclass
class UserBehavior:
    """用户行为数据"""
    query_item_pairs: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    item_popularity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_queries: deque = field(default_factory=lambda: deque(maxlen=100))
    user_preferences: Dict[str, float] = field(default_factory=dict)

@dataclass
class PlanPath:
    """规划路径（增强版）"""
    matches: List[MenuMatch]
    total_score: float
    confidence: float
    requires_clarification: bool
    clarification_reason: str = ""
    semantic_coherence: float = 0.0
    diversity_score: float = 0.0

@dataclass
class SearchMetrics:
    """搜索性能指标"""
    total_searches: int = 0
    avg_search_time: float = 0.0
    cache_hit_rate: float = 0.0
    exact_match_rate: float = 0.0
    semantic_match_rate: float = 0.0
    user_satisfaction_score: float = 0.0

class SimpleEmbedding:
    """简单的文本嵌入器（可替换为更复杂的模型）"""
    
    def __init__(self):
        self.vocab = {}
        self.idf_scores = {}
        self.dimension = 100
        
    def _build_vocabulary(self, texts: List[str]):
        """构建词汇表"""
        word_counts = defaultdict(int)
        doc_counts = defaultdict(int)
        
        for text in texts:
            words = set(self._tokenize(text.lower()))
            for word in words:
                doc_counts[word] += 1
            
            for word in self._tokenize(text.lower()):
                word_counts[word] += 1
        
        # 计算 IDF 分数
        total_docs = len(texts)
        for word, doc_count in doc_counts.items():
            self.idf_scores[word] = math.log(total_docs / (doc_count + 1))
        
        # 构建词汇表索引
        self.vocab = {word: idx for idx, word in enumerate(word_counts.keys())}
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 移除标点符号并分词
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word.strip() for word in text.split() if word.strip()]
    
    @lru_cache(maxsize=1000)
    def encode(self, text: str) -> List[float]:
        """编码文本为向量"""
        if not self.vocab:
            return [0.0] * self.dimension
        
        words = self._tokenize(text.lower())
        vector = [0.0] * self.dimension
        
        # 简单的 TF-IDF 向量
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        for word, count in word_counts.items():
            if word in self.vocab:
                idx = self.vocab[word] % self.dimension
                tf = count / len(words)
                idf = self.idf_scores.get(word, 1.0)
                vector[idx] += tf * idf
        
        # 归一化
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class AdvancedMenuMatcher:
    """高级菜单匹配器"""
    
    def __init__(self):
        self.menu_data = self._load_menu_data()
        self.search_index = SearchIndex()
        self.embedding_model = SimpleEmbedding()
        self.user_behavior = UserBehavior()
        self.search_cache = {}
        self.metrics = SearchMetrics()
        self.dynamic_thresholds = {
            'exact_match': 1.0,
            'alias_match': 0.95,
            'keyword_match': 0.8,
            'fuzzy_match': 0.6,
            'semantic_match': 0.7,
            'learned_match': 0.85
        }
        
        self._build_enhanced_index()
        self._load_user_behavior()
    
    def _load_menu_data(self) -> Dict[str, Any]:
        """加载菜单数据"""
        try:
            kb_path = Path(settings.MENU_KB_FILE)
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load menu data: {e}")
        
        # 返回增强的默认菜单结构
        return {
            "menu_categories": {
                "combinaciones": {
                    "name": "Combinaciones",
                    "description": "Platos principales combinados",
                    "items": [
                        {
                            "item_id": "combo_pollo_teriyaki",
                            "item_name": "Pollo Teriyaki",
                            "variant_id": "default",
                            "variant_name": "Regular",
                            "price": 12.99,
                            "sku": "PT001",
                            "aliases": ["chicken teriyaki", "鸡肉照烧"],
                            "keywords": ["pollo", "chicken", "teriyaki", "鸡肉"],
                            "description": "Delicioso pollo en salsa teriyaki",
                            "ingredients": ["pollo", "salsa teriyaki", "arroz", "vegetales"]
                        }
                    ]
                },
                "adicionales": {
                    "name": "Adicionales",
                    "description": "Acompañamientos y extras",
                    "items": []
                }
            }
        }
    
    def _build_enhanced_index(self):
        """构建增强的搜索索引"""
        categories = self.menu_data.get("menu_categories", {})
        all_texts = []
        
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
                    'keywords': item.get('keywords', []),
                    'description': item.get('description', ''),
                    'ingredients': item.get('ingredients', [])
                }
                
                self.search_index.all_items.append(item_info)
                
                # 收集所有文本用于训练嵌入模型
                text_parts = [item_info['item_name']]
                text_parts.extend(item_info['aliases'])
                text_parts.extend(item_info['keywords'])
                text_parts.append(item_info['description'])
                text_parts.extend(item_info['ingredients'])
                
                all_texts.extend([text for text in text_parts if text])
                
                # 精确名称索引
                self._add_to_index(self.search_index.exact_names, item_info['item_name'].lower(), item_info)
                
                # 别名索引
                for alias in item_info['aliases']:
                    self._add_to_index(self.search_index.aliases, alias.lower(), item_info)
                
                # 关键词索引
                for keyword in item_info['keywords']:
                    self._add_to_index(self.search_index.keywords, keyword.lower(), item_info)
                
                # N-gram 索引
                self._build_ngram_index(item_info)
        
        # 训练嵌入模型
        if all_texts:
            self.embedding_model._build_vocabulary(all_texts)
            self._build_embedding_index()
    
    def _add_to_index(self, index: Dict, key: str, item: Dict):
        """添加到索引"""
        if key not in index:
            index[key] = []
        index[key].append(item)
    
    def _build_ngram_index(self, item_info: Dict):
        """构建 N-gram 索引"""
        text = f"{item_info['item_name']} {' '.join(item_info['aliases'])} {' '.join(item_info['keywords'])}"
        
        # 生成 2-gram 和 3-gram
        for n in [2, 3]:
            for ngram in self._generate_ngrams(text.lower(), n):
                self._add_to_index(self.search_index.n_grams, ngram, item_info)
    
    def _generate_ngrams(self, text: str, n: int) -> List[str]:
        """生成 N-gram"""
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    def _build_embedding_index(self):
        """构建嵌入索引"""
        for item in self.search_index.all_items:
            # 为每个菜单项创建嵌入
            item_text = f"{item['item_name']} {' '.join(item['aliases'])} {item['description']}"
            embedding = self.embedding_model.encode(item_text)
            self.search_index.item_embeddings[item['item_id']] = embedding
        
        # 为每个类别创建嵌入
        categories = self.menu_data.get("menu_categories", {})
        for category_key, category_data in categories.items():
            category_text = f"{category_data.get('name', '')} {category_data.get('description', '')}"
            embedding = self.embedding_model.encode(category_text)
            self.search_index.category_embeddings[category_key] = embedding
    
    def _load_user_behavior(self):
        """加载用户行为数据"""
        try:
            behavior_file = Path("user_behavior.json")
            if behavior_file.exists():
                with open(behavior_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_behavior.query_item_pairs = defaultdict(lambda: defaultdict(int), data.get('query_item_pairs', {}))
                    self.user_behavior.item_popularity = defaultdict(int, data.get('item_popularity', {}))
                    self.user_behavior.user_preferences = data.get('user_preferences', {})
        except Exception as e:
            print(f"Warning: Failed to load user behavior: {e}")
    
    def _save_user_behavior(self):
        """保存用户行为数据"""
        try:
            behavior_file = Path("user_behavior.json")
            data = {
                'query_item_pairs': dict(self.user_behavior.query_item_pairs),
                'item_popularity': dict(self.user_behavior.item_popularity),
                'user_preferences': self.user_behavior.user_preferences
            }
            with open(behavior_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save user behavior: {e}")
    
    def _update_dynamic_thresholds(self):
        """动态调整阈值"""
        # 基于搜索成功率调整阈值
        if self.metrics.exact_match_rate < 0.3:
            self.dynamic_thresholds['fuzzy_match'] = max(0.5, self.dynamic_thresholds['fuzzy_match'] - 0.05)
            self.dynamic_thresholds['semantic_match'] = max(0.6, self.dynamic_thresholds['semantic_match'] - 0.05)
        elif self.metrics.exact_match_rate > 0.7:
            self.dynamic_thresholds['fuzzy_match'] = min(0.7, self.dynamic_thresholds['fuzzy_match'] + 0.02)
            self.dynamic_thresholds['semantic_match'] = min(0.8, self.dynamic_thresholds['semantic_match'] + 0.02)
    
    def _get_cache_key(self, query: str, max_results: int) -> str:
        """生成缓存键"""
        return hashlib.md5(f"{query}_{max_results}".encode()).hexdigest()
    
    def _calculate_popularity_score(self, item_id: str) -> float:
        """计算流行度分数"""
        popularity = self.user_behavior.item_popularity.get(item_id, 0)
        max_popularity = max(self.user_behavior.item_popularity.values(), default=1)
        return popularity / max(max_popularity, 1)
    
    def _calculate_learned_score(self, query: str, item_id: str) -> float:
        """计算学习分数"""
        query_normalized = self._normalize_query(query)
        return self.user_behavior.query_item_pairs.get(query_normalized, {}).get(item_id, 0) / 10.0
    
    def _semantic_search(self, query: str, max_results: int = 5) -> List[MenuMatch]:
        """语义搜索"""
        query_embedding = self.embedding_model.encode(query)
        semantic_matches = []
        
        for item in self.search_index.all_items:
            item_embedding = self.search_index.item_embeddings.get(item['item_id'])
            if item_embedding:
                similarity = SimpleEmbedding.cosine_similarity(query_embedding, item_embedding)
                
                if similarity >= self.dynamic_thresholds['semantic_match']:
                    match = MenuMatch(
                        item_id=item['item_id'],
                        item_name=item['item_name'],
                        variant_id=item['variant_id'],
                        variant_name=item['variant_name'],
                        category_name=item['category_name'],
                        price=item['price'],
                        sku=item['sku'],
                        match_score=similarity,
                        semantic_score=similarity,
                        match_type='semantic',
                        original_query=query,
                        explanation=f"Semantic similarity: {similarity:.3f}"
                    )
                    semantic_matches.append(match)
        
        return sorted(semantic_matches, key=lambda x: x.semantic_score, reverse=True)[:max_results]
    
    def _fuzzy_similarity(self, text1: str, text2: str) -> float:
        """改进的模糊相似度计算"""
        # 使用多种相似度算法的组合
        seq_ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Jaccard 相似度
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard = intersection / union if union > 0 else 0
        
        # 最长公共子序列相似度
        lcs_len = self._lcs_length(text1.lower(), text2.lower())
        lcs_ratio = 2 * lcs_len / (len(text1) + len(text2)) if (len(text1) + len(text2)) > 0 else 0
        
        # 加权组合
        return 0.5 * seq_ratio + 0.3 * jaccard + 0.2 * lcs_ratio
    
    def _lcs_length(self, text1: str, text2: str) -> int:
        """计算最长公共子序列长度"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _normalize_query(self, query: str) -> str:
        """增强的查询标准化"""
        # 移除多余空格
        query = re.sub(r'\s+', ' ', query.strip())
        
        # 移除数量词和修饰词
        patterns_to_remove = [
            r'^\d+\s*(?:个|份|块|件|piezas?|presas?|combos?|pieces?)\s*',
            r'\s*\d+\s*(?:个|份|块|件|piezas?|presas?|combos?|pieces?)\s*$',
            r'^(?:我要|给我|来|点|quiero|dame|want|order|get|necesito)\s*',
            r'\s*(?:请|por favor|please)\s*$'
        ]
        
        for pattern in patterns_to_remove:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        return query.strip()
    
    def search_menu_items(self, query: str, max_results: int = 5, use_learning: bool = True) -> List[MenuMatch]:
        """增强的菜单项搜索"""
        start_time = time.time()
        self.metrics.total_searches += 1
        
        if not query or not query.strip():
            return []
        
        # 检查缓存
        cache_key = self._get_cache_key(query, max_results)
        if cache_key in self.search_cache:
            search_time = time.time() - start_time
            self._update_metrics(search_time, True)
            return self.search_cache[cache_key]
        
        normalized_query = self._normalize_query(query)
        query_lower = normalized_query.lower()
        matches = []
        
        # 1. 精确匹配
        exact_matches = self._exact_search(query_lower)
        matches.extend(exact_matches)
        
        # 2. 别名匹配
        alias_matches = self._alias_search(query_lower)
        matches.extend([m for m in alias_matches if not any(em.item_id == m.item_id for em in matches)])
        
        # 3. 关键词匹配
        keyword_matches = self._keyword_search(query_lower)
        matches.extend([m for m in keyword_matches if not any(em.item_id == m.item_id for em in matches)])
        
        # 4. N-gram 匹配
        ngram_matches = self._ngram_search(query_lower)
        matches.extend([m for m in ngram_matches if not any(em.item_id == m.item_id for em in matches)])
        
        # 5. 语义搜索
        semantic_matches = self._semantic_search(query)
        matches.extend([m for m in semantic_matches if not any(em.item_id == m.item_id for em in matches)])
        
        # 6. 模糊匹配
        fuzzy_matches = self._fuzzy_search(query_lower)
        matches.extend([m for m in fuzzy_matches if not any(em.item_id == m.item_id for em in matches)])
        
        # 7. 学习增强
        if use_learning:
            for match in matches:
                match.learned_score = self._calculate_learned_score(query, match.item_id)
                match.popularity_score = self._calculate_popularity_score(match.item_id)
        
        # 计算最终分数
        for match in matches:
            match.final_score = self._calculate_final_score(match)
        
        # 排序并限制结果
        matches.sort(key=lambda x: x.final_score, reverse=True)
        final_matches = matches[:max_results]
        
        # 缓存结果
        if len(self.search_cache) < 1000:
            self.search_cache[cache_key] = final_matches
        
        search_time = time.time() - start_time
        self._update_metrics(search_time, False)
        
        return final_matches
    
    def _exact_search(self, query: str) -> List[MenuMatch]:
        """精确搜索"""
        matches = []
        if query in self.search_index.exact_names:
            for item in self.search_index.exact_names[query]:
                match = MenuMatch(
                    item_id=item['item_id'],
                    item_name=item['item_name'],
                    variant_id=item['variant_id'],
                    variant_name=item['variant_name'],
                    category_name=item['category_name'],
                    price=item['price'],
                    sku=item['sku'],
                    match_score=self.dynamic_thresholds['exact_match'],
                    match_type='exact',
                    original_query=query,
                    explanation="Exact name match"
                )
                matches.append(match)
        return matches
    
    def _alias_search(self, query: str) -> List[MenuMatch]:
        """别名搜索"""
        matches = []
        if query in self.search_index.aliases:
            for item in self.search_index.aliases[query]:
                match = MenuMatch(
                    item_id=item['item_id'],
                    item_name=item['item_name'],
                    variant_id=item['variant_id'],
                    variant_name=item['variant_name'],
                    category_name=item['category_name'],
                    price=item['price'],
                    sku=item['sku'],
                    match_score=self.dynamic_thresholds['alias_match'],
                    match_type='alias',
                    original_query=query,
                    explanation="Alias match"
                )
                matches.append(match)
        return matches
    
    def _keyword_search(self, query: str) -> List[MenuMatch]:
        """关键词搜索"""
        matches = []
        query_words = set(query.split())
        
        for keyword, items in self.search_index.keywords.items():
            if keyword in query:
                for item in items:
                    # 计算关键词覆盖度
                    keyword_words = set(keyword.split())
                    coverage = len(keyword_words.intersection(query_words)) / len(query_words)
                    match_score = self.dynamic_thresholds['keyword_match'] * coverage
                    
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
                        original_query=query,
                        explanation=f"Keyword match: {keyword} (coverage: {coverage:.2f})"
                    )
                    matches.append(match)
        
        return matches
    
    def _ngram_search(self, query: str) -> List[MenuMatch]:
        """N-gram 搜索"""
        matches = []
        
        # 生成查询的 N-gram
        for n in [2, 3]:
            query_ngrams = self._generate_ngrams(query, n)
            for ngram in query_ngrams:
                if ngram in self.search_index.n_grams:
                    for item in self.search_index.n_grams[ngram]:
                        match_score = self.dynamic_thresholds['keyword_match'] * 0.8  # N-gram 匹配稍低分
                        
                        match = MenuMatch(
                            item_id=item['item_id'],
                            item_name=item['item_name'],
                            variant_id=item['variant_id'],
                            variant_name=item['variant_name'],
                            category_name=item['category_name'],
                            price=item['price'],
                            sku=item['sku'],
                            match_score=match_score,
                            match_type='ngram',
                            original_query=query,
                            explanation=f"{n}-gram match: {ngram}"
                        )
                        matches.append(match)
        
        return matches
    
    def _fuzzy_search(self, query: str) -> List[MenuMatch]:
        """模糊搜索"""
        matches = []
        
        for item in self.search_index.all_items:
            # 对项目名称进行模糊匹配
            name_similarity = self._fuzzy_similarity(query, item['item_name'])
            
            # 对别名进行模糊匹配
            alias_similarity = 0.0
            for alias in item['aliases']:
                alias_sim = self._fuzzy_similarity(query, alias)
                alias_similarity = max(alias_similarity, alias_sim)
            
            # 对关键词进行模糊匹配
            keyword_similarity = 0.0
            for keyword in item['keywords']:
                keyword_sim = self._fuzzy_similarity(query, keyword)
                keyword_similarity = max(keyword_similarity, keyword_sim)
            
            best_similarity = max(name_similarity, alias_similarity, keyword_similarity)
            
            if best_similarity >= self.dynamic_thresholds['fuzzy_match']:
                explanation_type = "name" if best_similarity == name_similarity else ("alias" if best_similarity == alias_similarity else "keyword")
                
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
                    original_query=query,
                    explanation=f"Fuzzy {explanation_type} match: {best_similarity:.3f}"
                )
                matches.append(match)
        
        return matches
    
    def _calculate_final_score(self, match: MenuMatch) -> float:
        """计算最终分数"""
        # 基础分数
        base_score = match.match_score
        
        # 语义分数加成
        semantic_bonus = match.semantic_score * 0.1
        
        # 学习分数加成
        learned_bonus = match.learned_score * 0.15
        
        # 流行度加成
        popularity_bonus = match.popularity_score * 0.05
        
        # 匹配类型权重
        type_weights = {
            'exact': 1.0,
            'alias': 0.95,
            'keyword': 0.8,
            'ngram': 0.75,
            'semantic': 0.85,
            'fuzzy': 0.7,
            'learned': 0.9
        }
        
        type_weight = type_weights.get(match.match_type, 0.5)
        
        final_score = (base_score * type_weight) + semantic_bonus + learned_bonus + popularity_bonus
        return min(1.0, final_score)
    
    def _update_metrics(self, search_time: float, cache_hit: bool):
        """更新搜索指标"""
        # 更新平均搜索时间
        total_time = self.metrics.avg_search_time * (self.metrics.total_searches - 1) + search_time
        self.metrics.avg_search_time = total_time / self.metrics.total_searches
        
        # 更新缓存命中率
        cache_hits = self.metrics.cache_hit_rate * (self.metrics.total_searches - 1)
        if cache_hit:
            cache_hits += 1
        self.metrics.cache_hit_rate = cache_hits / self.metrics.total_searches
    
    def record_user_interaction(self, query: str, selected_item_id: str, satisfaction_score: float = 1.0):
        """记录用户交互"""
        normalized_query = self._normalize_query(query)
        
        # 更新查询-项目对
        self.user_behavior.query_item_pairs[normalized_query][selected_item_id] += 1
        
        # 更新项目流行度
        self.user_behavior.item_popularity[selected_item_id] += 1
        
        # 记录最近查询
        self.user_behavior.recent_queries.append({
            'query': normalized_query,
            'item_id': selected_item_id,
            'timestamp': time.time(),
            'satisfaction': satisfaction_score
        })
        
        # 更新用户满意度
        total_satisfaction = self.metrics.user_satisfaction_score * (self.metrics.total_searches - 1) + satisfaction_score
        self.metrics.user_satisfaction_score = total_satisfaction / self.metrics.total_searches
        
        # 定期保存
        if len(self.user_behavior.recent_queries) % 10 == 0:
            self._save_user_behavior()
            self._update_dynamic_thresholds()
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """获取搜索分析数据"""
        return {
            'metrics': {
                'total_searches': self.metrics.total_searches,
                'avg_search_time': round(self.metrics.avg_search_time, 4),
                'cache_hit_rate': round(self.metrics.cache_hit_rate * 100, 2),
                'user_satisfaction': round(self.metrics.user_satisfaction_score, 3)
            },
            'dynamic_thresholds': self.dynamic_thresholds,
            'popular_items': dict(sorted(
                self.user_behavior.item_popularity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'cache_size': len(self.search_cache),
            'index_stats': {
                'total_items': len(self.search_index.all_items),
                'exact_names': len(self.search_index.exact_names),
                'aliases': len(self.search_index.aliases),
                'keywords': len(self.search_index.keywords),
                'ngrams': len(self.search_index.n_grams)
            }
        }

class IntelligentJumpPlanner:
    """智能跳跃路径规划器"""
    
    def __init__(self):
        self.matcher = AdvancedMenuMatcher()
        self.base_clarification_threshold = 0.7
        self.multiple_match_threshold = 3
        self.diversity_weight = 0.1
        self.coherence_weight = 0.15
    
    def plan(self, co: Dict[str, Any], session_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        智能路径规划
        
        Args:
            co: 对话对象字典
            session_context: 会话上下文
            
        Returns:
            规划结果字典
        """
        objects = co.get('objects', [])
        intent = co.get('intent', 'inquiry')
        confidence = co.get('confidence', 0.0)
        
        if not objects or intent not in ['order', 'modify']:
            return self._create_empty_plan('no_valid_items')
        
        # 动态调整阈值
        clarification_threshold = self._calculate_dynamic_threshold(confidence, session_context)
        
        all_paths = []
        
        # 为每个对象寻找匹配
        for obj in objects:
            if obj['item_type'] in ['main_dish']:
                content = obj['content']
                quantity = obj.get('quantity', 1)
                
                # 使用增强搜索
                matches = self.matcher.search_menu_items(content, max_results=8)
                
                if matches:
                    # 为每个匹配创建路径
                    for match in matches:
                        # 调整匹配分数基于数量合理性
                        quantity_penalty = self._calculate_quantity_penalty(quantity)
                        adjusted_score = match.final_score * (1 - quantity_penalty)
                        
                        path = PlanPath(
                            matches=[match],
                            total_score=adjusted_score,
                            confidence=adjusted_score * confidence,
                            requires_clarification=False
                        )
                        all_paths.append(path)
        
        if not all_paths:
            return self._create_empty_plan('no_menu_matches')
        
        # 计算语义连贯性和多样性
        for path in all_paths:
            path.semantic_coherence = self._calculate_semantic_coherence(path, co)
            path.diversity_score = self._calculate_diversity_score(path, session_context)
        
        # 选择最佳路径
        best_path = self._select_best_path(all_paths)
        
        # 检查是否需要澄清
        clarification_result = self._analyze_clarification_need(
            all_paths, best_path, clarification_threshold
        )
        
        best_path.requires_clarification = clarification_result['required']
        best_path.clarification_reason = clarification_result['reason']
        
        return self._format_plan_result(best_path, all_paths, co)
    
    def _calculate_dynamic_threshold(self, confidence: float, session_context: Optional[Dict]) -> float:
        """动态计算澄清阈值"""
        base_threshold = self.base_clarification_threshold
        
        # 基于对话置信度调整
        confidence_adjustment = (confidence - 0.5) * 0.2
        
        # 基于会话历史调整
        session_adjustment = 0.0
        if session_context:
            successful_orders = session_context.get('successful_orders', 0)
            total_attempts = session_context.get('total_attempts', 1)
            success_rate = successful_orders / max(total_attempts, 1)
            
            # 成功率高的用户降低阈值
            session_adjustment = (success_rate - 0.5) * 0.1
        
        return max(0.4, min(0.9, base_threshold + confidence_adjustment + session_adjustment))
    
    def _calculate_quantity_penalty(self, quantity: int) -> float:
        """计算数量合理性惩罚"""
        if quantity <= 0:
            return 0.5
        elif quantity > 20:  # 异常大的数量
            return 0.3
        elif quantity > 10:
            return 0.1
        else:
            return 0.0
    
    def _calculate_semantic_coherence(self, path: PlanPath, co: Dict) -> float:
        """计算语义连贯性"""
        # 检查路径中的匹配是否与原始查询语义连贯
        raw_text = co.get('raw_text', '')
        if not raw_text or not path.matches:
            return 0.5
        
        # 使用嵌入模型计算连贯性
        query_embedding = self.matcher.embedding_model.encode(raw_text)
        
        coherence_scores = []
        for match in path.matches:
            item_embedding = self.matcher.search_index.item_embeddings.get(match.item_id)
            if item_embedding:
                similarity = SimpleEmbedding.cosine_similarity(query_embedding, item_embedding)
                coherence_scores.append(similarity)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_diversity_score(self, path: PlanPath, session_context: Optional[Dict]) -> float:
        """计算多样性分数"""
        if not session_context or not path.matches:
            return 0.5
        
        recent_items = session_context.get('recent_items', [])
        if not recent_items:
            return 1.0  # 新用户给满分
        
        # 检查是否与最近订单重复
        current_items = {match.item_id for match in path.matches}
        recent_items_set = set(recent_items[-10:])  # 最近10个项目
        
        overlap = len(current_items.intersection(recent_items_set))
        total_current = len(current_items)
        
        diversity = 1.0 - (overlap / max(total_current, 1))
        return diversity
    
    def _select_best_path(self, paths: List[PlanPath]) -> PlanPath:
        """选择最佳路径"""
        if not paths:
            return PlanPath(matches=[], total_score=0.0, confidence=0.0, requires_clarification=True)
        
        # 计算综合分数
        for path in paths:
            comprehensive_score = (
                path.confidence * 0.7 +
                path.semantic_coherence * self.coherence_weight +
                path.diversity_score * self.diversity_weight
            )
            path.total_score = comprehensive_score
        
        return max(paths, key=lambda p: p.total_score)
    
    def _analyze_clarification_need(self, all_paths: List[PlanPath], best_path: PlanPath, 
                                  threshold: float) -> Dict[str, Any]:
        """分析是否需要澄清"""
        
        # 检查最佳路径分数
        if best_path.confidence < threshold:
            return {
                'required': True,
                'reason': 'low_confidence',
                'details': f'Best match confidence {best_path.confidence:.3f} below threshold {threshold:.3f}'
            }
        
        # 检查多个高分匹配
        high_score_paths = [p for p in all_paths if p.confidence >= threshold * 0.9]
        if len(high_score_paths) >= self.multiple_match_threshold:
            score_diff = max(p.confidence for p in high_score_paths) - min(p.confidence for p in high_score_paths)
            if score_diff < 0.1:  # 分数过于接近
                return {
                    'required': True,
                    'reason': 'multiple_similar_matches',
                    'details': f'{len(high_score_paths)} matches with similar scores'
                }
        
        # 检查语义连贯性
        if best_path.semantic_coherence < 0.6:
            return {
                'required': True,
                'reason': 'low_semantic_coherence',
                'details': f'Semantic coherence {best_path.semantic_coherence:.3f} too low'
            }
        
        return {'required': False, 'reason': 'confident_match'}
    
    def _create_empty_plan(self, reason: str) -> Dict[str, Any]:
        """创建空规划结果"""
        return {
            'path': [],
            'score': 0.0,
            'confidence': 0.0,
            'requires_clarification': True,
            'clarification_reason': reason,
            'alternative_paths': [],
            'analytics': None
        }
    
    def _format_plan_result(self, best_path: PlanPath, all_paths: List[PlanPath], 
                          co: Dict) -> Dict[str, Any]:
        """格式化规划结果"""
        
        # 准备主路径
        formatted_matches = []
        for match in best_path.matches:
            formatted_match = {
                'item_id': match.item_id,
                'item_name': match.item_name,
                'variant_id': match.variant_id,
                'variant_name': match.variant_name,
                'category_name': match.category_name,
                'price': match.price,
                'sku': match.sku,
                'match_score': round(match.match_score, 3),
                'final_score': round(match.final_score, 3),
                'match_type': match.match_type,
                'original_query': match.original_query,
                'explanation': match.explanation
            }
            
            # 添加额外的分数信息
            if match.semantic_score > 0:
                formatted_match['semantic_score'] = round(match.semantic_score, 3)
            if match.learned_score > 0:
                formatted_match['learned_score'] = round(match.learned_score, 3)
            if match.popularity_score > 0:
                formatted_match['popularity_score'] = round(match.popularity_score, 3)
                
            formatted_matches.append(formatted_match)
        
        # 准备备选路径
        alternative_paths = []
        sorted_paths = sorted(all_paths, key=lambda p: p.total_score, reverse=True)
        
        for path in sorted_paths[:5]:  # 最多5个备选
            if path != best_path and path.matches:
                alternative = {
                    'matches': [
                        {
                            'item_id': match.item_id,
                            'item_name': match.item_name,
                            'price': match.price,
                            'final_score': round(match.final_score, 3),
                            'match_type': match.match_type
                        }
                        for match in path.matches
                    ],
                    'total_score': round(path.total_score, 3),
                    'confidence': round(path.confidence, 3),
                    'semantic_coherence': round(path.semantic_coherence, 3),
                    'diversity_score': round(path.diversity_score, 3)
                }
                alternative_paths.append(alternative)
        
        return {
            'path': formatted_matches,
            'score': round(best_path.total_score, 3),
            'confidence': round(best_path.confidence, 3),
            'semantic_coherence': round(best_path.semantic_coherence, 3),
            'diversity_score': round(best_path.diversity_score, 3),
            'requires_clarification': best_path.requires_clarification,
            'clarification_reason': best_path.clarification_reason,
            'alternative_paths': alternative_paths,
            'analytics': self.matcher.get_search_analytics() if len(all_paths) % 50 == 0 else None  # 定期提供分析
        }
    
    def record_user_feedback(self, query: str, selected_item_id: str, satisfaction_score: float):
        """记录用户反馈"""
        self.matcher.record_user_interaction(query, selected_item_id, satisfaction_score)


def plan(co: Dict[str, Any], session_context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    外部调用接口 - 智能路径规划
    
    Args:
        co: 对话对象字典
        session_context: 会话上下文
        
    Returns:
        规划结果字典
    """
    planner = IntelligentJumpPlanner()
    return planner.plan(co, session_context)


def record_feedback(query: str, selected_item_id: str, satisfaction_score: float = 1.0):
    """
    记录用户反馈 - 外部接口
    
    Args:
        query: 原始查询
        selected_item_id: 用户选择的项目ID
        satisfaction_score: 满意度分数 (0.0-1.0)
    """
    planner = IntelligentJumpPlanner()
    planner.record_user_feedback(query, selected_item_id, satisfaction_score)


def get_analytics() -> Dict[str, Any]:
    """获取分析数据 - 外部接口"""
    matcher = AdvancedMenuMatcher()
    return matcher.get_search_analytics()


def benchmark_search_performance(test_queries: List[str], iterations: int = 100) -> Dict[str, Any]:
    """搜索性能基准测试"""
    matcher = AdvancedMenuMatcher()
    
    start_time = time.time()
    results = []
    
    for _ in range(iterations):
        for query in test_queries:
            query_start = time.time()
            matches = matcher.search_menu_items(query)
            query_time = time.time() - query_start
            
            results.append({
                'query': query,
                'matches_count': len(matches),
                'best_score': matches[0].final_score if matches else 0.0,
                'search_time': query_time,
                'has_exact_match': any(m.match_type == 'exact' for m in matches),
                'has_semantic_match': any(m.match_type == 'semantic' for m in matches)
            })
    
    total_time = time.time() - start_time
    
    # 统计分析
    avg_search_time = sum(r['search_time'] for r in results) / len(results)
    avg_matches = sum(r['matches_count'] for r in results) / len(results)
    avg_best_score = sum(r['best_score'] for r in results) / len(results)
    exact_match_rate = sum(1 for r in results if r['has_exact_match']) / len(results)
    semantic_match_rate = sum(1 for r in results if r['has_semantic_match']) / len(results)
    
    return {
        'total_time': round(total_time, 3),
        'total_queries': len(results),
        'queries_per_second': round(len(results) / total_time, 2),
        'avg_search_time': round(avg_search_time, 4),
        'avg_matches_per_query': round(avg_matches, 2),
        'avg_best_score': round(avg_best_score, 3),
        'exact_match_rate': round(exact_match_rate * 100, 1),
        'semantic_match_rate': round(semantic_match_rate * 100, 1),
        'search_analytics': matcher.get_search_analytics()
    }


# 综合测试函数
def run_comprehensive_tests():
    """运行综合测试"""
    
    print("=== 优化版 Jump Planner 综合测试 ===\n")
    
    # 测试用例
    test_cases = [
        {
            'objects': [{'item_type': 'main_dish', 'content': 'Pollo Teriyaki', 'quantity': 1}],
            'intent': 'order',
            'confidence': 0.8,
            'raw_text': 'quiero Pollo Teriyaki'
        },
        {
            'objects': [{'item_type': 'main_dish', 'content': 'chicken rice', 'quantity': 2}],
            'intent': 'order',
            'confidence': 0.6,
            'raw_text': 'I want 2 chicken rice'
        },
        {
            'objects': [{'item_type': 'main_dish', 'content': '鸡肉套餐', 'quantity': 1}],
            'intent': 'order',
            'confidence': 0.7,
            'raw_text': '我要一个鸡肉套餐'
        }
    ]
    
    # 1. 基本规划测试
    print("1. 基本路径规划测试:")
    for i, test_co in enumerate(test_cases):
        print(f"\n   测试 {i+1}: {test_co['raw_text']}")
        result = plan(test_co)
        
        print(f"   规划成功: {'是' if result['path'] else '否'}")
        print(f"   置信度: {result['confidence']}")
        print(f"   需要澄清: {'是' if result['requires_clarification'] else '否'}")
        
        if result['path']:
            best_match = result['path'][0]
            print(f"   最佳匹配: {best_match['item_name']} (分数: {best_match['final_score']})")
            print(f"   匹配类型: {best_match['match_type']}")
        
        if result['alternative_paths']:
            print(f"   备选方案: {len(result['alternative_paths'])} 个")
    
    # 2. 性能基准测试
    print("\n2. 性能基准测试:")
    benchmark_queries = [
        "Pollo Teriyaki", "chicken rice", "beef combo", "鸡肉套餐",
        "carne con broccoli", "arroz con pollo", "combo familiar"
    ]
    
    benchmark_result = benchmark_search_performance(benchmark_queries, iterations=20)
    for key, value in benchmark_result.items():
        if key != 'search_analytics':
            print(f"   {key}: {value}")
    
    # 3. 学习机制测试
    print("\n3. 学习机制测试:")
    
    # 模拟用户反馈
    feedback_data = [
        ("pollo", "combo_pollo_teriyaki", 0.9),
        ("chicken", "combo_pollo_teriyaki", 0.8),
        ("鸡肉", "combo_pollo_teriyaki", 0.85)
    ]
    
    print("   记录用户反馈...")
    for query, item_id, satisfaction in feedback_data:
        record_feedback(query, item_id, satisfaction)
    
    # 测试学习效果
    print("   测试学习效果:")
    learning_test_co = {
        'objects': [{'item_type': 'main_dish', 'content': 'pollo', 'quantity': 1}],
        'intent': 'order',
        'confidence': 0.6,
        'raw_text': 'quiero pollo'
    }
    
    result_after_learning = plan(learning_test_co)
    if result_after_learning['path']:
        print(f"   学习后最佳匹配: {result_after_learning['path'][0]['item_name']}")
        print(f"   学习加成: {result_after_learning['path'][0].get('learned_score', 0)}")
    
    # 4. 分析数据
    print("\n4. 系统分析:")
    analytics = get_analytics()
    
    metrics = analytics.get('metrics', {})
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    print(f"   动态阈值: {analytics.get('dynamic_thresholds', {})}")
    print(f"   索引统计: {analytics.get('index_stats', {})}")


if __name__ == "__main__":
    run_comprehensive_tests()
