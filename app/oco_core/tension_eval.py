#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Tension Evaluation
张力评估与自学习系统
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import os

@dataclass
class TensionRecord:
    """张力记录"""
    timestamp: float
    co_confidence: float
    path_score: float
    clarification_count: int
    order_success: bool
    user_satisfaction: Optional[float] = None
    resolution_time: Optional[float] = None
    error_type: Optional[str] = None

@dataclass
class TensionMetrics:
    """张力指标"""
    avg_confidence: float
    success_rate: float
    avg_clarification_count: float
    avg_resolution_time: float
    tension_score: float

class TensionTracker:
    """张力跟踪器"""
    
    def __init__(self, data_dir: str = "/mnt/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.tension_file = self.data_dir / "tension_history.json"
        self.metrics_file = self.data_dir / "tension_metrics.json"
        self.lock = threading.Lock()
        
        # 加载历史数据
        self.history = self._load_history()
        self.current_session = {}
    
    def _load_history(self) -> List[TensionRecord]:
        """加载历史张力记录"""
        try:
            if self.tension_file.exists():
                with open(self.tension_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [TensionRecord(**record) for record in data]
        except Exception as e:
            print(f"Warning: Failed to load tension history: {e}")
        return []
    
    def _save_history(self):
        """保存历史记录"""
        try:
            with self.lock:
                data = [asdict(record) for record in self.history]
                with open(self.tension_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving tension history: {e}")
    
    def start_session(self, session_id: str, co_confidence: float, path_score: float):
        """开始新的会话跟踪"""
        self.current_session[session_id] = {
            'start_time': time.time(),
            'co_confidence': co_confidence,
            'path_score': path_score,
            'clarification_count': 0,
            'order_success': False,
            'errors': []
        }
    
    def add_clarification(self, session_id: str):
        """添加澄清计数"""
        if session_id in self.current_session:
            self.current_session[session_id]['clarification_count'] += 1
    
    def add_error(self, session_id: str, error_type: str):
        """添加错误记录"""
        if session_id in self.current_session:
            self.current_session[session_id]['errors'].append(error_type)
    
    def complete_session(self, session_id: str, order_success: bool, user_satisfaction: Optional[float] = None):
        """完成会话并记录"""
        if session_id not in self.current_session:
            return
        
        session = self.current_session[session_id]
        resolution_time = time.time() - session['start_time']
        
        record = TensionRecord(
            timestamp=time.time(),
            co_confidence=session['co_confidence'],
            path_score=session['path_score'],
            clarification_count=session['clarification_count'],
            order_success=order_success,
            user_satisfaction=user_satisfaction,
            resolution_time=resolution_time,
            error_type=session['errors'][0] if session['errors'] else None
        )
        
        self.history.append(record)
        del self.current_session[session_id]
        
        # 保存并更新指标
        self._save_history()
        self._update_metrics()
    
    def _update_metrics(self):
        """更新张力指标"""
        if not self.history:
            return
        
        # 获取最近的记录进行分析
        recent_records = self.history[-100:]  # 最近100条记录
        
        avg_confidence = sum(r.co_confidence for r in recent_records) / len(recent_records)
        success_rate = sum(1 for r in recent_records if r.order_success) / len(recent_records)
        avg_clarification = sum(r.clarification_count for r in recent_records) / len(recent_records)
        
        valid_times = [r.resolution_time for r in recent_records if r.resolution_time]
        avg_resolution_time = sum(valid_times) / len(valid_times) if valid_times else 0
        
        # 计算综合张力分数
        tension_score = self._calculate_tension_score(
            avg_confidence, success_rate, avg_clarification, avg_resolution_time
        )
        
        metrics = TensionMetrics(
            avg_confidence=avg_confidence,
            success_rate=success_rate,
            avg_clarification_count=avg_clarification,
            avg_resolution_time=avg_resolution_time,
            tension_score=tension_score
        )
        
        # 保存指标
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def _calculate_tension_score(self, confidence: float, success_rate: float, 
                                clarification_count: float, resolution_time: float) -> float:
        """计算综合张力分数"""
        # 基础分数来自置信度和成功率
        base_score = (confidence + success_rate) / 2
        
        # 澄清次数惩罚
        clarification_penalty = min(0.3, clarification_count * 0.1)
        
        # 响应时间惩罚（超过60秒开始惩罚）
        time_penalty = max(0, (resolution_time - 60) / 300) * 0.2
        
        # 最终分数
        final_score = max(0, base_score - clarification_penalty - time_penalty)
        return min(1.0, final_score)
    
    def get_current_metrics(self) -> TensionMetrics:
        """获取当前指标"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return TensionMetrics(**data)
        except Exception:
            pass
        
        # 返回默认指标
        return TensionMetrics(
            avg_confidence=0.5,
            success_rate=0.5,
            avg_clarification_count=1.0,
            avg_resolution_time=30.0,
            tension_score=0.5
        )

class AdaptiveTensionEvaluator:
    """自适应张力评估器"""
    
    def __init__(self):
        self.tracker = TensionTracker()
        
        # 动态阈值
        self.base_thresholds = {
            'clarify_threshold': 0.4,    # 需要澄清的阈值
            'execute_threshold': 0.8,    # 直接执行的阈值
            'confidence_threshold': 0.6   # 置信度阈值
        }
        
    def evaluate_path(self, path_data: Dict[str, Any]) -> float:
        """
        评估路径张力分数
        
        Args:
            path_data: 路径数据字典
            
        Returns:
            张力分数 (0-1)
        """
        if not path_data or not path_data.get('path'):
            return 0.0
        
        base_score = path_data.get('score', 0.0)
        confidence = path_data.get('confidence', 0.0)
        requires_clarification = path_data.get('requires_clarification', False)
        
        # 获取当前系统指标
        metrics = self.tracker.get_current_metrics()
        
        # 基于历史表现调整分数
        historical_adjustment = self._calculate_historical_adjustment(metrics)
        
        # 基础张力分数
        tension_score = (base_score + confidence) / 2
        
        # 应用历史调整
        adjusted_score = tension_score * historical_adjustment
        
        # 澄清需求惩罚
        if requires_clarification:
            adjusted_score *= 0.7
        
        # 动态阈值调整
        dynamic_thresholds = self._get_dynamic_thresholds(metrics)
        
        # 根据动态阈值重新评估
        if adjusted_score < dynamic_thresholds['clarify_threshold']:
            adjusted_score *= 0.8  # 进一步降低，倾向于澄清
        elif adjusted_score > dynamic_thresholds['execute_threshold']:
            adjusted_score = min(1.0, adjusted_score * 1.1)  # 提升，倾向于执行
        
        return round(min(1.0, max(0.0, adjusted_score)), 3)
    
    def _calculate_historical_adjustment(self, metrics: TensionMetrics) -> float:
        """基于历史指标计算调整因子"""
        # 成功率调整
        success_adjustment = 0.8 + (metrics.success_rate * 0.4)
        
        # 澄清频率调整
        clarify_adjustment = max(0.7, 1.2 - (metrics.avg_clarification_count * 0.2))
        
        # 置信度调整
        confidence_adjustment = 0.8 + (metrics.avg_confidence * 0.4)
        
        # 综合调整因子
        return (success_adjustment + clarify_adjustment + confidence_adjustment) / 3
    
    def _get_dynamic_thresholds(self, metrics: TensionMetrics) -> Dict[str, float]:
        """获取动态阈值"""
        thresholds = self.base_thresholds.copy()
        
        # 如果成功率低，提高澄清阈值（更容易澄清）
        if metrics.success_rate < 0.6:
            thresholds['clarify_threshold'] += 0.1
            thresholds['execute_threshold'] += 0.1
        
        # 如果澄清次数多，降低澄清阈值（减少澄清）
        if metrics.avg_clarification_count > 1.5:
            thresholds['clarify_threshold'] -= 0.1
        
        # 如果平均置信度低，调整阈值
        if metrics.avg_confidence < 0.5:
            thresholds['confidence_threshold'] += 0.1
        
        return thresholds
    
    def should_clarify(self, tension_score: float) -> bool:
        """判断是否需要澄清"""
        metrics = self.tracker.get_current_metrics()
        thresholds = self._get_dynamic_thresholds(metrics)
        return tension_score < thresholds['clarify_threshold']
    
    def should_execute(self, tension_score: float) -> bool:
        """判断是否可以直接执行"""
        metrics = self.tracker.get_current_metrics()
        thresholds = self._get_dynamic_thresholds(metrics)
        return tension_score > thresholds['execute_threshold']
    
    def get_action_recommendation(self, tension_score: float) -> str:
        """获取动作推荐"""
        if self.should_clarify(tension_score):
            return "clarify"
        elif self.should_execute(tension_score):
            return "execute"
        else:
            return "analyze"

# 全局评估器实例
_global_evaluator = AdaptiveTensionEvaluator()

def score(path_data: Dict[str, Any]) -> float:
    """
    外部调用接口 - 评估路径张力分数
    
    Args:
        path_data: 路径数据字典
        
    Returns:
        张力分数 (0-1)
    """
    return _global_evaluator.evaluate_path(path_data)

def should_clarify(tension_score: float) -> bool:
    """外部接口 - 判断是否需要澄清"""
    return _global_evaluator.should_clarify(tension_score)

def should_execute(tension_score: float) -> bool:
    """外部接口 - 判断是否可以直接执行"""
    return _global_evaluator.should_execute(tension_score)

def get_action_recommendation(tension_score: float) -> str:
    """外部接口 - 获取动作推荐"""
    return _global_evaluator.get_action_recommendation(tension_score)

def start_session_tracking(session_id: str, co_confidence: float, path_score: float):
    """开始会话跟踪"""
    _global_evaluator.tracker.start_session(session_id, co_confidence, path_score)

def add_clarification(session_id: str):
    """添加澄清记录"""
    _global_evaluator.tracker.add_clarification(session_id)

def complete_session(session_id: str, order_success: bool, user_satisfaction: Optional[float] = None):
    """完成会话"""
    _global_evaluator.tracker.complete_session(session_id, order_success, user_satisfaction)

def get_system_metrics() -> Dict[str, Any]:
    """获取系统指标"""
    metrics = _global_evaluator.tracker.get_current_metrics()
    return asdict(metrics)


# 测试函数
if __name__ == "__main__":
    # 测试张力评估
    test_path = {
        'path': [{'item_name': 'Pollo Teriyaki', 'price': 11.99}],
        'score': 0.85,
        'confidence': 0.8,
        'requires_clarification': False
    }
    
    tension_score = score(test_path)
    print(f"张力分数: {tension_score}")
    print(f"动作推荐: {get_action_recommendation(tension_score)}")
    print(f"系统指标: {get_system_metrics()}")
