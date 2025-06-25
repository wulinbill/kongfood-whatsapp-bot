#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O_co MicroCore - Tension Evaluation (完全优化版)
张力评估与自学习系统 - 从简单版本升级到企业级解决方案
"""

import json
import time
import sqlite3
import hashlib
import threading
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import lru_cache
import statistics
import os
from enum import Enum

# 保持向后兼容的简单接口
@dataclass
class TensionRecord:
    """张力记录（兼容原版）"""
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
    """张力指标（兼容原版）"""
    avg_confidence: float
    success_rate: float
    avg_clarification_count: float
    avg_resolution_time: float
    tension_score: float

# 新增的增强功能
class EvaluationLevel(Enum):
    """评估级别"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

class AnomalyType(Enum):
    """异常类型"""
    UNUSUAL_QUANTITY = "unusual_quantity"
    PRICE_ANOMALY = "price_anomaly"
    TIME_ANOMALY = "time_anomaly"
    PATTERN_DEVIATION = "pattern_deviation"
    RAPID_CHANGES = "rapid_changes"
    CONFIDENCE_DROP = "confidence_drop"

@dataclass
class EnhancedOrderMetrics:
    """增强的订单指标"""
    parse_confidence: float = 0.0
    match_confidence: float = 0.0
    semantic_coherence: float = 0.0
    response_time: float = 0.0
    user_satisfaction: float = 0.0
    completion_rate: float = 0.0
    error_rate: float = 0.0
    clarity_score: float = 0.0
    efficiency_score: float = 0.0
    personalization_score: float = 0.0

@dataclass
class AnomalyDetection:
    """异常检测结果"""
    is_anomaly: bool = False
    anomaly_types: List[AnomalyType] = field(default_factory=list)
    confidence: float = 0.0
    severity: str = "low"  # low, medium, high, critical
    explanation: str = ""
    recommended_action: str = ""

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    total_orders: int = 0
    avg_satisfaction: float = 0.0
    preferred_items: List[str] = field(default_factory=list)
    order_patterns: Dict[str, Any] = field(default_factory=dict)
    learning_speed: float = 1.0
    anomaly_tolerance: float = 0.5
    interaction_style: str = "standard"
    language_preference: str = "es"

class EnhancedDatabaseManager:
    """增强的数据库管理器"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # 优先使用挂载目录，回退到当前目录
            data_dir = Path("/mnt/data") if Path("/mnt/data").exists() else Path(".")
            db_path = data_dir / "enhanced_tension_eval.db"
        
        self.db_path = Path(db_path)
        self.connection = None
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """初始化增强数据库"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_enhanced_tables()
    
    def _create_enhanced_tables(self):
        """创建增强的数据库表"""
        cursor = self.connection.cursor()
        
        # 增强的张力记录表（保持向后兼容）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tension_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                timestamp REAL,
                co_confidence REAL,
                path_score REAL,
                clarification_count INTEGER,
                order_success BOOLEAN,
                user_satisfaction REAL,
                resolution_time REAL,
                error_type TEXT,
                -- 新增字段
                parse_confidence REAL DEFAULT 0.0,
                match_confidence REAL DEFAULT 0.0,
                semantic_coherence REAL DEFAULT 0.0,
                response_time REAL DEFAULT 0.0,
                completion_rate REAL DEFAULT 0.0,
                error_rate REAL DEFAULT 0.0,
                clarity_score REAL DEFAULT 0.0,
                efficiency_score REAL DEFAULT 0.0,
                personalization_score REAL DEFAULT 0.0,
                evaluation_level TEXT DEFAULT 'average',
                final_tension_score REAL DEFAULT 0.0,
                is_anomaly BOOLEAN DEFAULT FALSE,
                anomaly_types TEXT,
                anomaly_confidence REAL DEFAULT 0.0,
                anomaly_severity TEXT,
                ab_test_group TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 用户画像表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                total_orders INTEGER DEFAULT 0,
                avg_satisfaction REAL DEFAULT 0.0,
                preferred_items TEXT DEFAULT '[]',
                order_patterns TEXT DEFAULT '{}',
                learning_speed REAL DEFAULT 1.0,
                anomaly_tolerance REAL DEFAULT 0.5,
                interaction_style TEXT DEFAULT 'standard',
                language_preference TEXT DEFAULT 'es',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 异常记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                anomaly_types TEXT,
                confidence REAL,
                severity TEXT,
                explanation TEXT,
                recommended_action TEXT,
                is_resolved BOOLEAN DEFAULT FALSE,
                timestamp REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # A/B测试表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                test_name TEXT,
                description TEXT,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                groups TEXT,
                success_metric TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 系统指标表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_date DATE UNIQUE,
                avg_confidence REAL,
                success_rate REAL,
                avg_clarification_count REAL,
                avg_resolution_time REAL,
                tension_score REAL,
                total_sessions INTEGER,
                anomaly_rate REAL,
                user_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_tension_timestamp ON tension_records(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_tension_user_id ON tension_records(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_tension_success ON tension_records(order_success)",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_records(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_severity ON anomaly_records(severity)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_date ON system_metrics(metric_date)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.connection.commit()
    
    def save_enhanced_record(self, record: TensionRecord, enhanced_metrics: EnhancedOrderMetrics,
                           user_id: str, session_id: str, anomaly_detection: AnomalyDetection = None,
                           evaluation_level: EvaluationLevel = EvaluationLevel.AVERAGE,
                           ab_test_group: str = None):
        """保存增强的张力记录"""
        with self.lock:
            cursor = self.connection.cursor()
            
            # 处理异常检测数据
            anomaly_types_json = "[]"
            anomaly_conf = 0.0
            anomaly_sev = "low"
            is_anomaly = False
            
            if anomaly_detection:
                is_anomaly = anomaly_detection.is_anomaly
                anomaly_types_json = json.dumps([a.value for a in anomaly_detection.anomaly_types])
                anomaly_conf = anomaly_detection.confidence
                anomaly_sev = anomaly_detection.severity
            
            cursor.execute("""
                INSERT INTO tension_records (
                    session_id, user_id, timestamp, co_confidence, path_score,
                    clarification_count, order_success, user_satisfaction, resolution_time,
                    error_type, parse_confidence, match_confidence, semantic_coherence,
                    response_time, completion_rate, error_rate, clarity_score,
                    efficiency_score, personalization_score, evaluation_level,
                    final_tension_score, is_anomaly, anomaly_types, anomaly_confidence,
                    anomaly_severity, ab_test_group
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, user_id, record.timestamp, record.co_confidence, record.path_score,
                record.clarification_count, record.order_success, record.user_satisfaction,
                record.resolution_time, record.error_type, enhanced_metrics.parse_confidence,
                enhanced_metrics.match_confidence, enhanced_metrics.semantic_coherence,
                enhanced_metrics.response_time, enhanced_metrics.completion_rate,
                enhanced_metrics.error_rate, enhanced_metrics.clarity_score,
                enhanced_metrics.efficiency_score, enhanced_metrics.personalization_score,
                evaluation_level.value, enhanced_metrics.efficiency_score,  # 使用efficiency作为final_score
                is_anomaly, anomaly_types_json, anomaly_conf, anomaly_sev, ab_test_group
            ))
            
            # 保存异常记录
            if is_anomaly and anomaly_detection:
                cursor.execute("""
                    INSERT INTO anomaly_records (
                        session_id, user_id, anomaly_types, confidence, severity,
                        explanation, recommended_action, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, user_id, anomaly_types_json, anomaly_conf, anomaly_sev,
                    anomaly_detection.explanation, anomaly_detection.recommended_action,
                    record.timestamp
                ))
            
            self.connection.commit()
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row:
            return UserProfile(
                user_id=row['user_id'],
                total_orders=row['total_orders'],
                avg_satisfaction=row['avg_satisfaction'],
                preferred_items=json.loads(row['preferred_items']),
                order_patterns=json.loads(row['order_patterns']),
                learning_speed=row['learning_speed'],
                anomaly_tolerance=row['anomaly_tolerance'],
                interaction_style=row['interaction_style'],
                language_preference=row['language_preference']
            )
        
        return None
    
    def update_user_profile(self, profile: UserProfile):
        """更新用户画像"""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles (
                    user_id, total_orders, avg_satisfaction, preferred_items,
                    order_patterns, learning_speed, anomaly_tolerance,
                    interaction_style, language_preference, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                profile.user_id, profile.total_orders, profile.avg_satisfaction,
                json.dumps(profile.preferred_items), json.dumps(profile.order_patterns),
                profile.learning_speed, profile.anomaly_tolerance,
                profile.interaction_style, profile.language_preference
            ))
            self.connection.commit()
    
    def get_historical_records(self, user_id: Optional[str] = None, 
                              days: int = 30, limit: int = 1000) -> List[Dict]:
        """获取历史记录"""
        cursor = self.connection.cursor()
        
        query = "SELECT * FROM tension_records WHERE timestamp > ?"
        params = [time.time() - (days * 24 * 3600)]
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def calculate_enhanced_metrics(self, days: int = 7) -> TensionMetrics:
        """计算增强的系统指标"""
        cursor = self.connection.cursor()
        since_timestamp = time.time() - (days * 24 * 3600)
        
        cursor.execute("""
            SELECT 
                AVG(co_confidence) as avg_confidence,
                AVG(CAST(order_success AS FLOAT)) as success_rate,
                AVG(clarification_count) as avg_clarification_count,
                AVG(resolution_time) as avg_resolution_time,
                AVG(final_tension_score) as tension_score,
                COUNT(*) as total_sessions,
                AVG(CAST(is_anomaly AS FLOAT)) as anomaly_rate
            FROM tension_records 
            WHERE timestamp > ?
        """, (since_timestamp,))
        
        row = cursor.fetchone()
        
        if row and row['total_sessions'] > 0:
            # 保存日常指标
            today = datetime.now().date()
            cursor.execute("""
                INSERT OR REPLACE INTO system_metrics (
                    metric_date, avg_confidence, success_rate, avg_clarification_count,
                    avg_resolution_time, tension_score, total_sessions, anomaly_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                today, row['avg_confidence'], row['success_rate'],
                row['avg_clarification_count'], row['avg_resolution_time'],
                row['tension_score'], row['total_sessions'], row['anomaly_rate']
            ))
            self.connection.commit()
            
            return TensionMetrics(
                avg_confidence=row['avg_confidence'] or 0.5,
                success_rate=row['success_rate'] or 0.5,
                avg_clarification_count=row['avg_clarification_count'] or 1.0,
                avg_resolution_time=row['avg_resolution_time'] or 30.0,
                tension_score=row['tension_score'] or 0.5
            )
        
        # 返回默认值
        return TensionMetrics(0.5, 0.5, 1.0, 30.0, 0.5)

class EnhancedAnomalyDetector:
    """增强的异常检测器"""
    
    def __init__(self, db_manager: EnhancedDatabaseManager):
        self.db_manager = db_manager
        self.thresholds = {
            'confidence_drop_threshold': 0.3,
            'response_time_threshold': 10.0,
            'clarification_threshold': 3,
            'satisfaction_threshold': 0.3,
            'rapid_change_window': 300  # 5分钟
        }
    
    def detect_session_anomalies(self, record: TensionRecord, enhanced_metrics: EnhancedOrderMetrics,
                                user_id: str, historical_data: List[Dict] = None) -> AnomalyDetection:
        """检测会话异常"""
        anomalies = []
        explanations = []
        confidence_scores = []
        
        # 1. 置信度异常
        if record.co_confidence < self.thresholds['confidence_drop_threshold']:
            anomalies.append(AnomalyType.CONFIDENCE_DROP)
            explanations.append(f"Very low confidence: {record.co_confidence:.2f}")
            confidence_scores.append(0.8)
        
        # 2. 响应时间异常
        if enhanced_metrics.response_time > self.thresholds['response_time_threshold']:
            anomalies.append(AnomalyType.TIME_ANOMALY)
            explanations.append(f"Excessive response time: {enhanced_metrics.response_time:.2f}s")
            confidence_scores.append(0.7)
        
        # 3. 澄清次数异常
        if record.clarification_count >= self.thresholds['clarification_threshold']:
            anomalies.append(AnomalyType.RAPID_CHANGES)
            explanations.append(f"Too many clarifications: {record.clarification_count}")
            confidence_scores.append(0.6)
        
        # 4. 满意度异常
        if (record.user_satisfaction is not None and 
            record.user_satisfaction < self.thresholds['satisfaction_threshold']):
            anomalies.append(AnomalyType.PATTERN_DEVIATION)
            explanations.append(f"Very low satisfaction: {record.user_satisfaction:.2f}")
            confidence_scores.append(0.9)
        
        # 5. 基于历史数据的模式检测
        if historical_data and len(historical_data) > 5:
            pattern_anomaly = self._detect_pattern_anomaly(record, historical_data)
            if pattern_anomaly:
                anomalies.append(AnomalyType.PATTERN_DEVIATION)
                explanations.append(pattern_anomaly['explanation'])
                confidence_scores.append(pattern_anomaly['confidence'])
        
        if anomalies:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            severity = self._calculate_severity(anomalies, confidence_scores)
            
            return AnomalyDetection(
                is_anomaly=True,
                anomaly_types=anomalies,
                confidence=avg_confidence,
                severity=severity,
                explanation=" | ".join(explanations),
                recommended_action=self._get_recommended_action(anomalies, severity)
            )
        
        return AnomalyDetection()
    
    def _detect_pattern_anomaly(self, current_record: TensionRecord, 
                               historical_data: List[Dict]) -> Optional[Dict]:
        """检测模式异常"""
        recent_records = [r for r in historical_data if r['timestamp'] > time.time() - 7*24*3600]
        
        if len(recent_records) < 3:
            return None
        
        # 计算历史平均值
        avg_confidence = statistics.mean([r['co_confidence'] for r in recent_records])
        avg_success = statistics.mean([r['order_success'] for r in recent_records])
        
        # 检测显著偏差
        confidence_deviation = abs(current_record.co_confidence - avg_confidence)
        success_deviation = abs((1 if current_record.order_success else 0) - avg_success)
        
        if confidence_deviation > 0.4:
            return {
                'confidence': confidence_deviation,
                'explanation': f"Confidence deviation from user pattern: {confidence_deviation:.2f}"
            }
        
        if success_deviation > 0.5:
            return {
                'confidence': success_deviation,
                'explanation': f"Success rate deviation from user pattern: {success_deviation:.2f}"
            }
        
        return None
    
    def _calculate_severity(self, anomalies: List[AnomalyType], confidence_scores: List[float]) -> str:
        """计算异常严重程度"""
        max_confidence = max(confidence_scores) if confidence_scores else 0.0
        anomaly_count = len(anomalies)
        
        critical_types = {AnomalyType.CONFIDENCE_DROP, AnomalyType.PATTERN_DEVIATION}
        has_critical = any(a in critical_types for a in anomalies)
        
        if has_critical and max_confidence > 0.8:
            return "critical"
        elif anomaly_count >= 3 or max_confidence > 0.7:
            return "high"
        elif anomaly_count >= 2 or max_confidence > 0.5:
            return "medium"
        else:
            return "low"
    
    def _get_recommended_action(self, anomalies: List[AnomalyType], severity: str) -> str:
        """获取推荐行动"""
        actions = {
            "critical": "Immediate intervention required - escalate to human agent",
            "high": "Enhanced validation and monitoring recommended",
            "medium": "Additional verification steps suggested",
            "low": "Continue monitoring for patterns"
        }
        return actions.get(severity, "Monitor and log")

class EnhancedTensionTracker:
    """增强的张力跟踪器（保持向后兼容）"""
    
    def __init__(self, data_dir: str = "/mnt/data"):
        # 初始化增强组件
        self.db_manager = EnhancedDatabaseManager()
        self.anomaly_detector = EnhancedAnomalyDetector(self.db_manager)
        
        # 保持原有接口兼容性
        self.current_session = {}
        self.lock = threading.Lock()
        
        # 学习参数
        self.learning_rates = {
            'confidence_weight': 0.3,
            'success_weight': 0.4,
            'clarification_weight': 0.2,
            'time_weight': 0.1
        }
    
    # 保持原有接口
    def start_session(self, session_id: str, co_confidence: float, path_score: float):
        """开始新的会话跟踪（兼容原版）"""
        self.current_session[session_id] = {
            'start_time': time.time(),
            'co_confidence': co_confidence,
            'path_score': path_score,
            'clarification_count': 0,
            'order_success': False,
            'errors': [],
            'enhanced_metrics': EnhancedOrderMetrics(
                parse_confidence=co_confidence,
                match_confidence=path_score
            )
        }
    
    def add_clarification(self, session_id: str):
        """添加澄清计数（兼容原版）"""
        if session_id in self.current_session:
            self.current_session[session_id]['clarification_count'] += 1
    
    def add_error(self, session_id: str, error_type: str):
        """添加错误记录（兼容原版）"""
        if session_id in self.current_session:
            self.current_session[session_id]['errors'].append(error_type)
    
    def complete_session(self, session_id: str, order_success: bool, 
                        user_satisfaction: Optional[float] = None, user_id: str = "anonymous"):
        """完成会话并记录（增强版）"""
        if session_id not in self.current_session:
            return
        
        session = self.current_session[session_id]
        resolution_time = time.time() - session['start_time']
        
        # 创建基础记录（保持兼容）
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
        
        # 创建增强指标
        enhanced_metrics = session['enhanced_metrics']
        enhanced_metrics.response_time = resolution_time
        enhanced_metrics.user_satisfaction = user_satisfaction or 0.5
        enhanced_metrics.completion_rate = 1.0 if order_success else 0.0
        enhanced_metrics.error_rate = len(session['errors']) / max(1, session['clarification_count'] + 1)
        
        # 计算其他指标
        enhanced_metrics.clarity_score = max(0, 1.0 - (session['clarification_count'] * 0.2))
        enhanced_metrics.efficiency_score = max(0, 1.0 - (resolution_time / 60.0))
        
        # 获取用户历史数据进行异常检测
        historical_data = self.db_manager.get_historical_records(user_id, days=30, limit=50)
        
        # 异常检测
        anomaly_detection = self.anomaly_detector.detect_session_anomalies(
            record, enhanced_metrics, user_id, historical_data
        )
        
        # 确定评估级别
        evaluation_level = self._determine_evaluation_level(
            enhanced_metrics, anomaly_detection, order_success
        )
        
        # 保存增强记录
        self.db_manager.save_enhanced_record(
            record, enhanced_metrics, user_id, session_id,
            anomaly_detection, evaluation_level
        )
        
        # 更新用户画像
        self._update_user_profile(user_id, record, enhanced_metrics)
        
        # 清理会话
        del self.current_session[session_id]
    
    def _determine_evaluation_level(self, metrics: EnhancedOrderMetrics,
                                   anomaly: AnomalyDetection, success: bool) -> EvaluationLevel:
        """确定评估级别"""
        if anomaly.is_anomaly and anomaly.severity in ['critical', 'high']:
            return EvaluationLevel.CRITICAL
        
        if not success:
            return EvaluationLevel.POOR
        
        # 计算综合分数
        score = (metrics.parse_confidence + metrics.match_confidence + 
                metrics.user_satisfaction + metrics.efficiency_score) / 4
        
        if score >= 0.9:
            return EvaluationLevel.EXCELLENT
        elif score >= 0.75:
            return EvaluationLevel.GOOD
        elif score >= 0.6:
            return EvaluationLevel.AVERAGE
        else:
            return EvaluationLevel.POOR
    
    def _update_user_profile(self, user_id: str, record: TensionRecord, 
                           metrics: EnhancedOrderMetrics):
        """更新用户画像"""
        profile = self.db_manager.get_user_profile(user_id)
        if not profile:
            profile = UserProfile(user_id=user_id)
        
        # 更新统计
        profile.total_orders += 1
        
        # 更新平均满意度
        if record.user_satisfaction is not None:
            old_avg = profile.avg_satisfaction
            profile.avg_satisfaction = (old_avg * (profile.total_orders - 1) + 
                                      record.user_satisfaction) / profile.total_orders
        
        # 动态调整学习速度
        if profile.total_orders > 5:
            if profile.avg_satisfaction > 0.8:
                profile.learning_speed = max(0.5, profile.learning_speed * 0.95)  # 专家用户学习慢
            elif profile.avg_satisfaction < 0.5:
                profile.learning_speed = min(2.0, profile.learning_speed * 1.1)   # 困难用户学习快
        
        self.db_manager.update_user_profile(profile)
    
    def get_current_metrics(self) -> TensionMetrics:
        """获取当前指标（兼容原版）"""
        return self.db_manager.calculate_enhanced_metrics()
    
    # 新增的增强方法
    def get_enhanced_analytics(self, days: int = 7) -> Dict[str, Any]:
        """获取增强分析数据"""
        metrics = self.db_manager.calculate_enhanced_metrics(days)
        historical_data = self.db_manager.get_historical_records(days=days)
        
        # 异常统计
        anomaly_count = sum(1 for r in historical_data if r.get('is_anomaly'))
        anomaly_rate = anomaly_count / len(historical_data) if historical_data else 0
        
        # 评估级别分布
        level_counts = {}
        for record in historical_data:
            level = record.get('evaluation_level', 'average')
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'basic_metrics': asdict(metrics),
            'total_sessions': len(historical_data),
            'anomaly_rate': anomaly_rate,
            'evaluation_level_distribution': level_counts,
            'user_count': len(set(r['user_id'] for r in historical_data if r.get('user_id'))),
            'avg_response_time': statistics.mean([r['response_time'] for r in historical_data if r.get('response_time')]) if historical_data else 0,
            'period_days': days
        }

class AdaptiveTensionEvaluator:
    """自适应张力评估器（增强版，保持向后兼容）"""
    
    def __init__(self):
        self.tracker = EnhancedTensionTracker()
        
        # 保持原有的基础阈值
        self.base_thresholds = {
            'clarify_threshold': 0.4,
            'execute_threshold': 0.8,
            'confidence_threshold': 0.6
        }
        
        # 新增的动态学习参数
        self.learning_config = {
            'adaptation_rate': 0.05,
            'min_samples_for_adaptation': 10,
            'confidence_boost_factor': 1.1,
            'experience_bonus_threshold': 20
        }
        
        # 用户个性化缓存
        self.user_threshold_cache = {}
        self.cache_expiry = 3600  # 1小时缓存
    
    def evaluate_path(self, path_data: Dict[str, Any], user_id: str = "anonymous", 
                     session_context: Dict[str, Any] = None) -> float:
        """
        增强的路径评估（保持向后兼容）
        
        Args:
            path_data: 路径数据字典
            user_id: 用户ID（新增）
            session_context: 会话上下文（新增）
            
        Returns:
            张力分数 (0-1)
        """
        if not path_data or not path_data.get('path'):
            return 0.0
        
        # 基础分数计算（保持原逻辑）
        base_score = path_data.get('score', 0.0)
        confidence = path_data.get('confidence', 0.0)
        requires_clarification = path_data.get('requires_clarification', False)
        
        # 获取系统指标
        metrics = self.tracker.get_current_metrics()
        
        # 获取用户个性化阈值
        personalized_thresholds = self._get_personalized_thresholds(user_id)
        
        # 基础张力分数
        tension_score = (base_score + confidence) / 2
        
        # 历史调整（保持原逻辑）
        historical_adjustment = self._calculate_historical_adjustment(metrics)
        adjusted_score = tension_score * historical_adjustment
        
        # 澄清需求惩罚
        if requires_clarification:
            clarification_penalty = personalized_thresholds.get('clarification_penalty', 0.3)
            adjusted_score *= (1 - clarification_penalty)
        
        # 新增：用户经验加成
        user_experience_bonus = self._calculate_user_experience_bonus(user_id)
        adjusted_score += user_experience_bonus
        
        # 新增：上下文加成
        if session_context:
            context_bonus = self._calculate_context_bonus(session_context, path_data)
            adjusted_score += context_bonus
        
        # 新增：时间因素
        time_factor = self._calculate_time_factor()
        adjusted_score *= time_factor
        
        # 最终分数
        final_score = max(0.0, min(1.0, adjusted_score))
        
        return round(final_score, 3)
    
    def _get_personalized_thresholds(self, user_id: str) -> Dict[str, float]:
        """获取个性化阈值"""
        # 检查缓存
        cache_key = f"{user_id}_{int(time.time() // self.cache_expiry)}"
        if cache_key in self.user_threshold_cache:
            return self.user_threshold_cache[cache_key]
        
        # 获取用户画像
        user_profile = self.tracker.db_manager.get_user_profile(user_id)
        thresholds = self.base_thresholds.copy()
        
        if user_profile:
            # 基于用户经验调整
            if user_profile.total_orders > self.learning_config['experience_bonus_threshold']:
                # 经验用户可以接受更低的置信度
                thresholds['clarify_threshold'] *= 0.85
                thresholds['execute_threshold'] *= 0.9
                thresholds['clarification_penalty'] = 0.2  # 减少惩罚
            elif user_profile.total_orders < 5:
                # 新用户需要更高的置信度
                thresholds['clarify_threshold'] *= 1.2
                thresholds['execute_threshold'] *= 1.1
                thresholds['clarification_penalty'] = 0.4  # 增加惩罚
            
            # 基于满意度调整
            if user_profile.avg_satisfaction > 0.8:
                thresholds['confidence_boost'] = self.learning_config['confidence_boost_factor']
            elif user_profile.avg_satisfaction < 0.5:
                thresholds['confidence_boost'] = 0.9
            else:
                thresholds['confidence_boost'] = 1.0
            
            # 基于学习速度调整
            adaptation_factor = user_profile.learning_speed * self.learning_config['adaptation_rate']
            thresholds['adaptation_factor'] = adaptation_factor
        
        # 缓存结果
        self.user_threshold_cache[cache_key] = thresholds
        
        # 清理过期缓存
        if len(self.user_threshold_cache) > 1000:
            old_keys = [k for k in self.user_threshold_cache.keys() 
                       if int(k.split('_')[-1]) < int(time.time() // self.cache_expiry) - 1]
            for key in old_keys:
                del self.user_threshold_cache[key]
        
        return thresholds
    
    def _calculate_user_experience_bonus(self, user_id: str) -> float:
        """计算用户经验加成"""
        user_profile = self.tracker.db_manager.get_user_profile(user_id)
        if not user_profile:
            return 0.0
        
        # 经验加成
        experience_bonus = 0.0
        if user_profile.total_orders > 10:
            experience_bonus = min(0.1, user_profile.total_orders * 0.002)
        
        # 满意度加成
        satisfaction_bonus = max(0, (user_profile.avg_satisfaction - 0.5) * 0.1)
        
        return experience_bonus + satisfaction_bonus
    
    def _calculate_context_bonus(self, session_context: Dict[str, Any], 
                                path_data: Dict[str, Any]) -> float:
        """计算上下文加成"""
        bonus = 0.0
        
        # 语言一致性加成
        if (session_context.get('detected_language') and 
            session_context.get('user_preferred_language')):
            if session_context['detected_language'] == session_context['user_preferred_language']:
                bonus += 0.05
        
        # 重复订单加成
        if session_context.get('is_repeat_order'):
            bonus += 0.08
        
        # 时间一致性加成
        current_hour = datetime.now().hour
        preferred_hours = session_context.get('user_preferred_hours', [])
        if preferred_hours and current_hour in preferred_hours:
            bonus += 0.03
        
        return bonus
    
    def _calculate_time_factor(self) -> float:
        """计算时间因素"""
        current_hour = datetime.now().hour
        
        # 业务高峰期提升置信度
        if 11 <= current_hour <= 14 or 18 <= current_hour <= 21:  # 午餐和晚餐时间
            return 1.05
        # 非营业时间降低置信度
        elif current_hour < 6 or current_hour > 23:
            return 0.95
        else:
            return 1.0
    
    def _calculate_historical_adjustment(self, metrics: TensionMetrics) -> float:
        """基于历史指标计算调整因子（保持原逻辑）"""
        success_adjustment = 0.8 + (metrics.success_rate * 0.4)
        clarify_adjustment = max(0.7, 1.2 - (metrics.avg_clarification_count * 0.2))
        confidence_adjustment = 0.8 + (metrics.avg_confidence * 0.4)
        
        return (success_adjustment + clarify_adjustment + confidence_adjustment) / 3
    
    def _get_dynamic_thresholds(self, metrics: TensionMetrics, user_id: str = "anonymous") -> Dict[str, float]:
        """获取动态阈值（增强版）"""
        personalized = self._get_personalized_thresholds(user_id)
        
        # 合并系统级和个人级调整
        thresholds = self.base_thresholds.copy()
        
        # 系统级调整（保持原逻辑）
        if metrics.success_rate < 0.6:
            thresholds['clarify_threshold'] += 0.1
            thresholds['execute_threshold'] += 0.1
        
        if metrics.avg_clarification_count > 1.5:
            thresholds['clarify_threshold'] -= 0.1
        
        # 应用个性化调整
        for key in thresholds:
            if key in personalized:
                adjustment_factor = personalized.get('adaptation_factor', 0.05)
                thresholds[key] = (thresholds[key] * (1 - adjustment_factor) + 
                                 personalized[key] * adjustment_factor)
        
        return thresholds
    
    # 保持原有接口
    def should_clarify(self, tension_score: float, user_id: str = "anonymous") -> bool:
        """判断是否需要澄清（增强版）"""
        metrics = self.tracker.get_current_metrics()
        thresholds = self._get_dynamic_thresholds(metrics, user_id)
        return tension_score < thresholds['clarify_threshold']
    
    def should_execute(self, tension_score: float, user_id: str = "anonymous") -> bool:
        """判断是否可以直接执行（增强版）"""
        metrics = self.tracker.get_current_metrics()
        thresholds = self._get_dynamic_thresholds(metrics, user_id)
        return tension_score > thresholds['execute_threshold']
    
    def get_action_recommendation(self, tension_score: float, user_id: str = "anonymous", 
                                session_context: Dict[str, Any] = None) -> str:
        """获取动作推荐（增强版）"""
        if self.should_clarify(tension_score, user_id):
            # 个性化澄清策略
            user_profile = self.tracker.db_manager.get_user_profile(user_id)
            if user_profile and user_profile.interaction_style == "concise":
                return "clarify_brief"
            elif user_profile and user_profile.interaction_style == "verbose":
                return "clarify_detailed"
            else:
                return "clarify"
        elif self.should_execute(tension_score, user_id):
            return "execute"
        else:
            # 新增：基于上下文的分析建议
            if session_context and session_context.get('has_ambiguity'):
                return "analyze_ambiguity"
            else:
                return "analyze"
    
    # 新增的增强方法
    def get_confidence_explanation(self, tension_score: float, path_data: Dict[str, Any],
                                 user_id: str = "anonymous") -> Dict[str, Any]:
        """获取置信度解释"""
        explanation = {
            'final_score': tension_score,
            'components': {
                'base_score': path_data.get('score', 0.0),
                'path_confidence': path_data.get('confidence', 0.0),
                'requires_clarification': path_data.get('requires_clarification', False)
            },
            'adjustments': {},
            'recommendation': self.get_action_recommendation(tension_score, user_id),
            'user_factors': {}
        }
        
        # 添加用户因素
        user_profile = self.tracker.db_manager.get_user_profile(user_id)
        if user_profile:
            explanation['user_factors'] = {
                'total_orders': user_profile.total_orders,
                'avg_satisfaction': user_profile.avg_satisfaction,
                'learning_speed': user_profile.learning_speed,
                'experience_level': 'expert' if user_profile.total_orders > 20 else 
                                  'intermediate' if user_profile.total_orders > 5 else 'beginner'
            }
        
        return explanation

# 全局评估器实例（保持向后兼容）
_global_evaluator = AdaptiveTensionEvaluator()

# 保持原有的外部接口
def score(path_data: Dict[str, Any], user_id: str = "anonymous", 
         session_context: Dict[str, Any] = None) -> float:
    """
    外部调用接口 - 评估路径张力分数（增强版）
    
    Args:
        path_data: 路径数据字典
        user_id: 用户ID（新增，向后兼容）
        session_context: 会话上下文（新增，向后兼容）
        
    Returns:
        张力分数 (0-1)
    """
    return _global_evaluator.evaluate_path(path_data, user_id, session_context)

def should_clarify(tension_score: float, user_id: str = "anonymous") -> bool:
    """外部接口 - 判断是否需要澄清（增强版）"""
    return _global_evaluator.should_clarify(tension_score, user_id)

def should_execute(tension_score: float, user_id: str = "anonymous") -> bool:
    """外部接口 - 判断是否可以直接执行（增强版）"""
    return _global_evaluator.should_execute(tension_score, user_id)

def get_action_recommendation(tension_score: float, user_id: str = "anonymous",
                            session_context: Dict[str, Any] = None) -> str:
    """外部接口 - 获取动作推荐（增强版）"""
    return _global_evaluator.get_action_recommendation(tension_score, user_id, session_context)

# 保持原有的会话管理接口
def start_session_tracking(session_id: str, co_confidence: float, path_score: float):
    """开始会话跟踪（兼容原版）"""
    _global_evaluator.tracker.start_session(session_id, co_confidence, path_score)

def add_clarification(session_id: str):
    """添加澄清记录（兼容原版）"""
    _global_evaluator.tracker.add_clarification(session_id)

def complete_session(session_id: str, order_success: bool, 
                    user_satisfaction: Optional[float] = None, user_id: str = "anonymous"):
    """完成会话（增强版）"""
    _global_evaluator.tracker.complete_session(session_id, order_success, user_satisfaction, user_id)

def get_system_metrics() -> Dict[str, Any]:
    """获取系统指标（兼容原版）"""
    metrics = _global_evaluator.tracker.get_current_metrics()
    return asdict(metrics)

# 新增的增强接口
def get_enhanced_analytics(days: int = 7) -> Dict[str, Any]:
    """获取增强分析数据"""
    return _global_evaluator.tracker.get_enhanced_analytics(days)

def get_user_insights(user_id: str) -> Dict[str, Any]:
    """获取用户洞察"""
    profile = _global_evaluator.tracker.db_manager.get_user_profile(user_id)
    if not profile:
        return {'error': 'User not found'}
    
    historical_data = _global_evaluator.tracker.db_manager.get_historical_records(
        user_id, days=30, limit=100
    )
    
    insights = {
        'user_profile': asdict(profile),
        'recent_performance': {
            'total_sessions': len(historical_data),
            'success_rate': sum(1 for r in historical_data if r['order_success']) / len(historical_data) if historical_data else 0,
            'avg_satisfaction': statistics.mean([r['user_satisfaction'] for r in historical_data if r['user_satisfaction']]) if historical_data else 0,
            'anomaly_rate': sum(1 for r in historical_data if r.get('is_anomaly')) / len(historical_data) if historical_data else 0
        },
        'personalized_thresholds': _global_evaluator._get_personalized_thresholds(user_id)
    }
    
    return insights

def get_confidence_explanation(tension_score: float, path_data: Dict[str, Any],
                             user_id: str = "anonymous") -> Dict[str, Any]:
    """获取置信度详细解释"""
    return _global_evaluator.get_confidence_explanation(tension_score, path_data, user_id)

def reset_user_profile(user_id: str) -> bool:
    """重置用户画像"""
    try:
        new_profile = UserProfile(user_id=user_id)
        _global_evaluator.tracker.db_manager.update_user_profile(new_profile)
        return True
    except Exception as e:
        print(f"Error resetting user profile: {e}")
        return False

def export_analytics_data(days: int = 30, format: str = "json") -> Dict[str, Any]:
    """导出分析数据"""
    analytics = get_enhanced_analytics(days)
    historical_data = _global_evaluator.tracker.db_manager.get_historical_records(days=days)
    
    export_data = {
        'export_timestamp': time.time(),
        'period_days': days,
        'analytics': analytics,
        'sample_records': historical_data[:100],  # 限制样本大小
        'format': format
    }
    
    return export_data

# 测试和基准函数
def run_enhanced_tests():
    """运行增强测试"""
    print("=== 增强版 Tension Evaluator 测试 ===\n")
    
    # 1. 向后兼容性测试
    print("1. 向后兼容性测试:")
    
    # 使用原有接口
    test_path = {
        'path': [{'item_name': 'Pollo Teriyaki', 'price': 11.99}],
        'score': 0.85,
        'confidence': 0.8,
        'requires_clarification': False
    }
    
    # 原有调用方式
    tension_score = score(test_path)
    print(f"   原有接口张力分数: {tension_score}")
    print(f"   原有接口动作推荐: {get_action_recommendation(tension_score)}")
    
    # 新接口调用方式
    enhanced_score = score(test_path, user_id="test_user_1")
    print(f"   增强接口张力分数: {enhanced_score}")
    print(f"   增强接口动作推荐: {get_action_recommendation(enhanced_score, 'test_user_1')}")
    
    # 2. 会话跟踪测试
    print("\n2. 会话跟踪测试:")
    
    session_id = "test_session_001"
    start_session_tracking(session_id, 0.7, 0.8)
    add_clarification(session_id)
    complete_session(session_id, True, 0.9, "test_user_1")
    
    print(f"   会话 {session_id} 已完成并记录")
    
    # 3. 用户学习测试
    print("\n3. 用户个性化学习测试:")
    
    # 模拟多次交互
    for i in range(5):
        session_id = f"learning_session_{i}"
        start_session_tracking(session_id, 0.6 + i*0.05, 0.7 + i*0.05)
        
        if i < 2:  # 前两次需要澄清
            add_clarification(session_id)
            
        complete_session(session_id, True, 0.7 + i*0.05, "learning_user")
    
    # 获取学习后的用户洞察
    insights = get_user_insights("learning_user")
    if 'error' not in insights:
        profile = insights['user_profile']
        print(f"   学习用户总订单: {profile['total_orders']}")
        print(f"   平均满意度: {profile['avg_satisfaction']:.3f}")
        print(f"   学习速度: {profile['learning_speed']:.2f}")
        
        # 测试个性化效果
        test_score_before = score(test_path)
        test_score_after = score(test_path, "learning_user")
        print(f"   个性化前分数: {test_score_before}")
        print(f"   个性化后分数: {test_score_after}")
        print(f"   个性化提升: {test_score_after - test_score_before:.3f}")
    
    # 4. 异常检测测试
    print("\n4. 异常检测测试:")
    
    # 创建异常会话
    anomaly_session = "anomaly_session_001"
    start_session_tracking(anomaly_session, 0.1, 0.2)  # 极低置信度
    
    # 添加多次澄清
    for _ in range(4):
        add_clarification(anomaly_session)
    
    complete_session(anomaly_session, False, 0.1, "anomaly_user")  # 失败且低满意度
    
    print("   异常会话已记录，检查分析数据...")
    
    # 5. 系统分析
    print("\n5. 系统分析:")
    
    # 基础指标
    basic_metrics = get_system_metrics()
    print(f"   基础指标: {basic_metrics}")
    
    # 增强分析
    enhanced_analytics = get_enhanced_analytics(days=1)
    print(f"   增强分析: {enhanced_analytics}")
    
    # 6. 性能测试
    print("\n6. 性能测试:")
    
    import time as time_module
    
    # 测试评分性能
    start_time = time_module.time()
    for i in range(100):
        test_score = score(test_path, f"perf_user_{i % 10}")
    
    end_time = time_module.time()
    avg_time = (end_time - start_time) / 100
    
    print(f"   100次评分平均时间: {avg_time:.4f}秒")
    print(f"   每秒处理能力: {1/avg_time:.1f} 评分/秒")
    
    # 7. 数据导出测试
    print("\n7. 数据导出测试:")
    
    export_data = export_analytics_data(days=1)
    print(f"   导出数据包含 {len(export_data['sample_records'])} 条记录")
    print(f"   导出时间戳: {export_data['export_timestamp']}")
    
    print("\n=== 增强测试完成 ===")

# 主程序兼容性测试
if __name__ == "__main__":
    # 保持原有测试的向后兼容性
    print("=== 向后兼容性验证 ===")
    
    # 原有测试代码
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
    
    print("\n=== 增强功能演示 ===")
    run_enhanced_tests()
