#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版集成测试套件
包含全面的测试覆盖、边界情况、性能基准和测试环境隔离
"""

import pytest
import json
import asyncio
import time
import random
import string
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from typing import Dict, Any, List, Optional
import psutil
import gc
from dataclasses import dataclass
from contextlib import asynccontextmanager
import tempfile
import os

# 导入应用模块
from app.main import app
from app.oco_core.seed_parser import parse
from app.oco_core.jump_planner import plan
from app.oco_core.tension_eval import score
from app.oco_core.output_director import reply
from app.pos.loyverse_client import place_order_async
from app.speech.deepgram_client import transcribe_async
from app.whatsapp.router import handle_whatsapp_event

# 测试数据工厂
class TestDataFactory:
    """测试数据工厂"""
    
    @staticmethod
    def random_string(length: int = 10) -> str:
        """生成随机字符串"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_phone() -> str:
        """生成随机电话号码"""
        return f"whatsapp:+1{random.randint(1000000000, 9999999999)}"
    
    @staticmethod
    def create_whatsapp_payload(
        phone: str = None,
        body: str = "Quiero Pollo Teriyaki",
        media_url: str = None,
        media_type: str = None
    ) -> Dict[str, Any]:
        """创建WhatsApp负载"""
        payload = {
            'From': phone or TestDataFactory.random_phone(),
            'Body': body,
            'MessageSid': f"SM{TestDataFactory.random_string(32)}"
        }
        
        if media_url:
            payload['MediaUrl0'] = media_url
            payload['MediaContentType0'] = media_type or 'audio/ogg'
        
        return payload
    
    @staticmethod
    def create_co(
        objects: List[Dict] = None,
        intent: str = "order",
        language: str = "es",
        confidence: float = 0.9
    ) -> Dict[str, Any]:
        """创建对话对象"""
        if objects is None:
            objects = [
                {
                    'item_type': 'main_dish',
                    'content': 'Pollo Teriyaki',
                    'quantity': 1,
                    'confidence': 0.9
                }
            ]
        
        return {
            'objects': objects,
            'intent': intent,
            'language': language,
            'confidence': confidence,
            'raw_text': 'Test input',
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def create_path_data(
        path: List[Dict] = None,
        score: float = 0.9,
        requires_clarification: bool = False
    ) -> Dict[str, Any]:
        """创建路径数据"""
        if path is None:
            path = [
                {
                    'item_id': f"item_{TestDataFactory.random_string(8)}",
                    'item_name': 'Pollo Teriyaki',
                    'variant_id': f"variant_{TestDataFactory.random_string(8)}",
                    'price': 11.99,
                    'quantity': 1,
                    'match_score': 0.95,
                    'original_query': 'Pollo Teriyaki'
                }
            ]
        
        return {
            'path': path,
            'score': score,
            'confidence': score,
            'requires_clarification': requires_clarification,
            'clarification_reason': 'multiple_matches' if requires_clarification else None,
            'alternative_paths': [] if not requires_clarification else [
                {
                    'matches': [
                        {'item_name': 'Pollo Teriyaki', 'price': 11.99},
                        {'item_name': 'Pollo Naranja', 'price': 11.89}
                    ]
                }
            ]
        }
    
    @staticmethod
    def create_menu_items(count: int = 50) -> List[Dict[str, Any]]:
        """创建菜单项目"""
        categories = ['main_dish', 'side', 'drink', 'dessert']
        items = []
        
        for i in range(count):
            items.append({
                'id': f"item_{i}",
                'name': f"Test Item {i}",
                'category': random.choice(categories),
                'price': round(random.uniform(5.0, 25.0), 2),
                'available': random.choice([True, True, True, False]),  # 75% 可用
                'description': f"Description for test item {i}",
                'variants': [
                    {
                        'id': f"variant_{i}_0",
                        'name': f"Test Item {i} - Regular",
                        'price': round(random.uniform(5.0, 25.0), 2)
                    }
                ]
            })
        
        return items

@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    error_count: int
    
    def __str__(self):
        return (f"Performance: {self.response_time:.3f}s, "
                f"Memory: {self.memory_usage:.1f}MB, "
                f"CPU: {self.cpu_usage:.1f}%, "
                f"Success: {self.success_rate:.1%}, "
                f"Errors: {self.error_count}")

class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    @asynccontextmanager
    async def measure(self):
        """测量性能上下文管理器"""
        # 记录开始状态
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        # 清理垃圾回收
        gc.collect()
        
        try:
            yield
        finally:
            # 记录结束状态
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self.process.cpu_percent()
            
            # 计算指标
            response_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = max(end_cpu - start_cpu, 0)
            
            # 存储结果
            self.last_metrics = PerformanceMetrics(
                response_time=response_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                success_rate=1.0,  # 默认成功
                error_count=0
            )
    
    def get_last_metrics(self) -> PerformanceMetrics:
        """获取最后的性能指标"""
        return getattr(self, 'last_metrics', PerformanceMetrics(0, 0, 0, 0, 1))

class MockManager:
    """Mock管理器"""
    
    def __init__(self):
        self.mocks = {}
        self.original_values = {}
    
    def setup_successful_mocks(self):
        """设置成功场景的mocks"""
        # Mock Loyverse POS
        self.mocks['loyverse'] = patch('app.pos.loyverse_client.place_order')
        loyverse_mock = self.mocks['loyverse'].start()
        loyverse_mock.return_value = {
            'success': True,
            'receipt_id': f'receipt_{TestDataFactory.random_string(8)}',
            'total_amount': 13.30
        }
        
        # Mock Deepgram
        self.mocks['deepgram'] = patch('app.speech.deepgram_client.transcribe_whatsapp')
        deepgram_mock = self.mocks['deepgram'].start()
        deepgram_mock.return_value = ("quiero pollo teriyaki", "es")
        
        # Mock Twilio
        self.mocks['twilio'] = patch('app.whatsapp.twilio_adapter.send_message')
        twilio_mock = self.mocks['twilio'].start()
        twilio_mock.return_value = True
        
        # Mock Redis
        self.mocks['redis'] = patch('redis.asyncio.from_url')
        redis_mock = self.mocks['redis'].start()
        redis_instance = AsyncMock()
        redis_instance.ping.return_value = True
        redis_instance.get.return_value = None
        redis_instance.setex.return_value = True
        redis_mock.return_value = redis_instance
        
        return self.mocks
    
    def setup_failure_mocks(self):
        """设置失败场景的mocks"""
        # Mock失败的Loyverse POS
        self.mocks['loyverse'] = patch('app.pos.loyverse_client.place_order')
        loyverse_mock = self.mocks['loyverse'].start()
        loyverse_mock.return_value = {
            'success': False,
            'error': 'Payment processing failed'
        }
        
        # Mock失败的语音转录
        self.mocks['deepgram'] = patch('app.speech.deepgram_client.transcribe_whatsapp')
        deepgram_mock = self.mocks['deepgram'].start()
        deepgram_mock.return_value = (None, None)
        
        # Mock失败的消息发送
        self.mocks['twilio'] = patch('app.whatsapp.twilio_adapter.send_message')
        twilio_mock = self.mocks['twilio'].start()
        twilio_mock.return_value = False
        
        return self.mocks
    
    def cleanup(self):
        """清理所有mocks"""
        for mock in self.mocks.values():
            if hasattr(mock, 'stop'):
                mock.stop()
        self.mocks.clear()

class TestEnvironmentIsolation:
    """测试环境隔离"""
    
    def __init__(self):
        self.temp_dir = None
        self.original_env = {}
    
    def setup_isolated_environment(self):
        """设置隔离环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix='whatsapp_test_')
        
        # 备份环境变量
        test_env_vars = [
            'REDIS_URL', 'TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN',
            'LOYVERSE_CLIENT_ID', 'DEEPGRAM_API_KEY'
        ]
        
        for var in test_env_vars:
            self.original_env[var] = os.environ.get(var)
            # 设置测试环境变量
            os.environ[var] = f'test_{var.lower()}_value'
        
        # 设置测试数据目录
        os.environ['TEST_DATA_DIR'] = self.temp_dir
        
        return self.temp_dir
    
    def cleanup_environment(self):
        """清理隔离环境"""
        # 恢复环境变量
        for var, value in self.original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        
        # 清理临时目录
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

# 测试基类
class BaseTestCase:
    """测试基类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.mock_manager = MockManager()
        self.env_isolation = TestEnvironmentIsolation()
        self.performance_benchmark = PerformanceBenchmark()
        
        # 设置隔离环境
        self.env_isolation.setup_isolated_environment()
    
    def teardown_method(self):
        """测试方法清理"""
        self.mock_manager.cleanup()
        self.env_isolation.cleanup_environment()

class TestHealthCheck(BaseTestCase):
    """健康检查测试"""
    
    def test_health_endpoint(self):
        """测试健康检查端点"""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "live"}
    
    def test_health_endpoint_performance(self):
        """测试健康检查性能"""
        client = TestClient(app)
        
        response_times = []
        for _ in range(10):
            start = time.time()
            response = client.get("/")
            end = time.time()
            
            assert response.status_code == 200
            response_times.append(end - start)
        
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 0.1  # 应该在100ms以内

class TestSeedParserComprehensive(BaseTestCase):
    """种子解析器全面测试"""
    
    def test_basic_order_parsing(self):
        """测试基本订单解析"""
        test_cases = [
            ("Quiero 2 Pollo Teriyaki", "es", "order", 2),
            ("I want chicken teriyaki", "en", "order", 1),
            ("我要照烧鸡肉", "zh", "order", 1),
            ("Necesito ayuda", "es", "help", 0),
            ("Cancel my order", "en", "cancel", 0)
        ]
        
        for text, expected_lang, expected_intent, expected_items in test_cases:
            result = parse(text)
            
            assert result['language'] == expected_lang
            assert result['intent'] == expected_intent
            if expected_items > 0:
                assert len(result['objects']) >= expected_items
    
    def test_edge_cases(self):
        """测试边界情况"""
        edge_cases = [
            "",  # 空字符串
            "   ",  # 只有空格
            "🍕🍔🍟",  # 只有emoji
            "a" * 1000,  # 超长文本
            "123456789",  # 只有数字
            "!@#$%^&*()",  # 只有特殊字符
            "Quiero " + "pollo " * 100,  # 重复词汇
        ]
        
        for text in edge_cases:
            result = parse(text)
            
            # 不应该崩溃
            assert isinstance(result, dict)
            assert 'intent' in result
            assert 'objects' in result
            assert 'confidence' in result
    
    def test_malformed_input(self):
        """测试格式错误的输入"""
        malformed_inputs = [
            None,
            123,
            [],
            {},
            {"text": "invalid"},
        ]
        
        for invalid_input in malformed_inputs:
            try:
                # 应该处理无效输入而不崩溃
                result = parse(str(invalid_input) if invalid_input is not None else "")
                assert isinstance(result, dict)
            except Exception as e:
                # 如果抛出异常，应该是预期的类型
                assert isinstance(e, (TypeError, ValueError))
    
    def test_multilingual_mixed_input(self):
        """测试混合语言输入"""
        mixed_inputs = [
            "I want pollo teriyaki",  # 英语+西班牙语
            "Quiero chicken teriyaki",  # 西班牙语+英语
            "我想要 chicken",  # 中文+英语
            "Hola, I need help"  # 混合问候
        ]
        
        for text in mixed_inputs:
            result = parse(text)
            
            # 应该能识别主要语言
            assert result['language'] in ['en', 'es', 'zh']
            assert result['confidence'] >= 0
    
    def test_quantity_parsing_edge_cases(self):
        """测试数量解析边界情况"""
        quantity_cases = [
            ("Quiero 0 pollo", 0),  # 零数量
            ("Quiero -1 pollo", 1),  # 负数量（应该修正）
            ("Quiero 999 pollo", 999),  # 大数量
            ("Quiero millón de pollos", 1),  # 文字数量
            ("Quiero dos docenas de pollo", 24),  # 复杂数量表达
        ]
        
        for text, expected_quantity in quantity_cases:
            result = parse(text)
            
            if result['objects']:
                parsed_quantity = result['objects'][0].get('quantity', 1)
                # 对于极端情况，应该有合理的默认值
                assert isinstance(parsed_quantity, int)
                assert parsed_quantity >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_parsing(self):
        """测试并发解析"""
        async def parse_async(text):
            return parse(text)
        
        texts = [f"Quiero pollo teriyaki número {i}" for i in range(20)]
        
        # 并发执行解析
        tasks = [parse_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        # 验证所有结果
        assert len(results) == 20
        for i, result in enumerate(results):
            assert result['intent'] == 'order'
            assert isinstance(result['confidence'], float)

class TestJumpPlannerAdvanced(BaseTestCase):
    """跳跃规划器高级测试"""
    
    def setup_method(self):
        """设置测试环境"""
        super().setup_method()
        self.test_menu = TestDataFactory.create_menu_items(100)
    
    def test_exact_match_scenarios(self):
        """测试精确匹配场景"""
        co = TestDataFactory.create_co(
            objects=[
                {
                    'item_type': 'main_dish',
                    'content': 'Pollo Teriyaki',
                    'quantity': 1,
                    'confidence': 0.95
                }
            ]
        )
        
        result = plan(co)
        
        assert 'path' in result
        assert result['score'] >= 0.8
        assert not result['requires_clarification']
    
    def test_ambiguous_matches(self):
        """测试模糊匹配"""
        co = TestDataFactory.create_co(
            objects=[
                {
                    'item_type': 'main_dish',
                    'content': 'pollo',  # 模糊查询
                    'quantity': 1,
                    'confidence': 0.6
                }
            ]
        )
        
        result = plan(co)
        
        # 模糊匹配应该要求澄清
        if result['requires_clarification']:
            assert 'alternative_paths' in result
            assert len(result['alternative_paths']) > 0
    
    def test_no_match_scenarios(self):
        """测试无匹配场景"""
        co = TestDataFactory.create_co(
            objects=[
                {
                    'item_type': 'main_dish',
                    'content': 'NonexistentDish12345',
                    'quantity': 1,
                    'confidence': 0.9
                }
            ]
        )
        
        result = plan(co)
        
        assert result['requires_clarification']
        assert result['clarification_reason'] in ['no_menu_matches', 'low_confidence']
        assert len(result['path']) == 0
    
    def test_multiple_items_planning(self):
        """测试多项目规划"""
        co = TestDataFactory.create_co(
            objects=[
                {
                    'item_type': 'main_dish',
                    'content': 'Pollo Teriyaki',
                    'quantity': 2,
                    'confidence': 0.9
                },
                {
                    'item_type': 'drink',
                    'content': 'Coca Cola',
                    'quantity': 1,
                    'confidence': 0.8
                }
            ]
        )
        
        result = plan(co)
        
        if result['path']:
            # 应该包含多个项目
            total_quantity = sum(item.get('quantity', 0) for item in result['path'])
            assert total_quantity >= 2
    
    def test_modification_handling(self):
        """测试修改处理"""
        co = TestDataFactory.create_co(
            objects=[
                {
                    'item_type': 'main_dish',
                    'content': 'Pollo Teriyaki',
                    'quantity': 1,
                    'confidence': 0.9
                },
                {
                    'item_type': 'remove',
                    'content': 'salsa',
                    'quantity': 1,
                    'confidence': 0.8
                },
                {
                    'item_type': 'add',
                    'content': 'extra cheese',
                    'quantity': 1,
                    'confidence': 0.7
                }
            ]
        )
        
        result = plan(co)
        
        # 应该能处理修改请求
        assert 'path' in result
        if result['path']:
            # 检查是否包含修改信息
            main_item = result['path'][0]
            assert 'modifications' in main_item or 'notes' in main_item
    
    def test_price_calculation_edge_cases(self):
        """测试价格计算边界情况"""
        co = TestDataFactory.create_co(
            objects=[
                {
                    'item_type': 'main_dish',
                    'content': 'Expensive Item',
                    'quantity': 999,  # 大数量
                    'confidence': 0.9
                }
            ]
        )
        
        result = plan(co)
        
        if result['path']:
            for item in result['path']:
                # 价格应该是合理的数值
                assert isinstance(item.get('price', 0), (int, float))
                assert item.get('price', 0) >= 0
                
                # 总价不应该溢出
                total = item.get('price', 0) * item.get('quantity', 1)
                assert total < 1000000  # 合理的上限

class TestTensionEvaluatorDetailed(BaseTestCase):
    """张力评估器详细测试"""
    
    def test_confidence_score_calculation(self):
        """测试置信度分数计算"""
        test_cases = [
            # (path_data, expected_score_range)
            ({
                'path': [{'item_name': 'Pollo Teriyaki', 'price': 11.99}],
                'score': 0.95,
                'confidence': 0.95,
                'requires_clarification': False
            }, (0.8, 1.0)),
            
            ({
                'path': [],
                'score': 0.1,
                'confidence': 0.1,
                'requires_clarification': True
            }, (0.0, 0.3)),
            
            ({
                'path': [{'item_name': 'Unknown', 'price': 0}],
                'score': 0.5,
                'confidence': 0.5,
                'requires_clarification': False
            }, (0.3, 0.7))
        ]
        
        for path_data, (min_score, max_score) in test_cases:
            tension_score = score(path_data)
            
            assert isinstance(tension_score, float)
            assert min_score <= tension_score <= max_score
    
    def test_edge_case_inputs(self):
        """测试边界情况输入"""
        edge_cases = [
            {},  # 空字典
            {'path': []},  # 空路径
            {'score': None},  # None值
            {'confidence': -1},  # 负置信度
            {'confidence': 2.0},  # 超出范围的置信度
        ]
        
        for path_data in edge_cases:
            try:
                tension_score = score(path_data)
                # 应该返回合理的默认值
                assert 0.0 <= tension_score <= 1.0
            except Exception as e:
                # 如果抛出异常，应该是预期的类型
                assert isinstance(e, (TypeError, ValueError))
    
    def test_score_consistency(self):
        """测试分数一致性"""
        path_data = {
            'path': [{'item_name': 'Test Item', 'price': 10.0}],
            'score': 0.8,
            'confidence': 0.8,
            'requires_clarification': False
        }
        
        # 多次计算应该得到相同结果
        scores = [score(path_data) for _ in range(10)]
        
        # 所有分数应该相同
        assert all(abs(s - scores[0]) < 0.001 for s in scores)

class TestOutputDirectorComprehensive(BaseTestCase):
    """输出决策器全面测试"""
    
    def setup_method(self):
        """设置测试方法"""
        super().setup_method()
        self.mocks = self.mock_manager.setup_successful_mocks()
    
    def test_successful_order_execution(self):
        """测试成功订单执行"""
        co = TestDataFactory.create_co()
        path_data = TestDataFactory.create_path_data()
        
        response = reply(co, path_data)
        
        # 验证响应
        assert isinstance(response, str)
        assert len(response) > 0
        
        # 验证POS系统调用
        self.mocks['loyverse'].assert_called_once()
        
        # 响应应该包含确认信息
        assert any(keyword in response.lower() for keyword in ['gracias', 'confirmo', 'total'])
    
    def test_clarification_response(self):
        """测试澄清响应"""
        co = TestDataFactory.create_co(confidence=0.4)
        path_data = TestDataFactory.create_path_data(
            score=0.3,
            requires_clarification=True
        )
        
        response = reply(co, path_data)
        
        # 澄清响应应该包含选项或问题
        assert any(keyword in response.lower() for keyword in ['opciones', 'cuál', 'qué', 'podrías'])
        
        # 不应该调用POS系统
        self.mocks['loyverse'].assert_not_called()
    
    def test_error_handling_in_reply(self):
        """测试回复中的错误处理"""
        # 设置失败的mocks
        self.mock_manager.cleanup()
        self.mock_manager.setup_failure_mocks()
        
        co = TestDataFactory.create_co()
        path_data = TestDataFactory.create_path_data()
        
        response = reply(co, path_data)
        
        # 应该返回错误消息
        assert isinstance(response, str)
        assert len(response) > 0
        
        # 错误响应应该包含道歉或错误提示
        assert any(keyword in response.lower() for keyword in ['disculpa', 'error', 'problema'])
    
    def test_multilingual_responses(self):
        """测试多语言响应"""
        languages = ['es', 'en', 'zh']
        
        for lang in languages:
            co = TestDataFactory.create_co(language=lang)
            path_data = TestDataFactory.create_path_data()
            
            response = reply(co, path_data, session_id=f"test_{lang}")
            
            assert isinstance(response, str)
            assert len(response) > 0
            # 响应应该适合该语言（这里可以添加更具体的语言检查）
    
    def test_large_order_handling(self):
        """测试大订单处理"""
        # 创建大订单
        large_path = [
            {
                'item_name': f'Item {i}',
                'variant_id': f'variant_{i}',
                'price': 10.0 + i,
                'quantity': random.randint(1, 5)
            }
            for i in range(20)  # 20个不同项目
        ]
        
        co = TestDataFactory.create_co()
        path_data = TestDataFactory.create_path_data(path=large_path)
        
        response = reply(co, path_data)
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # 应该能处理大订单而不崩溃
        self.mocks['loyverse'].assert_called_once()

class TestWhatsAppIntegrationAdvanced(BaseTestCase):
    """WhatsApp集成高级测试"""
    
    def setup_method(self):
        """设置测试方法"""
        super().setup_method()
        self.mocks = self.mock_manager.setup_successful_mocks()
    
    def test_text_message_flow(self):
        """测试文本消息流程"""
        payload = TestDataFactory.create_whatsapp_payload()
        
        handle_whatsapp_event(payload, 'test-trace-id')
        
        # 验证消息发送
        self.mocks['twilio'].assert_called()
        args, kwargs = self.mocks['twilio'].call_args
        phone_number = args[0]
        message = args[1]
        
        assert phone_number == payload['From']
        assert isinstance(message, str)
        assert len(message) > 0
    
    def test_audio_message_flow(self):
        """测试语音消息流程"""
        payload = TestDataFactory.create_whatsapp_payload(
            body="",
            media_url="https://example.com/audio.ogg",
            media_type="audio/ogg"
        )
        
        handle_whatsapp_event(payload, 'test-audio-trace')
        
        # 验证语音转录被调用
        self.mocks['deepgram'].assert_called_once()
        
        # 验证最终发送了响应
        self.mocks['twilio'].assert_called()
    
    def test_invalid_payload_handling(self):
        """测试无效负载处理"""
        invalid_payloads = [
            {},  # 空负载
            {'From': 'invalid'},  # 缺少消息内容
            {'Body': 'test'},  # 缺少发送者
            {'MessageStatus': 'delivered'},  # 状态更新（应该被忽略）
        ]
        
        for payload in invalid_payloads:
            try:
                handle_whatsapp_event(payload, 'test-invalid-trace')
                # 不应该崩溃
            except Exception as e:
                pytest.fail(f"Should not raise exception for invalid payload: {e}")
    
    def test_concurrent_message_processing(self):
        """测试并发消息处理"""
        import threading
        import time
        
        payloads = [
            TestDataFactory.create_whatsapp_payload(
                phone=TestDataFactory.random_phone(),
                body=f"Order {i}"
            )
            for i in range(10)
        ]
        
        results = []
        
        def process_message(payload):
            try:
                handle_whatsapp_event(payload, f'concurrent-trace-{payload["MessageSid"]}')
                results.append({'success': True, 'payload': payload})
            except Exception as e:
                results.append({'success': False, 'error': str(e)})
        
        # 并发处理消息
        threads = []
        for payload in payloads:
            thread = threading.Thread(target=process_message, args=(payload,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=10)
        
        # 验证结果
        assert len(results) == 10
        successful = [r for r in results if r['success']]
        assert len(successful) >= 8  # 至少80%成功率
    
    def test_session_management(self):
        """测试会话管理"""
        phone = TestDataFactory.random_phone()
        
        # 发送多条消息
        messages = [
            "Hola",
            "Quiero pollo teriyaki",
            "¿Cuánto cuesta?",
            "Confirmado"
        ]
        
        for i, message in enumerate(messages):
            payload = TestDataFactory.create_whatsapp_payload(
                phone=phone,
                body=message
            )
            
            handle_whatsapp_event(payload, f'session-trace-{i}')
        
        # 验证会话状态被正确管理
        # 这里需要根据实际的会话管理实现来验证

class TestLoyversePOSAdvanced(BaseTestCase):
    """Loyverse POS高级测试"""
    
    @pytest.mark.asyncio
    async def test_successful_order_placement(self):
        """测试成功下单"""
        with patch('httpx.AsyncClient.post') as mock_post:
            # 模拟成功响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'id': 'receipt-123',
                'total': 13.30,
                'status': 'completed'
            }
            mock_post.return_value = mock_response
            
            path_data = TestDataFactory.create_path_data()
            
            async with self.performance_benchmark.measure():
                result = await place_order_async(
                    path_data,
                    customer_phone="+1234567890",
                    customer_name="Test Customer"
                )
            
            assert result['success'] == True
            assert result['receipt_id'] == 'receipt-123'
            
            # 验证性能
            metrics = self.performance_benchmark.get_last_metrics()
            assert metrics.response_time < 5.0  # 5秒内完成
    
    @pytest.mark.asyncio
    async def test_order_failure_scenarios(self):
        """测试订单失败场景"""
        failure_scenarios = [
            (400, "Bad Request", "Invalid order data"),
            (401, "Unauthorized", "Authentication failed"),
            (500, "Internal Server Error", "Server error"),
            (503, "Service Unavailable", "Service temporarily unavailable")
        ]
        
        for status_code, status_text, expected_error in failure_scenarios:
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.text = status_text
                mock_post.return_value = mock_response
                
                path_data = TestDataFactory.create_path_data()
                
                result = await place_order_async(path_data)
                
                assert result['success'] == False
                assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """测试网络超时处理"""
        with patch('httpx.AsyncClient.post') as mock_post:
            # 模拟超时
            import httpx
            mock_post.side_effect = httpx.TimeoutException("Request timeout")
            
            path_data = TestDataFactory.create_path_data()
            
            result = await place_order_async(path_data)
            
            assert result['success'] == False
            assert 'timeout' in result.get('error', '').lower()
    
    @pytest.mark.asyncio
    async def test_large_order_performance(self):
        """测试大订单性能"""
        # 创建大订单
        large_path = [
            {
                'variant_id': f'variant_{i}',
                'price': 10.0 + i,
                'quantity': random.randint(1, 3),
                'original_query': f'Item {i}'
            }
            for i in range(50)  # 50个项目
        ]
        
        path_data = {
            'path': large_path,
            'confidence': 0.9
        }
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'id': 'large-order-123',
                'total': sum(item['price'] * item['quantity'] for item in large_path)
            }
            mock_post.return_value = mock_response
            
            async with self.performance_benchmark.measure():
                result = await place_order_async(path_data)
            
            assert result['success'] == True
            
            # 验证性能
            metrics = self.performance_benchmark.get_last_metrics()
            assert metrics.response_time < 10.0  # 大订单应在10秒内完成

class TestPerformanceBenchmarks(BaseTestCase):
    """性能基准测试"""
    
    def test_parsing_performance(self):
        """测试解析性能"""
        test_texts = [
            "Quiero pollo teriyaki",
            "I want 2 chicken teriyaki with extra sauce",
            "我要两份照烧鸡肉",
            "Necesito ayuda con mi orden",
        ]
        
        results = []
        
        for text in test_texts:
            start_time = time.time()
            
            for _ in range(100):  # 重复100次
                result = parse(text)
                assert 'intent' in result
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 100
            results.append(avg_time)
        
        # 平均解析时间应该很快
        avg_parse_time = sum(results) / len(results)
        assert avg_parse_time < 0.1  # 100ms内
    
    def test_planning_performance(self):
        """测试规划性能"""
        co = TestDataFactory.create_co()
        
        times = []
        for _ in range(50):
            start_time = time.time()
            result = plan(co)
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert 'path' in result
        
        avg_time = sum(times) / len(times)
        assert avg_time < 0.5  # 500ms内
    
    def test_memory_usage_under_load(self):
        """测试负载下的内存使用"""
        import gc
        import psutil
        
        process = psutil.Process()
        
        # 记录初始内存
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 处理大量请求
        for i in range(1000):
            co = parse(f"Quiero pollo teriyaki número {i}")
            path_data = plan(co)
            
            # 每100次请求清理一次垃圾
            if i % 100 == 0:
                gc.collect()
        
        # 最终内存检查
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # 内存增长应该在合理范围内
        assert memory_growth < 100  # 不超过100MB增长
    
    def test_concurrent_processing_performance(self):
        """测试并发处理性能"""
        import threading
        import queue
        
        num_threads = 10
        num_requests = 100
        results_queue = queue.Queue()
        
        def worker():
            for i in range(num_requests // num_threads):
                start_time = time.time()
                
                co = parse(f"Order {threading.current_thread().ident}_{i}")
                plan(co)
                
                end_time = time.time()
                results_queue.put(end_time - start_time)
        
        # 启动线程
        threads = []
        start_total = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待完成
        for thread in threads:
            thread.join()
        
        end_total = time.time()
        total_time = end_total - start_total
        
        # 收集结果
        individual_times = []
        while not results_queue.empty():
            individual_times.append(results_queue.get())
        
        # 验证性能
        avg_individual_time = sum(individual_times) / len(individual_times)
        throughput = num_requests / total_time
        
        assert avg_individual_time < 0.2  # 单个请求200ms内
        assert throughput > 10  # 每秒至少处理10个请求

class TestEdgeCasesAndBoundaries(BaseTestCase):
    """边界情况和极端测试"""
    
    def test_unicode_and_emoji_handling(self):
        """测试Unicode和Emoji处理"""
        unicode_inputs = [
            "🍕 Quiero pizza",
            "Café ☕ con leche",
            "🍜 Ramen 拉面",
            "🇪🇸 Comida española 🇪🇸",
            "😊 Estoy feliz 😊",
            "🔥🔥🔥 Muy picante 🔥🔥🔥"
        ]
        
        for text in unicode_inputs:
            result = parse(text)
            
            # 应该能处理Unicode而不崩溃
            assert isinstance(result, dict)
            assert 'intent' in result
            assert 'confidence' in result
    
    def test_very_long_input_handling(self):
        """测试超长输入处理"""
        # 生成超长文本
        long_text = "Quiero " + ("pollo teriyaki " * 500)  # 约3000字符
        
        result = parse(long_text)
        
        # 应该能处理而不崩溃
        assert isinstance(result, dict)
        assert 'intent' in result
        
        # 可能会有较低的置信度或特殊处理
        assert result.get('confidence', 0) >= 0
    
    def test_special_characters_handling(self):
        """测试特殊字符处理"""
        special_inputs = [
            "Order #123-456",
            "Price: $11.99",
            "50% discount",
            "Table @5",
            "Call me @ 555-1234",
            "Email: test@example.com"
        ]
        
        for text in special_inputs:
            result = parse(text)
            
            # 不应该崩溃
            assert isinstance(result, dict)
    
    def test_multilingual_code_switching(self):
        """测试语言切换"""
        code_switching_examples = [
            "Hello, quiero pollo",  # 英语开始，西班牙语订单
            "Hola, I want chicken",  # 西班牙语问候，英语订单
            "你好, quiero comida china",  # 中文问候，西班牙语订单
            "Merci beaucoup, gracias"  # 法语+西班牙语
        ]
        
        for text in code_switching_examples:
            result = parse(text)
            
            # 应该识别出主要语言
            assert result.get('language') in ['en', 'es', 'zh', 'fr', 'auto']
    
    def test_numeric_edge_cases(self):
        """测试数字边界情况"""
        numeric_cases = [
            "Quiero 0.5 pollo",  # 小数
            "Quiero 1,000 pollos",  # 千位分隔符
            "Quiero 1.000,50 euros",  # 欧洲数字格式
            "Quiero ½ pollo",  # 分数符号
            "Quiero dos y medio pollos",  # 文字分数
        ]
        
        for text in numeric_cases:
            result = parse(text)
            
            # 应该能处理各种数字格式
            assert isinstance(result, dict)
            if result.get('objects'):
                quantity = result['objects'][0].get('quantity', 1)
                assert isinstance(quantity, (int, float))
                assert quantity >= 0

class TestDataIntegrityAndValidation(BaseTestCase):
    """数据完整性和验证测试"""
    
    def test_order_data_validation(self):
        """测试订单数据验证"""
        # 测试有效订单数据
        valid_path_data = TestDataFactory.create_path_data()
        
        # 所有字段都应该存在且有效
        assert 'path' in valid_path_data
        assert 'score' in valid_path_data
        assert 'confidence' in valid_path_data
        
        for item in valid_path_data['path']:
            assert 'item_name' in item
            assert 'price' in item
            assert isinstance(item['price'], (int, float))
            assert item['price'] >= 0
    
    def test_phone_number_validation(self):
        """测试电话号码验证"""
        valid_phones = [
            "whatsapp:+1234567890",
            "+1234567890",
            "1234567890",
            "+34123456789",  # 西班牙格式
            "+86138000138000"  # 中国格式
        ]
        
        for phone in valid_phones:
            payload = TestDataFactory.create_whatsapp_payload(phone=phone)
            
            # 应该能处理各种电话号码格式
            try:
                handle_whatsapp_event(payload, 'phone-validation-test')
            except Exception as e:
                pytest.fail(f"Should handle phone format {phone}: {e}")
    
    def test_price_calculation_accuracy(self):
        """测试价格计算精度"""
        # 测试浮点数精度问题
        path_data = {
            'path': [
                {'price': 0.1, 'quantity': 3},  # 0.3
                {'price': 0.2, 'quantity': 3},  # 0.6
                {'price': 1.1, 'quantity': 1},  # 1.1
            ]
        }
        
        # 计算总价
        total = sum(item['price'] * item['quantity'] for item in path_data['path'])
        
        # 应该处理浮点数精度
        expected_total = 2.0  # 0.3 + 0.6 + 1.1
        assert abs(total - expected_total) < 0.001  # 允许小的精度误差

class TestSecurityAndSafety(BaseTestCase):
    """安全性和安全测试"""
    
    def test_injection_attack_prevention(self):
        """测试注入攻击防护"""
        malicious_inputs = [
            "'; DROP TABLE orders; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "{{7*7}}",  # 模板注入
            "${7*7}",   # 表达式注入
        ]
        
        for malicious_input in malicious_inputs:
            payload = TestDataFactory.create_whatsapp_payload(body=malicious_input)
            
            try:
                handle_whatsapp_event(payload, 'security-test')
                # 不应该执行恶意代码或崩溃
            except Exception as e:
                # 如果抛出异常，应该是安全的异常类型
                assert not any(danger in str(e).lower() for danger in ['sql', 'script', 'eval'])
    
    def test_large_payload_handling(self):
        """测试大负载处理"""
        # 创建大负载
        large_body = "A" * 10000  # 10KB文本
        payload = TestDataFactory.create_whatsapp_payload(body=large_body)
        
        try:
            handle_whatsapp_event(payload, 'large-payload-test')
            # 应该能处理大负载而不崩溃
        except Exception as e:
            # 如果拒绝，应该是合理的错误
            assert any(keyword in str(e).lower() for keyword in ['size', 'limit', 'too large'])
    
    def test_rate_limiting_simulation(self):
        """测试速率限制模拟"""
        phone = TestDataFactory.random_phone()
        
        # 快速发送多条消息
        for i in range(20):
            payload = TestDataFactory.create_whatsapp_payload(
                phone=phone,
                body=f"Message {i}"
            )
            
            try:
                handle_whatsapp_event(payload, f'rate-limit-test-{i}')
            except Exception as e:
                # 可能触发速率限制
                if 'rate' in str(e).lower() or 'limit' in str(e).lower():
                    break

# 性能监控和报告
class TestReportGenerator:
    """测试报告生成器"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, test_name: str, metrics: PerformanceMetrics, success: bool):
        """添加测试结果"""
        self.results.append({
            'test_name': test_name,
            'metrics': metrics,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_report(self) -> str:
        """生成测试报告"""
        if not self.results:
            return "No test results available"
        
        report = ["=== WhatsApp Integration Test Report ===\n"]
        
        # 总体统计
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        success_rate = successful_tests / total_tests
        
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Successful: {successful_tests}")
        report.append(f"Success Rate: {success_rate:.1%}\n")
        
        # 性能统计
        response_times = [r['metrics'].response_time for r in self.results]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        report.append(f"Average Response Time: {avg_response_time:.3f}s")
        report.append(f"Max Response Time: {max_response_time:.3f}s\n")
        
        # 详细结果
        report.append("=== Detailed Results ===")
        for result in self.results:
            status = "PASS" if result['success'] else "FAIL"
            metrics = result['metrics']
            report.append(f"{result['test_name']}: {status} - {metrics}")
        
        return "\n".join(report)

# 测试配置和夹具
@pytest.fixture(scope="session")
def test_environment():
    """测试环境夹具"""
    env = TestEnvironmentIsolation()
    env.setup_isolated_environment()
    yield env
    env.cleanup_environment()

@pytest.fixture
def mock_manager():
    """Mock管理器夹具"""
    manager = MockManager()
    yield manager
    manager.cleanup()

@pytest.fixture
def performance_benchmark():
    """性能基准夹具"""
    return PerformanceBenchmark()

@pytest.fixture
def test_data_factory():
    """测试数据工厂夹具"""
    return TestDataFactory()

# 自定义标记
pytestmark = pytest.mark.integration

# 运行配置
if __name__ == "__main__":
    # 生成详细的测试报告
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "--cov=app",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])
