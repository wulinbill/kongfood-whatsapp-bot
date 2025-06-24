#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试套件
测试从语音输入到POS系统下单的完整流程
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

# 导入应用模块
from app.main import app
from app.oco_core.seed_parser import parse
from app.oco_core.jump_planner import plan
from app.oco_core.tension_eval import score
from app.oco_core.output_director import reply
from app.pos.loyverse_client import place_order_async
from app.speech.deepgram_client import transcribe_async
from app.whatsapp.router import handle_whatsapp_event

class TestHealthCheck:
    """健康检查测试"""
    
    def test_health_endpoint(self):
        """测试健康检查端点"""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "live"}

class TestSeedParser:
    """种子解析器测试"""
    
    def test_parse_spanish_order(self):
        """测试西班牙语订单解析"""
        text = "Quiero 2 Pollo Teriyaki"
        result = parse(text)
        
        assert result['intent'] == 'order'
        assert result['language'] == 'es'
        assert len(result['objects']) > 0
        assert result['confidence'] > 0.5
    
    def test_parse_english_order(self):
        """测试英语订单解析"""
        text = "I want chicken teriyaki"
        result = parse(text)
        
        assert result['intent'] == 'order'
        assert result['language'] == 'en'
        assert len(result['objects']) > 0
    
    def test_parse_chinese_order(self):
        """测试中文订单解析"""
        text = "我要照烧鸡肉"
        result = parse(text)
        
        assert result['intent'] == 'order'
        assert result['language'] == 'zh'
        assert len(result['objects']) > 0
    
    def test_parse_modification(self):
        """测试修改解析"""
        text = "Pollo Teriyaki sin salsa"
        result = parse(text)
        
        assert result['intent'] == 'order'
        modifications = [obj for obj in result['objects'] if obj['item_type'] == 'remove']
        assert len(modifications) > 0
    
    def test_parse_cancel_intent(self):
        """测试取消意图"""
        text = "cancelar orden"
        result = parse(text)
        
        assert result['intent'] == 'cancel'

class TestJumpPlanner:
    """跳跃规划器测试"""
    
    def test_plan_single_match(self):
        """测试单一匹配规划"""
        co = {
            'objects': [
                {
                    'item_type': 'main_dish',
                    'content': 'Pollo Teriyaki',
                    'quantity': 1,
                    'confidence': 0.9
                }
            ],
            'intent': 'order',
            'confidence': 0.8
        }
        
        result = plan(co)
        
        assert 'path' in result
        assert result['score'] > 0
        assert 'requires_clarification' in result
    
    def test_plan_no_matches(self):
        """测试无匹配规划"""
        co = {
            'objects': [
                {
                    'item_type': 'main_dish',
                    'content': 'NonexistentDish',
                    'quantity': 1,
                    'confidence': 0.1
                }
            ],
            'intent': 'order',
            'confidence': 0.3
        }
        
        result = plan(co)
        
        assert result['requires_clarification'] == True
        assert result['clarification_reason'] in ['no_menu_matches', 'low_confidence']

class TestTensionEvaluator:
    """张力评估器测试"""
    
    def test_high_confidence_score(self):
        """测试高置信度分数"""
        path_data = {
            'path': [{'item_name': 'Pollo Teriyaki', 'price': 11.99}],
            'score': 0.9,
            'confidence': 0.9,
            'requires_clarification': False
        }
        
        tension_score = score(path_data)
        
        assert tension_score > 0.7
        assert isinstance(tension_score, float)
    
    def test_low_confidence_score(self):
        """测试低置信度分数"""
        path_data = {
            'path': [{'item_name': 'Unknown Item', 'price': 0}],
            'score': 0.2,
            'confidence': 0.2,
            'requires_clarification': True
        }
        
        tension_score = score(path_data)
        
        assert tension_score < 0.5

class TestOutputDirector:
    """输出决策器测试"""
    
    @patch('app.pos.loyverse_client.place_order')
    def test_reply_order_execution(self, mock_place_order):
        """测试订单执行响应"""
        mock_place_order.return_value = {
            'success': True,
            'receipt_id': 'test-receipt-123',
            'total_amount': 13.30
        }
        
        co = {
            'objects': [
                {
                    'item_type': 'main_dish',
                    'content': 'Pollo Teriyaki',
                    'quantity': 1,
                    'confidence': 0.9
                }
            ],
            'intent': 'order',
            'language': 'es',
            'confidence': 0.9,
            'customer_name': 'Test Customer'
        }
        
        path_data = {
            'path': [
                {
                    'item_name': 'Pollo Teriyaki',
                    'variant_id': 'test-variant-id',
                    'price': 11.99,
                    'quantity': 1
                }
            ],
            'score': 0.9,
            'confidence': 0.9,
            'requires_clarification': False
        }
        
        response = reply(co, path_data)
        
        assert 'Gracias' in response or 'confirmo' in response.lower()
        assert '13.30' in response or '11.99' in response
        mock_place_order.assert_called_once()
    
    def test_reply_clarification_needed(self):
        """测试需要澄清的响应"""
        co = {
            'objects': [
                {
                    'item_type': 'main_dish',
                    'content': 'pollo',
                    'quantity': 1,
                    'confidence': 0.4
                }
            ],
            'intent': 'order',
            'language': 'es',
            'confidence': 0.4
        }
        
        path_data = {
            'path': [],
            'score': 0.3,
            'confidence': 0.3,
            'requires_clarification': True,
            'clarification_reason': 'multiple_matches',
            'alternative_paths': [
                {
                    'matches': [
                        {'item_name': 'Pollo Teriyaki', 'price': 11.99},
                        {'item_name': 'Pollo Naranja', 'price': 11.89}
                    ]
                }
            ]
        }
        
        response = reply(co, path_data)
        
        assert 'opciones' in response.lower() or 'cuál' in response.lower()

class TestSpeechTranscription:
    """语音转录测试"""
    
    @pytest.mark.asyncio
    @patch('app.speech.deepgram_client.DeepgramClient.transcribe_audio_url')
    async def test_transcribe_spanish_audio(self, mock_transcribe):
        """测试西班牙语音频转录"""
        mock_transcribe.return_value = ("quiero pollo teriyaki", "es")
        
        text, language = await transcribe_async("http://example.com/audio.ogg")
        
        assert text == "quiero pollo teriyaki"
        assert language == "es"

class TestWhatsAppIntegration:
    """WhatsApp集成测试"""
    
    @patch('app.whatsapp.twilio_adapter.send_message')
    @patch('app.pos.loyverse_client.place_order')
    def test_complete_order_flow(self, mock_place_order, mock_send_message):
        """测试完整订单流程"""
        mock_place_order.return_value = {
            'success': True,
            'receipt_id': 'test-receipt-123'
        }
        
        # 模拟WhatsApp消息
        payload = {
            'From': 'whatsapp:+1234567890',
            'Body': 'Quiero Pollo Teriyaki',
            'MessageSid': 'test-message-123'
        }
        
        # 处理消息
        handle_whatsapp_event(payload, 'test-trace-id')
        
        # 验证发送了响应
        mock_send_message.assert_called()
        
        # 获取发送的消息
        args, kwargs = mock_send_message.call_args
        phone_number = args[0]
        message = args[1]
        
        assert phone_number == 'whatsapp:+1234567890'
        assert isinstance(message, str)
        assert len(message) > 0
    
    @patch('app.whatsapp.twilio_adapter.send_message')
    def test_clarification_flow(self, mock_send_message):
        """测试澄清流程"""
        # 模拟需要澄清的消息
        payload = {
            'From': 'whatsapp:+1234567890',
            'Body': 'pollo',  # 模糊的订单
            'MessageSid': 'test-message-124'
        }
        
        handle_whatsapp_event(payload, 'test-trace-id')
        
        # 验证发送了澄清消息
        mock_send_message.assert_called()
        args, kwargs = mock_send_message.call_args
        message = args[1]
        
        # 澄清消息应该包含选项或问题
        assert any(keyword in message.lower() for keyword in ['opciones', 'cuál', 'qué'])

class TestLoyversePOSIntegration:
    """Loyverse POS集成测试"""
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_place_order_success(self, mock_post):
        """测试成功下单"""
        # 模拟成功的API响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'receipt-123',
            'total': 13.30,
            'status': 'completed'
        }
        mock_post.return_value = mock_response
        
        path_data = {
            'path': [
                {
                    'variant_id': 'test-variant-id',
                    'price': 11.99,
                    'quantity': 1,
                    'original_query': 'Pollo Teriyaki'
                }
            ],
            'confidence': 0.9
        }
        
        result = await place_order_async(
            path_data,
            customer_phone="+1234567890",
            customer_name="Test Customer"
        )
        
        assert result['success'] == True
        assert result['receipt_id'] == 'receipt-123'
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_place_order_failure(self, mock_post):
        """测试下单失败"""
        # 模拟失败的API响应
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid request"
        mock_post.return_value = mock_response
        
        path_data = {
            'path': [
                {
                    'variant_id': 'invalid-id',
                    'price': 11.99,
                    'quantity': 1
                }
            ]
        }
        
        result = await place_order_async(path_data)
        
        assert result['success'] == False
        assert 'error' in result

class TestEndToEndScenarios:
    """端到端场景测试"""
    
    @patch('app.whatsapp.twilio_adapter.send_message')
    @patch('app.pos.loyverse_client.place_order')
    @patch('app.speech.deepgram_client.transcribe_whatsapp')
    def test_voice_order_complete_flow(self, mock_transcribe, mock_place_order, mock_send_message):
        """测试语音订单完整流程"""
        # 设置模拟
        mock_transcribe.return_value = ("quiero dos pollo teriyaki", "es")
        mock_place_order.return_value = {
            'success': True,
            'receipt_id': 'receipt-456',
            'total_amount': 26.60
        }
        
        # 模拟语音消息
        payload = {
            'From': 'whatsapp:+1234567890',
            'MediaUrl0': 'https://example.com/audio.ogg',
            'MediaContentType0': 'audio/ogg',
            'MessageSid': 'test-audio-message'
        }
        
        handle_whatsapp_event(payload, 'test-voice-trace')
        
        # 验证转录被调用
        mock_transcribe.assert_called_once()
        
        # 验证最终发送了确认消息
        mock_send_message.assert_called()
        args, kwargs = mock_send_message.call_args
        message = args[1]
        
        # 确认消息应该包含订单详情
        assert any(keyword in message.lower() for keyword in ['confirmo', 'total', 'gracias'])
    
    def test_multilingual_support(self):
        """测试多语言支持"""
        test_cases = [
            ("Quiero pollo teriyaki", "es"),
            ("I want chicken teriyaki", "en"),
            ("我要照烧鸡肉", "zh")
        ]
        
        for text, expected_lang in test_cases:
            result = parse(text)
            assert result['language'] == expected_lang
            assert result['intent'] == 'order'
    
    @patch('app.whatsapp.twilio_adapter.send_message')
    def test_error_handling(self, mock_send_message):
        """测试错误处理"""
        # 发送无效payload
        invalid_payload = {
            'InvalidField': 'InvalidValue'
        }
        
        # 应该不会崩溃
        handle_whatsapp_event(invalid_payload, 'test-error-trace')
        
        # 可能会发送错误消息或忽略
        # 这取决于具体的错误处理策略

class TestMenuMatching:
    """菜单匹配测试"""
    
    def test_exact_match(self):
        """测试精确匹配"""
        co = {
            'objects': [
                {
                    'item_type': 'main_dish',
                    'content': 'Pollo Teriyaki',
                    'quantity': 1,
                    'confidence': 0.9
                }
            ],
            'intent': 'order',
            'confidence': 0.9
        }
        
        result = plan(co)
        
        if result['path']:
            # 如果找到匹配，分数应该很高
            assert result['score'] > 0.8
            assert not result['requires_clarification']
    
    def test_fuzzy_match(self):
        """测试模糊匹配"""
        co = {
            'objects': [
                {
                    'item_type': 'main_dish',
                    'content': 'teriyaki chicken',  # 英文描述
                    'quantity': 1,
                    'confidence': 0.7
                }
            ],
            'intent': 'order',
            'confidence': 0.7
        }
        
        result = plan(co)
        
        # 应该能找到某种匹配
        assert 'path' in result
        assert isinstance(result['score'], float)
    
    def test_modification_parsing(self):
        """测试修改解析"""
        co = {
            'objects': [
                {
                    'item_type': 'main_dish',
                    'content': 'Pollo Teriyaki',
                    'quantity': 1,
                    'confidence': 0.8
                },
                {
                    'item_type': 'remove',
                    'content': 'salsa',
                    'quantity': 1,
                    'confidence': 0.7
                }
            ],
            'intent': 'order',
            'confidence': 0.8
        }
        
        result = plan(co)
        
        # 应该包含主菜和修改
        assert 'path' in result

class TestPerformanceAndReliability:
    """性能和可靠性测试"""
    
    def test_concurrent_requests(self):
        """测试并发请求处理"""
        import threading
        import time
        
        results = []
        
        def process_order():
            co = {
                'objects': [
                    {
                        'item_type': 'main_dish',
                        'content': 'Pollo Teriyaki',
                        'quantity': 1,
                        'confidence': 0.9
                    }
                ],
                'intent': 'order',
                'confidence': 0.9
            }
            
            start_time = time.time()
            result = plan(co)
            end_time = time.time()
            
            results.append({
                'result': result,
                'duration': end_time - start_time
            })
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_order)
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(results) == 5
        for result_data in results:
            assert 'result' in result_data
            assert result_data['duration'] < 5.0  # 5秒超时
    
    def test_large_input_handling(self):
        """测试大输入处理"""
        # 生成长文本
        long_text = "Quiero " + "pollo teriyaki " * 50
        
        result = parse(long_text)
        
        # 应该能正确解析，不会崩溃
        assert 'intent' in result
        assert 'objects' in result
    
    def test_memory_usage(self):
        """测试内存使用"""
        import gc
        import sys
        
        # 记录初始内存
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # 执行多次解析
        for i in range(100):
            co = parse(f"Quiero pollo teriyaki numero {i}")
            plan(co)
        
        # 强制垃圾回收
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # 内存增长应该在合理范围内
        memory_growth = final_objects - initial_objects
        assert memory_growth < 1000  # 允许一些合理的内存增长

# 配置pytest fixture
@pytest.fixture
def client():
    """FastAPI测试客户端"""
    return TestClient(app)

@pytest.fixture
def sample_whatsapp_payload():
    """样例WhatsApp负载"""
    return {
        'From': 'whatsapp:+1234567890',
        'Body': 'Quiero Pollo Teriyaki',
        'MessageSid': 'SMtest123456789'
    }

@pytest.fixture
def sample_co():
    """样例对话对象"""
    return {
        'objects': [
            {
                'item_type': 'main_dish',
                'content': 'Pollo Teriyaki',
                'quantity': 1,
                'confidence': 0.9
            }
        ],
        'intent': 'order',
        'language': 'es',
        'confidence': 0.9,
        'raw_text': 'Quiero Pollo Teriyaki'
    }

@pytest.fixture
def sample_path_data():
    """样例路径数据"""
    return {
        'path': [
            {
                'item_id': 'test-item-id',
                'item_name': 'Pollo Teriyaki',
                'variant_id': 'test-variant-id',
                'price': 11.99,
                'quantity': 1,
                'match_score': 0.95,
                'original_query': 'Pollo Teriyaki'
            }
        ],
        'score': 0.9,
        'confidence': 0.9,
        'requires_clarification': False
    }

# 运行测试的主函数
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
