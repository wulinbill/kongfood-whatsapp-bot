#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后的 Loyverse POS API 客户端
实现简化的认证流程、智能重试机制、订单状态跟踪和事务处理
"""

import os
import json
import time
import httpx
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import backoff
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import uuid
from ..config import settings
from ..logger import logger

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

class RetryStrategy(Enum):
    """重试策略"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"

@dataclass
class OrderItem:
    """订单项"""
    variant_id: str
    quantity: int
    price: float
    cost: float = 0.0
    note: str = ""
    available_stock: Optional[int] = None
    
    def __post_init__(self):
        """验证订单项数据"""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.price < 0:
            raise ValueError("Price cannot be negative")

@dataclass 
class Payment:
    """支付信息"""
    payment_type_id: str
    money_amount: float
    name: str
    type: str = "CASH"
    
    def __post_init__(self):
        """验证支付数据"""
        if self.money_amount <= 0:
            raise ValueError("Payment amount must be positive")

@dataclass
class OrderTransaction:
    """订单事务信息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    receipt_id: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    rollback_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_status(self, status: OrderStatus, error_message: str = None):
        """更新订单状态"""
        self.status = status
        self.updated_at = datetime.now()
        if error_message:
            self.error_message = error_message

@dataclass
class Order:
    """订单"""
    store_id: str
    line_items: List[OrderItem]
    payments: List[Payment]
    note: str = ""
    source: str = "WhatsApp AI Bot"
    customer_id: Optional[str] = None
    transaction: OrderTransaction = field(default_factory=OrderTransaction)
    
    @property
    def total_amount(self) -> float:
        """计算订单总金额"""
        return sum(item.price * item.quantity for item in self.line_items)
    
    @property
    def total_payment(self) -> float:
        """计算总支付金额"""
        return sum(payment.money_amount for payment in self.payments)
    
    def validate(self) -> bool:
        """验证订单数据"""
        if not self.line_items:
            raise ValueError("Order must have at least one item")
        
        if not self.payments:
            raise ValueError("Order must have at least one payment")
        
        # 验证支付金额与订单总额是否匹配
        if abs(self.total_amount - self.total_payment) > 0.01:
            raise ValueError("Payment amount doesn't match order total")
        
        return True

class TokenManager:
    """简化的 Token 管理器"""
    
    def __init__(self, data_dir: str = "/mnt/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.token_file = self.data_dir / "loyverse_token.json"
        self._token_cache = None
        self._token_lock = asyncio.Lock()
        
        # 配置信息
        self.client_id = os.getenv("LOYVERSE_CLIENT_ID")
        self.client_secret = os.getenv("LOYVERSE_CLIENT_SECRET")
        self.refresh_token = os.getenv("LOYVERSE_REFRESH_TOKEN")
        
        if not all([self.client_id, self.client_secret]):
            logger.warning("Missing Loyverse OAuth credentials")
    
    async def get_access_token(self) -> Optional[str]:
        """获取有效的访问令牌（线程安全）"""
        async with self._token_lock:
            # 检查缓存的 token
            if self._token_cache and self._is_token_valid(self._token_cache):
                return self._token_cache["access_token"]
            
            # 从文件加载 token
            token_data = self._load_token_from_file()
            if token_data and self._is_token_valid(token_data):
                self._token_cache = token_data
                return token_data["access_token"]
            
            # 刷新 token
            new_token = await self._refresh_token()
            if new_token:
                self._token_cache = new_token
                self._save_token_to_file(new_token)
                return new_token["access_token"]
            
            logger.error("Unable to obtain valid access token")
            return None
    
    def _is_token_valid(self, token_data: Dict[str, Any]) -> bool:
        """检查 token 是否有效（提前 5 分钟过期）"""
        if not token_data or "access_token" not in token_data:
            return False
        
        expires_at = token_data.get("expires_at", 0)
        return expires_at - 300 > time.time()  # 提前 5 分钟
    
    def _load_token_from_file(self) -> Optional[Dict[str, Any]]:
        """从文件加载 token"""
        try:
            if self.token_file.exists():
                with open(self.token_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load token from file: {e}")
        return None
    
    def _save_token_to_file(self, token_data: Dict[str, Any]):
        """保存 token 到文件"""
        try:
            token_data["expires_at"] = int(time.time()) + token_data.get("expires_in", 3600)
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f)
        except Exception as e:
            logger.error(f"Failed to save token to file: {e}")
    
    async def _refresh_token(self) -> Optional[Dict[str, Any]]:
        """刷新访问令牌"""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.loyverse.com/oauth/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.refresh_token,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    logger.info("Successfully refreshed access token")
                    return response.json()
                else:
                    logger.error(f"Token refresh failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return None

class APIClient:
    """Loyverse API 客户端，支持智能重试和错误处理"""
    
    def __init__(self):
        self.token_manager = TokenManager()
        self.api_base = "https://api.loyverse.com/v1.0"
        self.default_store_id = settings.STORE_ID
        self._session_cache = {}  # 会话级缓存
    
    @backoff.on_exception(
        backoff.expo,
        (httpx.RequestError, httpx.HTTPStatusError),
        max_tries=3,
        max_time=60,
        jitter=backoff.random_jitter
    )
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        use_cache: bool = False
    ) -> Optional[Dict[str, Any]]:
        """发起 API 请求，支持指数退避重试"""
        
        # 检查缓存
        cache_key = f"{method}:{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        if use_cache and cache_key in self._session_cache:
            cache_entry = self._session_cache[cache_key]
            if cache_entry["expires"] > datetime.now():
                return cache_entry["data"]
        
        access_token = await self.token_manager.get_access_token()
        if not access_token:
            raise Exception("No valid access token available")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        
        async with httpx.AsyncClient() as client:
            response = await self._execute_request(client, method, url, headers, data, params)
            
            if response.status_code in [200, 201]:
                result = response.json() if response.content else {}
                
                # 缓存 GET 请求结果
                if use_cache and method.upper() == "GET":
                    self._session_cache[cache_key] = {
                        "data": result,
                        "expires": datetime.now() + timedelta(minutes=5)
                    }
                
                return result
            elif response.status_code == 401:
                # Token 过期，清除缓存并重试
                self.token_manager._token_cache = None
                raise httpx.HTTPStatusError("Token expired", request=response.request, response=response)
            else:
                logger.error(f"API request failed: {response.status_code} {response.text}")
                raise httpx.HTTPStatusError(f"API request failed: {response.status_code}", 
                                          request=response.request, response=response)
    
    async def _execute_request(self, client, method, url, headers, data, params):
        """执行具体的 HTTP 请求"""
        method = method.upper()
        if method == "GET":
            return await client.get(url, headers=headers, params=params, timeout=30)
        elif method == "POST":
            return await client.post(url, headers=headers, json=data, timeout=30)
        elif method == "PUT":
            return await client.put(url, headers=headers, json=data, timeout=30)
        elif method == "DELETE":
            return await client.delete(url, headers=headers, timeout=30)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    
    async def get_stores(self) -> Optional[List[Dict[str, Any]]]:
        """获取门店列表"""
        return await self._make_request("GET", "/stores", use_cache=True)
    
    async def get_items(self, store_id: str = None) -> Optional[List[Dict[str, Any]]]:
        """获取商品列表，包含库存信息"""
        store_id = store_id or self.default_store_id
        return await self._make_request("GET", f"/stores/{store_id}/items", use_cache=True)
    
    async def get_item_stock(self, item_id: str, store_id: str = None) -> Optional[Dict[str, Any]]:
        """获取商品库存信息"""
        store_id = store_id or self.default_store_id
        return await self._make_request("GET", f"/stores/{store_id}/items/{item_id}/stock")
    
    async def check_stock_availability(self, items: List[OrderItem], store_id: str = None) -> Dict[str, Any]:
        """检查库存可用性"""
        store_id = store_id or self.default_store_id
        stock_status = {"available": True, "issues": []}
        
        for item in items:
            try:
                stock_info = await self.get_item_stock(item.variant_id, store_id)
                if stock_info:
                    available_quantity = stock_info.get("quantity", 0)
                    if available_quantity < item.quantity:
                        stock_status["available"] = False
                        stock_status["issues"].append({
                            "variant_id": item.variant_id,
                            "requested": item.quantity,
                            "available": available_quantity,
                            "message": f"Insufficient stock for item {item.variant_id}"
                        })
                    else:
                        item.available_stock = available_quantity
            except Exception as e:
                logger.warning(f"Could not check stock for item {item.variant_id}: {e}")
        
        return stock_status
    
    async def get_payment_types(self, store_id: str = None) -> Optional[List[Dict[str, Any]]]:
        """获取支付方式"""
        store_id = store_id or self.default_store_id
        return await self._make_request("GET", f"/stores/{store_id}/payment_types", use_cache=True)
    
    async def create_receipt(self, order_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """创建收据"""
        return await self._make_request("POST", "/receipts", data=order_data)
    
    async def get_receipt(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        """获取收据详情"""
        return await self._make_request("GET", f"/receipts/{receipt_id}")
    
    async def cancel_receipt(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        """取消收据"""
        return await self._make_request("DELETE", f"/receipts/{receipt_id}")

class OrderManager:
    """订单管理器，支持事务处理和状态跟踪"""
    
    def __init__(self):
        self.client = APIClient()
        self.active_transactions: Dict[str, OrderTransaction] = {}
        self.transaction_lock = asyncio.Lock()
    
    @asynccontextmanager
    async def order_transaction(self, order: Order):
        """订单事务上下文管理器"""
        async with self.transaction_lock:
            transaction_id = order.transaction.id
            self.active_transactions[transaction_id] = order.transaction
            
            try:
                order.transaction.update_status(OrderStatus.PROCESSING)
                logger.info(f"Started transaction {transaction_id}")
                yield order.transaction
                
                # 如果到这里说明事务成功
                if order.transaction.status == OrderStatus.PROCESSING:
                    order.transaction.update_status(OrderStatus.COMPLETED)
                    
            except Exception as e:
                # 事务失败，执行回滚
                order.transaction.update_status(OrderStatus.FAILED, str(e))
                await self._rollback_transaction(order.transaction)
                raise
            finally:
                # 清理事务记录
                self.active_transactions.pop(transaction_id, None)
                logger.info(f"Transaction {transaction_id} ended with status: {order.transaction.status.value}")
    
    async def _rollback_transaction(self, transaction: OrderTransaction):
        """回滚事务"""
        logger.warning(f"Rolling back transaction {transaction.id}")
        
        for rollback_action in reversed(transaction.rollback_actions):
            try:
                action_type = rollback_action.get("type")
                if action_type == "cancel_receipt":
                    receipt_id = rollback_action.get("receipt_id")
                    if receipt_id:
                        await self.client.cancel_receipt(receipt_id)
                        logger.info(f"Cancelled receipt {receipt_id}")
                
                # 可以添加更多回滚操作类型
                        
            except Exception as e:
                logger.error(f"Rollback action failed: {e}")
    
    async def process_order(self, order: Order, customer_phone: str = None, 
                          customer_name: str = None) -> Dict[str, Any]:
        """处理订单的主要方法"""
        try:
            # 验证订单
            order.validate()
            
            async with self.order_transaction(order) as transaction:
                # 1. 检查库存
                stock_status = await self.client.check_stock_availability(order.line_items, order.store_id)
                if not stock_status["available"]:
                    raise Exception(f"Stock check failed: {stock_status['issues']}")
                
                # 2. 处理客户信息
                if customer_phone and customer_name:
                    customer_id = await self._handle_customer(customer_phone, customer_name)
                    if customer_id:
                        order.customer_id = customer_id
                
                # 3. 准备收据数据
                receipt_data = self._prepare_receipt_data(order)
                
                # 4. 创建收据
                receipt_result = await self.client.create_receipt(receipt_data)
                if not receipt_result:
                    raise Exception("Failed to create receipt")
                
                receipt_id = receipt_result.get("id")
                transaction.receipt_id = receipt_id
                
                # 添加回滚操作
                transaction.rollback_actions.append({
                    "type": "cancel_receipt",
                    "receipt_id": receipt_id
                })
                
                # 5. 验证创建的收据
                receipt_verification = await self.client.get_receipt(receipt_id)
                if not receipt_verification:
                    raise Exception("Receipt verification failed")
                
                return {
                    "success": True,
                    "transaction_id": transaction.id,
                    "receipt_id": receipt_id,
                    "total_amount": order.total_amount,
                    "customer_id": order.customer_id,
                    "order_details": receipt_result,
                    "stock_status": stock_status
                }
                
        except Exception as e:
            logger.error(f"Order processing failed: {e}")
            return {
                "success": False,
                "transaction_id": order.transaction.id,
                "error": str(e),
                "status": order.transaction.status.value
            }
    
    async def _handle_customer(self, phone: str, name: str) -> Optional[str]:
        """处理客户信息（查找或创建）"""
        try:
            # 查找现有客户
            customers = await self.client._make_request("GET", "/customers", params={"phone_number": phone})
            if customers and len(customers) > 0:
                return customers[0]["id"]
            
            # 创建新客户
            customer_data = {"name": name, "phone_number": phone}
            new_customer = await self.client._make_request("POST", "/customers", data=customer_data)
            if new_customer:
                return new_customer["id"]
                
        except Exception as e:
            logger.warning(f"Customer handling failed: {e}")
        
        return None
    
    def _prepare_receipt_data(self, order: Order) -> Dict[str, Any]:
        """准备收据数据"""
        receipt_data = {
            "store_id": order.store_id,
            "line_items": [
                {
                    "variant_id": item.variant_id,
                    "quantity": item.quantity,
                    "price": item.price,
                    "cost": item.cost,
                    "note": item.note
                }
                for item in order.line_items
            ],
            "payments": [
                {
                    "payment_type_id": payment.payment_type_id,
                    "money_amount": payment.money_amount,
                    "name": payment.name,
                    "type": payment.type
                }
                for payment in order.payments
            ],
            "note": order.note,
            "source": order.source
        }
        
        if order.customer_id:
            receipt_data["customer_id"] = order.customer_id
        
        return receipt_data
    
    async def get_order_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        transaction = self.active_transactions.get(transaction_id)
        if not transaction:
            return None
        
        status_info = {
            "transaction_id": transaction_id,
            "status": transaction.status.value,
            "created_at": transaction.created_at.isoformat(),
            "updated_at": transaction.updated_at.isoformat() if transaction.updated_at else None,
            "receipt_id": transaction.receipt_id,
            "error_message": transaction.error_message,
            "retry_count": transaction.retry_count
        }
        
        # 如果有收据ID，获取收据详情
        if transaction.receipt_id:
            try:
                receipt_details = await self.client.get_receipt(transaction.receipt_id)
                status_info["receipt_details"] = receipt_details
            except Exception as e:
                logger.warning(f"Could not fetch receipt details: {e}")
        
        return status_info

# 全局实例
_order_manager = OrderManager()

async def place_order_async(path_data: Dict[str, Any], customer_phone: str = None, 
                           customer_name: str = None) -> Dict[str, Any]:
    """异步下单接口"""
    if not path_data or not path_data.get('path'):
        return {'success': False, 'error': 'Invalid path data'}
    
    try:
        # 构建订单项
        line_items = []
        for item_data in path_data['path']:
            order_item = OrderItem(
                variant_id=item_data['variant_id'],
                quantity=item_data.get('quantity', 1),
                price=item_data.get('price', 0.0),
                note=f"Query: {item_data.get('original_query', '')}"
            )
            line_items.append(order_item)
        
        # 获取支付方式
        payment_types = await _order_manager.client.get_payment_types()
        cash_payment_id = "cash"
        if payment_types:
            for pt in payment_types:
                if pt.get('type', '').upper() == 'CASH':
                    cash_payment_id = pt['id']
                    break
        
        # 计算总金额
        total_amount = sum(item.price * item.quantity for item in line_items)
        
        # 创建支付
        payment = Payment(
            payment_type_id=cash_payment_id,
            money_amount=total_amount,
            name="Cash",
            type="CASH"
        )
        
        # 创建订单
        order = Order(
            store_id=_order_manager.client.default_store_id,
            line_items=line_items,
            payments=[payment],
            note=f"WhatsApp order - Confidence: {path_data.get('confidence', 0.0)}",
            source="WhatsApp AI Bot"
        )
        
        # 处理订单
        return await _order_manager.process_order(order, customer_phone, customer_name)
        
    except Exception as e:
        logger.error(f"Error in place_order_async: {e}")
        return {'success': False, 'error': str(e)}

def place_order(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """同步下单接口"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            
            def run_async():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        place_order_async(
                            order_data.get('path_data', {}),
                            order_data.get('customer_phone'),
                            order_data.get('customer_name')
                        )
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(
                place_order_async(
                    order_data.get('path_data', {}),
                    order_data.get('customer_phone'),
                    order_data.get('customer_name')
                )
            )
            
    except Exception as e:
        logger.error(f"Error in place_order: {e}")
        return {'success': False, 'error': str(e)}

async def get_order_status_async(transaction_id: str) -> Optional[Dict[str, Any]]:
    """获取订单状态"""
    return await _order_manager.get_order_status(transaction_id)

async def get_menu_items_async(store_id: str = None) -> Optional[List[Dict[str, Any]]]:
    """获取菜单项"""
    return await _order_manager.client.get_items(store_id)

async def get_stores_async() -> Optional[List[Dict[str, Any]]]:
    """获取门店列表"""
    return await _order_manager.client.get_stores()

# 测试函数
if __name__ == "__main__":
    async def test_optimized_client():
        """测试优化后的客户端"""
        print("Testing optimized Loyverse client...")
        
        # 测试门店获取
        stores = await get_stores_async()
        print(f"Stores: {len(stores) if stores else 0}")
        
        # 测试订单处理
        test_path_data = {
            'path': [
                {
                    'variant_id': 'test-variant-id',
                    'price': 15.99,
                    'quantity': 2,
                    'original_query': 'Chicken Teriyaki'
                }
            ],
            'confidence': 0.9
        }
        
        result = await place_order_async(
            test_path_data,
            customer_phone="+1234567890",
            customer_name="Test User"
        )
        
        print(f"Order result: {result}")
        
        # 如果订单成功，测试状态查询
        if result.get('success') and result.get('transaction_id'):
            status = await get_order_status_async(result['transaction_id'])
            print(f"Order status: {status}")
    
    asyncio.run(test_optimized_client())
