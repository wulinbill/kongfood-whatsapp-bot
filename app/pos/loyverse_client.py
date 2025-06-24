#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loyverse POS API 客户端
实现OAuth2认证和订单管理
"""

import os
import json
import time
import httpx
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from ..config import settings
from ..logger import logger

@dataclass
class OrderItem:
    """订单项"""
    variant_id: str
    quantity: int
    price: float
    cost: float = 0.0
    note: str = ""

@dataclass 
class Payment:
    """支付信息"""
    payment_type_id: str
    money_amount: float
    name: str
    type: str = "CASH"

@dataclass
class Order:
    """订单"""
    store_id: str
    line_items: List[OrderItem]
    payments: List[Payment]
    note: str = ""
    source: str = "WhatsApp AI Bot"
    customer_id: Optional[str] = None

class LoyverseAuthManager:
    """Loyverse OAuth2 认证管理器"""
    
    def __init__(self, data_dir: str = "/mnt/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.token_file = self.data_dir / "loyverse_token.json"
        
        # OAuth2 配置
        self.client_id = os.getenv("LOYVERSE_CLIENT_ID")
        self.client_secret = os.getenv("LOYVERSE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("LOYVERSE_REDIRECT_URI", "http://localhost:8000/oauth/callback")
        
        # API端点
        self.auth_url = "https://api.loyverse.com/oauth/authorize"
        self.token_url = "https://api.loyverse.com/oauth/token"
        self.api_base = "https://api.loyverse.com/v1.0"
        
        if not self.client_id or not self.client_secret:
            logger.warning("Loyverse OAuth credentials not found in environment")
    
    def _save_token(self, token_data: Dict[str, Any]):
        """保存token到文件"""
        try:
            token_data["expires_at"] = int(time.time()) + token_data.get("expires_in", 3600)
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f)
            logger.info("Token saved successfully")
        except Exception as e:
            logger.error(f"Failed to save token: {e}")
    
    def _load_token(self) -> Optional[Dict[str, Any]]:
        """从文件加载token"""
        try:
            if self.token_file.exists():
                with open(self.token_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load token: {e}")
        return None
    
    def get_authorization_url(self, state: str = None) -> str:
        """获取授权URL"""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "read write"
        }
        if state:
            params["state"] = state
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.auth_url}?{query_string}"
    
    async def exchange_code_for_token(self, authorization_code: str) -> bool:
        """用授权码换取访问令牌"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data={
                        "grant_type": "authorization_code",
                        "code": authorization_code,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "redirect_uri": self.redirect_uri
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self._save_token(token_data)
                    logger.info("Successfully exchanged authorization code for token")
                    return True
                else:
                    logger.error(f"Token exchange failed: {response.status_code} {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error during token exchange: {e}")
            return False
    
    async def refresh_access_token(self) -> Optional[str]:
        """刷新访问令牌"""
        token_data = self._load_token()
        refresh_token = (token_data and token_data.get("refresh_token")) or os.getenv("LOYVERSE_REFRESH_TOKEN")
        
        if not refresh_token:
            logger.error("No refresh token available")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    new_token_data = response.json()
                    self._save_token(new_token_data)
                    logger.info("Successfully refreshed access token")
                    return new_token_data["access_token"]
                else:
                    logger.error(f"Token refresh failed: {response.status_code} {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error during token refresh: {e}")
            return None
    
    async def get_valid_access_token(self) -> Optional[str]:
        """获取有效的访问令牌"""
        token_data = self._load_token()
        
        if token_data:
            # 检查token是否过期（提前60秒刷新）
            if token_data.get("expires_at", 0) - 60 > time.time():
                return token_data["access_token"]
        
        # Token过期或不存在，尝试刷新
        return await self.refresh_access_token()

class LoyverseAPIClient:
    """Loyverse API 客户端"""
    
    def __init__(self):
        self.auth_manager = LoyverseAuthManager()
        self.api_base = "https://api.loyverse.com/v1.0"
        self.default_store_id = settings.STORE_ID
    
    async def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None, 
                           params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """发起API请求"""
        access_token = await self.auth_manager.get_valid_access_token()
        if not access_token:
            logger.error("No valid access token available")
            return None
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        
        try:
            async with httpx.AsyncClient() as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers, params=params, timeout=30)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, json=data, timeout=30)
                elif method.upper() == "PUT":
                    response = await client.put(url, headers=headers, json=data, timeout=30)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers, timeout=30)
                else:
                    logger.error(f"Unsupported HTTP method: {method}")
                    return None
                
                if response.status_code in [200, 201]:
                    return response.json() if response.content else {}
                elif response.status_code == 401:
                    # Token可能过期，尝试刷新
                    logger.warning("Received 401, attempting token refresh")
                    new_token = await self.auth_manager.refresh_access_token()
                    if new_token:
                        # 重试请求
                        headers["Authorization"] = f"Bearer {new_token}"
                        if method.upper() == "POST":
                            retry_response = await client.post(url, headers=headers, json=data, timeout=30)
                        else:
                            retry_response = await client.get(url, headers=headers, params=params, timeout=30)
                        
                        if retry_response.status_code in [200, 201]:
                            return retry_response.json() if retry_response.content else {}
                
                logger.error(f"API request failed: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error making API request: {e}")
            return None
    
    async def get_stores(self) -> Optional[List[Dict[str, Any]]]:
        """获取门店列表"""
        return await self._make_request("GET", "/stores")
    
    async def get_items(self, store_id: str = None) -> Optional[List[Dict[str, Any]]]:
        """获取商品列表"""
        store_id = store_id or self.default_store_id
        return await self._make_request("GET", f"/stores/{store_id}/items")
    
    async def get_categories(self, store_id: str = None) -> Optional[List[Dict[str, Any]]]:
        """获取商品分类"""
        store_id = store_id or self.default_store_id
        return await self._make_request("GET", f"/stores/{store_id}/categories")
    
    async def create_customer(self, name: str, phone: str = None, email: str = None) -> Optional[Dict[str, Any]]:
        """创建客户"""
        customer_data = {
            "name": name,
            "phone_number": phone,
            "email": email
        }
        # 移除空值
        customer_data = {k: v for k, v in customer_data.items() if v is not None}
        
        return await self._make_request("POST", "/customers", data=customer_data)
    
    async def find_customer_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """根据电话号码查找客户"""
        customers = await self._make_request("GET", "/customers", params={"phone_number": phone})
        if customers and len(customers) > 0:
            return customers[0]
        return None
    
    async def create_receipt(self, order_data: Order) -> Optional[Dict[str, Any]]:
        """创建收据（下单）"""
        receipt_data = {
            "store_id": order_data.store_id,
            "line_items": [
                {
                    "variant_id": item.variant_id,
                    "quantity": item.quantity,
                    "price": item.price,
                    "cost": item.cost,
                    "note": item.note
                }
                for item in order_data.line_items
            ],
            "payments": [
                {
                    "payment_type_id": payment.payment_type_id,
                    "money_amount": payment.money_amount,
                    "name": payment.name,
                    "type": payment.type
                }
                for payment in order_data.payments
            ],
            "note": order_data.note,
            "source": order_data.source
        }
        
        if order_data.customer_id:
            receipt_data["customer_id"] = order_data.customer_id
        
        return await self._make_request("POST", "/receipts", data=receipt_data)
    
    async def get_payment_types(self, store_id: str = None) -> Optional[List[Dict[str, Any]]]:
        """获取支付方式"""
        store_id = store_id or self.default_store_id
        return await self._make_request("GET", f"/stores/{store_id}/payment_types")
    
    async def get_receipt(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        """获取收据详情"""
        return await self._make_request("GET", f"/receipts/{receipt_id}")

class OrderProcessor:
    """订单处理器"""
    
    def __init__(self):
        self.client = LoyverseAPIClient()
    
    async def process_order_from_path(self, path_data: Dict[str, Any], customer_phone: str = None, 
                                    customer_name: str = None) -> Optional[Dict[str, Any]]:
        """从路径数据处理订单"""
        if not path_data or not path_data.get('path'):
            logger.error("Invalid path data for order processing")
            return None
        
        store_id = self.client.default_store_id
        line_items = []
        total_amount = 0.0
        
        # 转换路径数据为订单项
        for item_data in path_data['path']:
            quantity = item_data.get('quantity', 1)
            price = item_data.get('price', 0.0)
            
            order_item = OrderItem(
                variant_id=item_data['variant_id'],
                quantity=quantity,
                price=price,
                cost=0.0,
                note=f"Original query: {item_data.get('original_query', '')}"
            )
            line_items.append(order_item)
            total_amount += price * quantity
        
        # 处理客户信息
        customer_id = None
        if customer_phone and customer_name:
            # 查找或创建客户
            existing_customer = await self.client.find_customer_by_phone(customer_phone)
            if existing_customer:
                customer_id = existing_customer['id']
                logger.info(f"Found existing customer: {customer_id}")
            else:
                new_customer = await self.client.create_customer(customer_name, customer_phone)
                if new_customer:
                    customer_id = new_customer['id']
                    logger.info(f"Created new customer: {customer_id}")
        
        # 获取支付方式（默认使用现金）
        payment_types = await self.client.get_payment_types(store_id)
        cash_payment_type = "cash"  # 默认值
        
        if payment_types:
            for pt in payment_types:
                if pt.get('type', '').upper() == 'CASH':
                    cash_payment_type = pt['id']
                    break
        
        # 创建支付信息
        payment = Payment(
            payment_type_id=cash_payment_type,
            money_amount=total_amount,
            name="Cash",
            type="CASH"
        )
        
        # 创建订单
        order = Order(
            store_id=store_id,
            line_items=line_items,
            payments=[payment],
            note=f"WhatsApp order - Confidence: {path_data.get('confidence', 0.0)}",
            source="WhatsApp AI Bot",
            customer_id=customer_id
        )
        
        # 提交订单
        try:
            result = await self.client.create_receipt(order)
            if result:
                logger.info(f"Order created successfully: {result.get('id')}")
                return {
                    'success': True,
                    'receipt_id': result.get('id'),
                    'total_amount': total_amount,
                    'customer_id': customer_id,
                    'order_details': result
                }
            else:
                logger.error("Failed to create order")
                return {
                    'success': False,
                    'error': 'Failed to submit order to POS system'
                }
                
        except Exception as e:
            logger.error(f"Error processing order: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# 全局客户端和处理器实例
_client = LoyverseAPIClient()
_processor = OrderProcessor()

def place_order(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    同步接口 - 下单
    
    Args:
        order_data: 订单数据
        
    Returns:
        订单结果
    """
    try:
        # 如果在异步环境中运行
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 创建新的事件循环在线程中运行
            import concurrent.futures
            import threading
            
            def run_async():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        _processor.process_order_from_path(
                            order_data.get('path_data', {}),
                            order_data.get('customer_phone'),
                            order_data.get('customer_name')
                        )
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                result = future.result(timeout=30)
                return result or {'success': False, 'error': 'Order processing failed'}
        else:
            # 直接在当前事件循环中运行
            return loop.run_until_complete(
                _processor.process_order_from_path(
                    order_data.get('path_data', {}),
                    order_data.get('customer_phone'),
                    order_data.get('customer_name')
                )
            ) or {'success': False, 'error': 'Order processing failed'}
            
    except Exception as e:
        logger.error(f"Error in place_order: {e}")
        return {
            'success': False,
            'error': str(e)
        }

async def place_order_async(path_data: Dict[str, Any], customer_phone: str = None, 
                           customer_name: str = None) -> Dict[str, Any]:
    """
    异步接口 - 下单
    
    Args:
        path_data: 路径数据
        customer_phone: 客户电话
        customer_name: 客户姓名
        
    Returns:
        订单结果
    """
    return await _processor.process_order_from_path(path_data, customer_phone, customer_name)

async def get_menu_items_async(store_id: str = None) -> Optional[List[Dict[str, Any]]]:
    """异步获取菜单项"""
    return await _client.get_items(store_id)

async def get_stores_async() -> Optional[List[Dict[str, Any]]]:
    """异步获取门店列表"""
    return await _client.get_stores()

def get_authorization_url(state: str = None) -> str:
    """获取OAuth授权URL"""
    return _client.auth_manager.get_authorization_url(state)

async def handle_oauth_callback(authorization_code: str) -> bool:
    """处理OAuth回调"""
    return await _client.auth_manager.exchange_code_for_token(authorization_code)


# 测试函数
if __name__ == "__main__":
    import asyncio
    
    async def test_loyverse_client():
        """测试Loyverse客户端"""
        print("Testing Loyverse API Client...")
        
        # 测试获取门店
        stores = await get_stores_async()
        if stores:
            print(f"Found {len(stores)} stores")
        else:
            print("No stores found or authentication failed")
        
        # 测试订单处理
        test_path_data = {
            'path': [
                {
                    'variant_id': 'test-variant-id',
                    'price': 11.99,
                    'quantity': 1,
                    'original_query': 'Pollo Teriyaki'
                }
            ],
            'confidence': 0.8
        }
        
        result = await place_order_async(
            test_path_data,
            customer_phone="+1234567890",
            customer_name="Test Customer"
        )
        
        print(f"Order result: {result}")
    
    # 运行测试
    asyncio.run(test_loyverse_client())
