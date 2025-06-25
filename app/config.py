import os
from typing import Optional
from pathlib import Path
from pydantic import BaseSettings, validator, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Settings(BaseSettings):
    """
    应用程序配置设置
    
    使用 Pydantic BaseSettings 进行配置验证和管理
    支持从环境变量、.env 文件和默认值加载配置
    """
    
    # API 密钥配置
    ANTHROPIC_API_KEY: str = Field(
        ..., 
        description="Anthropic API 密钥，用于 Claude AI 服务"
    )
    DEEPGRAM_API_KEY: str = Field(
        ..., 
        description="Deepgram API 密钥，用于语音转文字服务"
    )
    
    # WhatsApp 配置
    WHATSAPP_VENDOR: str = Field(
        default="twilio",
        description="WhatsApp 服务提供商"
    )
    
    # 商店配置
    STORE_ID: str = Field(
        default="kong-food-store-1",
        description="商店唯一标识符"
    )
    
    # 知识库配置
    MENU_KB_FILE: Optional[str] = Field(
        default=None,
        description="菜单知识库文件路径"
    )
    
    # 环境配置
    ENVIRONMENT: str = Field(
        default="development",
        description="运行环境: development, staging, production"
    )
    
    # 日志配置
    LOG_LEVEL: str = Field(
        default="INFO",
        description="日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    
    # 应用配置
    DEBUG: bool = Field(
        default=False,
        description="调试模式开关"
    )
    
    # 数据库配置（如果需要）
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="数据库连接 URL"
    )
    
    # 缓存配置
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis 缓存服务器 URL"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @validator("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY")
    def validate_api_keys(cls, v, field):
        """验证 API 密钥不能为空"""
        if not v or v.strip() == "":
            raise ValueError(f"{field.name} 不能为空")
        return v.strip()
    
    @validator("WHATSAPP_VENDOR")
    def validate_whatsapp_vendor(cls, v):
        """验证 WhatsApp 服务商"""
        allowed_vendors = ["twilio", "whatsapp_business", "messagebird"]
        if v.lower() not in allowed_vendors:
            raise ValueError(f"WHATSAPP_VENDOR 必须是以下之一: {', '.join(allowed_vendors)}")
        return v.lower()
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """验证运行环境"""
        allowed_envs = ["development", "staging", "production"]
        if v.lower() not in allowed_envs:
            raise ValueError(f"ENVIRONMENT 必须是以下之一: {', '.join(allowed_envs)}")
        return v.lower()
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """验证日志级别"""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"LOG_LEVEL 必须是以下之一: {', '.join(allowed_levels)}")
        return v.upper()
    
    @validator("MENU_KB_FILE", always=True)
    def set_menu_kb_file(cls, v, values):
        """设置菜单知识库文件路径"""
        if v is None:
            store_id = values.get("STORE_ID", "kong-food-store-1")
            return f"app/knowledge_base/{store_id}_menu_kb.json"
        return v
    
    @validator("DEBUG", always=True)
    def set_debug_mode(cls, v, values):
        """根据环境设置调试模式"""
        if values.get("ENVIRONMENT") == "development":
            return True
        return v
    
    def validate_file_paths(self) -> None:
        """验证文件路径是否存在"""
        if self.MENU_KB_FILE:
            kb_path = Path(self.MENU_KB_FILE)
            kb_dir = kb_path.parent
            if not kb_dir.exists():
                kb_dir.mkdir(parents=True, exist_ok=True)
                print(f"创建知识库目录: {kb_dir}")
    
    def get_kb_file_path(self) -> Path:
        """获取知识库文件路径对象"""
        return Path(self.MENU_KB_FILE)
    
    def is_production(self) -> bool:
        """检查是否为生产环境"""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """检查是否为开发环境"""
        return self.ENVIRONMENT == "development"


def load_settings() -> Settings:
    """
    加载并验证配置设置
    
    Returns:
        Settings: 验证后的配置对象
        
    Raises:
        ValueError: 配置验证失败时抛出异常
    """
    try:
        settings = Settings()
        
        # 验证文件路径
        settings.validate_file_paths()
        
        # 在开发环境下打印配置信息（隐藏敏感信息）
        if settings.is_development():
            print("=== 配置加载成功 ===")
            print(f"环境: {settings.ENVIRONMENT}")
            print(f"调试模式: {settings.DEBUG}")
            print(f"日志级别: {settings.LOG_LEVEL}")
            print(f"商店ID: {settings.STORE_ID}")
            print(f"WhatsApp服务商: {settings.WHATSAPP_VENDOR}")
            print(f"知识库文件: {settings.MENU_KB_FILE}")
            print(f"Anthropic API Key: {'*' * (len(settings.ANTHROPIC_API_KEY) - 4) + settings.ANTHROPIC_API_KEY[-4:]}")
            print(f"Deepgram API Key: {'*' * (len(settings.DEEPGRAM_API_KEY) - 4) + settings.DEEPGRAM_API_KEY[-4:]}")
            print("==================")
        
        return settings
        
    except Exception as e:
        print(f"配置加载失败: {str(e)}")
        print("请检查以下配置:")
        print("1. .env 文件是否存在")
        print("2. ANTHROPIC_API_KEY 是否设置")
        print("3. DEEPGRAM_API_KEY 是否设置")
        print("4. 其他必需的环境变量是否正确设置")
        raise


# 创建全局配置实例
settings = load_settings()
