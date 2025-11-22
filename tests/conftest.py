"""Pytest配置和共享fixture"""
import pytest
import sys
import os

# 添加src到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_config():
    """示例配置fixture"""
    return {"test_mode": True, "debug": False}
