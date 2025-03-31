"""日志配置模块"""

import os
import logging
import logging.handlers
from typing import Optional


def setup_logger(name: str, log_file: str, level: int = logging.DEBUG,
                console_level: int = logging.INFO) -> logging.Logger:
    """配置并返回一个命名的日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 文件日志级别
        console_level: 控制台日志级别
        
    Returns:
        配置好的日志记录器
    """
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 获取日志记录器
    logger = logging.getLogger(name)
    
    # 清除已有的处理器，避免重复添加
    if logger.handlers:
        logger.handlers = []
    
    # 设置日志级别
    logger.setLevel(level)
    
    # 禁止日志传播到根记录器
    logger.propagate = False
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s -- %(levelname)s: %(message)s')
    
    # 文件处理器 - 关键修改：指定编码为 utf-8
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when='H', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# 提供一个全局字典来存储已创建的日志记录器
LOGGERS = {}

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """获取或创建命名的日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（首次创建时需要）
        
    Returns:
        日志记录器
    """
    global LOGGERS
    
    if name in LOGGERS:
        return LOGGERS[name]
    
    if log_file is None:
        raise ValueError(f"首次创建日志记录器 '{name}' 时必须提供 log_file 参数")
    
    logger = setup_logger(name, log_file)
    LOGGERS[name] = logger
    
    return logger