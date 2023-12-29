import logging
import os

# 工具类
class Logger:
    
    logger = None
    
    def __init__(self, name : str) -> None:
        Logger.logger = logging.getLogger(name)

    
    # 配置日志信息
    @classmethod
    def init(cls, log_settings):
        # 如果日志目录不存在就创建
        logdir = os.path.dirname(log_settings.path)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        
        f_log_handler = logging.FileHandler(log_settings.path, encoding="utf8")
        Logger.__setLevel(f_log_handler, log_settings.level)

        s_log_handler = logging.StreamHandler()
        Logger.__setLevel(s_log_handler, log_settings.level)

        fmtter = logging.Formatter("[%(name)s] [%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] [%(message)s]")
        f_log_handler.setFormatter(fmtter)
        s_log_handler.setFormatter(fmtter)

        Logger.__setLevel(Logger.logger, log_settings.level)
        Logger.logger.addHandler(f_log_handler)
        Logger.logger.addHandler(s_log_handler)
        
        return Logger.logger

    @classmethod
    def __setLevel(cls, handler, level: str):
        if level.upper() == "INFO":
            handler.setLevel(logging.INFO)
        elif level.upper() == "WARNING":
            handler.setLevel(logging.WARNING)
        elif level.upper() == "CRITICAL":
            handler.setLevel(logging.CRITICAL)
        elif level.upper() == "ERROR":
            handler.setLevel(logging.ERROR)
        else:
            # 默认DEBUG
            handler.setLevel(logging.DEBUG)