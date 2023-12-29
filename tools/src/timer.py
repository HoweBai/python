import time


# 计时
class Timer:
    def __init__(self, title: str, indicator: str = "TIME_COST") -> None:
        """初始化计时器

        Args:
            title (str): 本次计时的标题
            indicator (str, optional): 本轮计时的标题. Defaults to "TIME_COST".
        """
        self.indicator = indicator
        self.start = time.time()
        self.title = title
        self.msgs = []
        self.isfinalized = False
        self.logger = print

    def __del__(self):
        if not self.isfinalized:
            self.finalize()

    def log(self, title: str) -> None:
        """计时打点

        Args:
            title (str): 本次计时的标题
        """
        self.end = time.time()
        self.msgs.append(
            "%s %s usd time: %fms"
            % (self.indicator, self.title, (self.end - self.start) * 1000)
        )
        self.title = title
        self.start = time.time()

    def finalize(self, logger=None) -> None:
        """结束本轮计时

        Args:
            logger : 传入的日志对象
        """
        if logger:
            self.logger = logger

        def show():
            if self.logger == print:
                for msg in self.msgs:
                    print(msg)
            else:
                for msg in self.msgs:
                    self.logger.debug(msg)

        self.end = time.time()
        self.msgs.append(
            "%s %s usd time: %fms"
            % (self.indicator, self.title, (self.end - self.start) * 1000)
        )
        show()
        self.isfinalized = True
