import sys

class TeeLogger:
    """将标准输出同时写入文件和终端的日志记录器。"""

    def __init__(self, log_file, mode='a'):
        self.terminal = sys.stdout
        self.log = open(log_file, mode, encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class TeeLoggerStderr:
    """将标准错误输出同时写入文件和终端的日志记录器。"""

    def __init__(self, log_file, mode='a'):
        self.terminal = sys.stderr
        self.log = open(log_file, mode, encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
