import os
import sys

sys.path.append("..\\..\\..")
import tools

settings = tools.settings
settings.projdir = os.path.join(os.path.dirname(__file__), "..")
logger = tools.Logger("ML2022Spring").init(settings.log)
MysqlHelper = tools.MysqlHelper
Timer = tools.Timer
