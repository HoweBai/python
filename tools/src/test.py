import sys
sys.path.append("../config")

from config import settings
import os
from logger import Logger
from timer import Timer 
from dbhelper import MysqlHelper

def test():
    print(settings.mysql)
    logger = Logger("TLCW").init(settings.log)
    
    tm = Timer("build mysql connection", "MYSQL")
    mysql_helper = MysqlHelper(settings.mysql, logger)
    
    tm.log("get all sell data")
    mysql_helper.get_db_data_sell()
    
    tm.finalize(logger)

if __name__ == "__main__":
    old_wd = os.getcwd()
    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    os.chdir(cur_dir)
    test()
    os.chdir(old_wd)
