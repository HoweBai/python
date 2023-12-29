from abc import ABC, abstractmethod
import pymysql

class DBBase(ABC):
    """DB 基类"""
    def __init__(self, logger) -> None:
        super().__init__()
        self.dbconn = None
        self.logger = logger
        try:
            self.init_connection()
            self.logger.info("init db connection success")
        except Exception as e:
            self.logger.critical("init db connection error, [{}]".format(e))
            exit()
        
    def __del__(self) ->None:
        try:
            self.__destroy_connection()
            self.logger.info("destroy db connection success")
        except Exception as e:
            self.logger.critical("destroy db connection error, [{}]".format(e))
            exit()
        
    @abstractmethod
    def init_connection(self):
        """创建数据库连接"""
        pass


    def get_db_data(self, sql_cmd: str):
        """根据sql命令获取数据"""
        if not self.__check_conn():
            self.logger.error("invalid connection when get_db_data, sql[{}]".format(sql_cmd))
            return None
        
        cursor = None
        db_data = None
        try:
            # 创建cursor
            cursor = self.dbconn.cursor()
                
            # 执行sql
            self.logger.info("get_db_data excute sql[{}]". format(sql_cmd))
            cursor.execute(sql_cmd)
            db_data = cursor.fetchall()
        except Exception as e:
            self.logger.error("get_db_data failed, [{}]".format(e))
        else:
            self.logger.info("get_db_data success, rows[{}]".format(len(db_data)))
        finally:
            # 关闭cursor
            if cursor:
                cursor.close()
                
        return db_data
            
        
    def insert_data(self, tb_name: str, values:list, num:int) -> bool:
        """将数据批量导入mysql, num 用于校验数据的列数

        Args:
            tb_name (str): 要插入数据的表名 
            values (list): 要插入的数据
            num (int): 插入数据的列数

        Returns:
            bool : 插入成功返回True，否则False
        """
        if not self.__check_conn():
            self.logger.error("invalid connection when insert_data, table[{}]".format(tb_name))
            return False
        
        # 简单校验
        if len(values) == 0:
            self.logger.warn("insert_data values is empty, no need to insert.")
            return True
        elif len(values[0]) != num:
            self.logger.error("insert_data length of values[{}] is not equal to num[{}]".format(len(values[0]), num))
            return False
        # 如果数据不是元组就转化为元组
        values = [(v) for v in values]

        # 拼接命令
        val_str = "%s," * (num - 1) + "%s"
        sql_cmd = "replace {} values({})".format(tb_name, val_str)
        cursor = None
        try:    
            # 创建cursor
            cursor = self.dbconn.cursor()

            # 执行sql, 批量插入
            self.logger.info("insert_data excute sql[{}]".format(sql_cmd))
            cursor.executemany(sql_cmd, values)
            # 提交
            self.dbconn.commit()
        except Exception as e:
            self.logger.error("insert_data failed [{}]".format(e))
            return False
        else:
            self.logger.info("insert_data success len[{}].".format(len(values)))
            return True
        finally:
            # 关闭cursor
            if cursor:
                cursor.close()
            

    def clear_data(self, tb_name: str) -> bool:
        """清除指定表的数据

        Args:
            tb_name (str): 要清楚的表名

        Returns:
            bool: 操作是否成功
        """
        if not self.__check_conn():
            self.logger.error("invalid connection when clear_data, table[{}]".format(tb_name))
            return False
        cursor = None

        # 拼接命令
        sql_cmd = "truncate {}".format(tb_name)
        
        try:
            # 创建cursor
            cursor = self.dbconn.cursor()
            self.logger.info("clear_data excute sql[{}]".format(sql_cmd))
            # 执行sql
            cursor.execute(sql_cmd)
            # 提交修改
            self.dbconn.commit()
        except Exception as e:
            self.logger.critical("clear_data failed [{}]".format(e))
            return False
        else:
            self.logger.info("clear_data success")
            return True
        finally:
            # 关闭cursor
            if cursor:
                cursor.close()

    
    def __destroy_connection(self):
        """销毁数据库连接"""
        if self.dbconn:
            self.dbconn.close()
    
    def __check_conn(self):
        if self.dbconn is None:
            self.logger.error("db connection is deprecated.")
            return False
        else:
            return True

class MysqlHelper(DBBase):
    """mysql工具类"""

    def __init__(self, settings, logger) -> None:
        self.settings = settings
        super().__init__(logger)

    def __del__(self) -> None:
        return super().__del__()

    # 连接数据库连接
    def init_connection(self):
        """连接数据库连接"""
        conf_mysql = self.settings
        # 连接数据库
        self.dbconn = pymysql.connect(
            host=conf_mysql.host,
            user=conf_mysql.username,
            passwd=conf_mysql.password,
            port=conf_mysql.port,
        )

    # 获取所有销售数据
    def get_db_data_sell(self):
        """获取所有销售数据"""
        #                      0      1        2            3            4         5           6         7        8          9        10        11
        fetch_str = "SELECT v_type, d_date, v_customer, v_employee, v_warehouse, v_goods, dc_quantity, v_unit, dc_price, dc_sales, dc_cost, dc_profit FROM tlcw.tb_saleinfo"
        return self.get_db_data(fetch_str)

    # 获取所有进货数据
    def get_db_data_buy(self):
        """获取所有进货数据"""
        #                      0      1        2            3        4          5        6        7          8          9
        fetch_str = "SELECT v_type, d_date, v_vendor, v_goods, dc_quantity, v_unit, dc_price, dc_cost, v_employee, v_warehouse FROM tlcw.tb_buyinfo"
        return self.get_db_data(fetch_str)

    # 获取所有单位数据
    def get_db_data_dwinfo(self):
        """获取所有客户数据"""
        fetch_str = "SELECT name, phone_number FROM tlcw.tb_dwinfo"
        return self.get_db_data(fetch_str)

    # 获取商品最近进货信息
    def get_goods_latest_bought_info(self):
        """获取商品最近进货信息"""
        fetch_str = """
            select v_goods, latest_date, dc_price, v_vendor from
            (
                SELECT v_goods, v_vendor , d_date AS latest_date, dc_price, ROW_NUMBER() over(partition by v_goods order by d_date desc) as rowNum
                FROM tlcw.tb_buyinfo
            ) temp
            where temp.rowNum = 1
        """
        latest_bought_map = {}
        for one_row in self.get_db_data(fetch_str):
            if one_row[0] not in latest_bought_map:
                latest_bought_map[one_row[0]] = one_row[1:]
        return latest_bought_map


