import json
import time
import psycopg2
import websocket
from datetime import datetime
import pytz
import threading
import pdb

"""
UWB RTLS 데이터를 Sewio WebSocket을 통해 받아와서 DB에 저장하는 코드
"""

class DataManager:
    def __init__(self, configs):
        self.table_name = configs['table_name']
        self.configs = configs
        self.db_connect(configs)
        
    def db_connect(self, configs):
        try:
            self.conn = psycopg2.connect(
                dbname=configs['db_name'],
                user=configs['db_user'],
                password=configs['db_password'],
                host=configs['db_host'],
                port=configs['db_port']
            )
            self.cursor = self.conn.cursor()
            print("Database connection successfully established.")
        except Exception as e:
            print(f"Failed to connect to the database: {e}")
            
    def store_data_in_db(self, tag_id, posX, posY, timestamp, anchor_info):
        try:
            query = f"INSERT INTO {self.table_name} (tag_id, x_position, y_position, timestamp, anchor_info) VALUES (%s, %s, %s, %s, %s)"
            self.cursor.execute(query, (tag_id, posX, posY, timestamp, anchor_info))
            self.conn.commit()
        except Exception as e:
            print(f"Failed to store data in the database: {e}")
            self.conn.rollback()        
        
        
class SewioWebSocket:
    def __init__(self, manager, api_key, socket_url, resource):
        self.manager = manager
        self.api_key = api_key
        self.resource = resource
        self.socket_url = socket_url
        self.last_message_time = time.time()
        self.check_interval = 300 # 체크할 간격 (초)
        self.timeout_interval = 600 # 타임아웃 간격 (초)

    def on_message(self, ws, message):
        self.last_message_time = time.time()
        # 데이터 처리
        tag_id, posX, posY, timestamp, anchor_info = self.process_message(message)
        try:
            # DB 저장
            self.manager.store_data_in_db(tag_id, posX, posY, timestamp, anchor_info)
        except Exception as e:
            print(f"Data Transfer Error: {e}")
    
    def process_message(self, message):
        data = json.loads(message)
        tag_id = data["body"]["id"]
        posX = float(data["body"]["datastreams"][0]["current_value"].replace('%', ''))
        posY = float(data["body"]["datastreams"][1]["current_value"].replace('%', ''))
        timestamp = self.timestamp_process(data["body"]["datastreams"][0]["at"])

        # extended_tag_position 존재 여부 확인 및 처리
        if "extended_tag_position" in data["body"]:
            anchor_info = json.dumps(data["body"]["extended_tag_position"])
        else:
            anchor_info = json.dumps({})
        
        return tag_id, posX, posY, timestamp, anchor_info
    
    # UTC -> KST
    def timestamp_process(self, stamp):
        # 문자열을 datetime 객체로 변환
        timestamp_utc = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S.%f')
        
        # UTC 시간대로 설정
        utc_timezone = pytz.timezone('UTC')
        timestamp_utc = utc_timezone.localize(timestamp_utc)
        
        # 원하는 시간대(KST)로 변환
        kst_timezone = pytz.timezone('Asia/Seoul')
        timestamp_kst = timestamp_utc.astimezone(kst_timezone)
        
        # 시간대 정보 없이 문자열로 반환
        return timestamp_kst.strftime('%Y-%m-%d %H:%M:%S.%f')
        
    def on_error(self, ws, error):
        print("Error: ", error)
        print("Try to Reconnect")
        self.reconnect()

    def on_close(self, ws, close_status_code, close_msg):
        print("Closed.")
        print("Try to Reconnect")
        self.reconnect()

    def on_open(self, ws):
        print("Opened connection")
        subscribe_message = f'{{"headers":{{"X-ApiKey":"{self.api_key}"}},\
                            "method":"subscribe","resource":"{self.resource}"}}'
        print(subscribe_message)
        ws.send(subscribe_message)
        self.last_message_time = time.time()
        self.start_monitoring()

    def start_monitoring(self):
        def monitor():
            while True:
                current_time = time.time()
                if current_time - self.last_message_time > self.timeout_interval:
                    print("No messages received for a while, Try to reconnect....")
                    self.reconnect()
                    break
                time.sleep(self.check_interval)

        monitoring_thread = threading.Thread(target=monitor)
        monitoring_thread.daemon = True
        monitoring_thread.start()
                
    def reconnect(self):
        time.sleep(5)
        print("Reconnecting...")
        self.run()

    def run(self):
        # websocket.enableTrace(True)
        ws = websocket.WebSocketApp(
            self.socket_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        ws.on_open = self.on_open
        ws.run_forever()

def main():
    # DB 연결 설정 json
    config_file = "configs.json"
    with open(config_file, 'r') as file:
        configs = json.load(file)
        
    manager = DataManager(configs)
    client = SewioWebSocket(manager, configs['X-ApiKey'], configs['socket_url'], configs['resource'])
    client.run()

if __name__ == "__main__":
    main()
