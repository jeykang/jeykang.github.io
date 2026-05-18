import json
import time
from db_websocket_builder import DataManager, SewioWebSocket
from confluent_kafka import KafkaError, Producer

"""
UWB RTLS 데이터를 Sewio WebSocket을 통해 받아와서
1. 전체 데이터는 DB에 저장
2. 필요한 데이터만 Kafka에 전송
"""

class WebSocket2Kafka(SewioWebSocket):
    def __init__(self, manager: DataManager, api_key, socket_url, resource, kafka_server, kafka_port, kafka_topic):
        super().__init__(manager, api_key, socket_url, resource)
        self.kafka_server = kafka_server
        self.kafka_port = kafka_port
        self.kafka_topic = kafka_topic
        
    def on_message(self, ws, message):
        print("메시지"+message)
        return
        self.last_message_time = time.time()
        # 데이터 처리
        tag_id, posX, posY, timestamp, anchor_info = self.process_message(message)
        try:
            # Kafka 전송
            self.kafka_producer(tag_id, posX, posY)
            # DB 저장
            self.manager.store_data_in_db(tag_id, posX, posY, timestamp, anchor_info)
        except Exception as e:
            print(f"Data Transfer Error: {e}")
            
    def kafka_producer(self, tag_id, posX, posY):
        conf = {'bootstrap.servers': f'{self.kafka_server}:{self.kafka_port}'}
        producer = Producer(conf)
        try:
            messages = {
                'tag_id':tag_id,
                'posX': posX,
                'posY': posY
            }
            message_str = json.dumps(messages)
            producer.produce(self.kafka_topic, value=message_str.encode('utf-8'))
            # 전송 완료
            producer.flush()
            # print("Producer connected and message sent successfully.")
        except KafkaError as e:
            print(f"Failed to connect Producer: {e}")
        
        self.producer = producer

def main():
    # DB 연결 설정 json
    config_file = "my_configs_tracking.json"
    with open(config_file, 'r') as file:
        configs = json.load(file)
        
    manager = DataManager(configs)
    client = WebSocket2Kafka(manager, configs['X-ApiKey'], configs['socket_url'], configs['resource'],
                             configs['kafka_server'], configs['kafka_port'], configs['kafka_topic'])
    client.run()

if __name__ == "__main__":
    main()