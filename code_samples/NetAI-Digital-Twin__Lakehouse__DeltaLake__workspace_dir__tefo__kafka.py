from confluent_kafka import KafkaError, Producer, Consumer
import signal
import sys

# kafka_server = '10.80.0.13'
# port = '9000'

kafka_server = '210.125.85.62'
port = '9094'

# 종료 신호 처리 함수 정의
def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting gracefully...')
    sys.exit(0)

# 시그널 핸들러 등록 (Ctrl+C 처리)
signal.signal(signal.SIGINT, signal_handler)

# Producer 테스트
def test_producer():
    conf = {'bootstrap.servers': f'{kafka_server}:{port}'}
    producer = Producer(conf)

    try:
        producer.produce('KOSME1', key='test', value='test_value')
        producer.flush(10)
        print("Producer connected and message sent successfully.")
    except KafkaError as e:
        print(f"Failed to connect Producer: {e}")

# Consumer 테스트
def test_consumer():
    conf = {
        'bootstrap.servers': f'{kafka_server}:{port}',
        'group.id': 'test_group',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(conf)
    consumer.subscribe(['KOSME1'])

    try:
        while True:
            msg = consumer.poll(1.0)  # 대기 시간을 1초로 줄임
            if msg is None:
                print("No messages received.")
            elif msg.error():
                print(f"Consumer error: {msg.error()}")
            else:
                print(f"Consumer connected and received message: {msg.value().decode('utf-8')}")
    except KafkaError as e:
        print(f"Failed to connect Consumer: {e}")
    finally:
        consumer.close()

# 실행
test_producer()
test_consumer()
