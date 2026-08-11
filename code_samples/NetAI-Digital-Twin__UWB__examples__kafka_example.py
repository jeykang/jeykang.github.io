from confluent_kafka import KafkaError, Producer, Consumer

# Producer 테스트
def test_producer():
    conf = {'bootstrap.servers': 'kafka_server:port'}
    producer = Producer(conf)

    try:
        producer.produce('k_test', key='test', value='test_value')
        producer.flush()
        print("Producer connected and message sent successfully.")
    except KafkaError as e:
        print(f"Failed to connect Producer: {e}")

# Consumer 테스트
def test_consumer():
    conf = {
        'bootstrap.servers': 'kafka_server:port',
        'group.id': 'test_group',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(conf)
    consumer.subscribe(['k_test'])

    try:
        msg = consumer.poll(10.0)
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
