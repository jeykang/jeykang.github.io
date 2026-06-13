import psycopg2
import os

from dotenv import load_dotenv
load_dotenv(dotenv_path='.my_env')

class StoreDB():
    # PostgreSQL에 연결
    def __init__(self):
        self.conn = psycopg2.connect(
            host=f"{os.getenv('LOCAL_IP_ADDRESS')}",        # 호스트 주소 (로컬의 경우 localhost)
            port="5432",            # 포트
            database="uwb_storing",  # 데이터베이스 이름
            user="postgres",    # 사용자 이름
            password="postgres" # 비밀번호
        )

        # 커서 생성
        self.cur = self.conn.cursor()

    # 데이터 테이블 생성 (필요한 경우)
    def create_table(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS uwb_kkr (
                id serial4 PRIMARY KEY,
                tag_id varchar(255),
                x_position float8,
                y_position float8,
                timestamp timestamp DEFAULT CURRENT_TIMESTAMP,
                anchor_info jsonb
            );                         
        """)
        

            # CREATE TABLE IF NOT EXISTS l40_metrics (
            #     batch_id BIGINT NOT NULL,            -- 데이터를 수집한 배치 ID
            #     id SERIAL PRIMARY KEY,               -- 각 레코드에 대한 고유 ID
            #     node_id VARCHAR(255) NOT NULL,       -- 노드의 고유 ID
            #     gpu_id VARCHAR(255) NOT NULL,        -- GPU의 고유 ID
            #     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,        -- 데이터를 수집한 시각
            #     gpu_utilization FLOAT NOT NULL,      -- GPU 사용률 (0.0 ~ 100.0)
            #     gpu_memory_used BIGINT NOT NULL,     -- GPU 메모리 사용량 (단위: MB 또는 Byte)
            #     gpu_memory_free BIGINT NOT NULL,     -- GPU 메모리 남은 양 (단위: MB 또는 Byte)
            #     gpu_temperature FLOAT NOT NULL,      -- GPU 온도 (섭씨)
            #     gpu_power_draw FLOAT NOT NULL        -- GPU 전력 사용량 (와트)                
            # )
                    
    # 데이터 삽입
    def insert_table(self):
    # cur.execute("""
    #     INSERT INTO your_table (name, age, city)
    #     VALUES (%s, %s, %s)
    # """, ("John Doe", 29, "New York"))

        # 여러 데이터 삽입 (배치 삽입)
        data = [ # node, gpu_id, util, mem_used, mem_free, temp, power_draw
            (0, "0", "1", 23.4, 15, 45595, 25.2, 37.844),
            (0, "0", "2", 21.2, 19, 45591, 25.2, 40.854),
            (0, "0", "3", 20.3, 234, 45352, 27.3, 42.714),
            (0, "0", "4", 22.4, 4505, 41050, 34.4, 82.127),
        ]

        self.cur.executemany("""
            INSERT INTO l40_metrics (batch_id, node_id, gpu_id, gpu_utilization, gpu_memory_used, gpu_memory_free, gpu_temperature, gpu_power_draw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, data)

    def commit(self):
        # 트랜잭션 커밋 (변경 사항 적용)
        self.conn.commit()
    # 연결 및 커서 닫기
    
    def close(self):
        self.cur.close()
        self.conn.close()

# print("Data uploaded successfully.")

if __name__=="__main__":
    db = StoreDB()
    db.create_table()
    # db.insert_table()
    db.commit()
    db.close()
    print("DB Task Finished successfully.")