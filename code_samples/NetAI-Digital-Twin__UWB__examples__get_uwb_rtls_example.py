""" 
Fleet/16 Tag(Master Anchor=13)의 UWB RTLS Data를 받아오는 코드
"""
import websocket

def on_message(ws, message):
    print("Received:", message)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")
    subscribe_message = '{"headers":{"X-ApiKey":"api_key"}, \
                        "method":"subscribe","resource":"/feeds/16"}'
    print(subscribe_message)
    ws.send(subscribe_message)
    
if __name__ == "__main__":
    # websocket.enableTrace(True)
    ws = websocket.WebSocketApp("web_socke_url",
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()

