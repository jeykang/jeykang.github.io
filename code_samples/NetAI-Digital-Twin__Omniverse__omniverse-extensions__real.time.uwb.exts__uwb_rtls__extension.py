import omni.ext
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf
import asyncio
from aiokafka import AIOKafkaConsumer
import json
import os

# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class UwbRtlsTrackingExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, "my_config.json")        
        # config_file = "my_config.json"
        try:
            with open(config_file, 'r') as file:
                self.configs = json.load(file)
            print("file is loaded")
        except:
            print("file is not loaded")
            
        print("[my.extension] MyExtension startup")
        self._window = ui.Window("UWB Tracking", width=300, height=200)
        self._stop_event = asyncio.Event() # 스탑용
        self._consumer_task = None # 스탑용
        with self._window.frame:
            with ui.VStack():
                self._prim_path_input = ui.StringField(name="Prim", width=300, height=30, tooltip="Type Tracking Prim")
                self._prim_path_input.model.set_value("/World/paman")
                
                def uwb_on():
                    self._stop_event.clear() # 소비 중지 플래그 초기화
                    self.tracking()
                    
                def uwb_off():
                    self.stop_tracking()
                    
                ui.Button("Tracking on", clicked_fn=uwb_on)
                ui.Button("Tracking off", clicked_fn=uwb_off)
                
    def on_shutdown(self):
        print("[uwb.rtls.tracking] uwb rtls tracking shutdown")
        self._stop_event.set()  # 소비 중지 플래그 설정
            
    def tracking(self):
        def _get_prim_info():
            # Extention Window에 기입된 Prim 경로 가져오기
            selected_prim_path = self._prim_path_input.model.get_value_as_string()
            # print(selected_prim_path)
            
            # Prim 경로로부터 Usd.Prim 가져오기
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(selected_prim_path)
            if not prim.IsValid():
                self._info_label.text = "Prim not found"
                return
            return prim
                
        prim = _get_prim_info()
        if prim == None:
            print("Prim not found")
            return
        
        loop = asyncio.get_event_loop()
        self._consumer_task = loop.create_task(self.consume(prim))
        # loop.run_until_complete(self.consume(prim))
        


    def stop_tracking(self):
        self._stop_event.set()  # 소비 중지 플래그 설정
        if self._consumer_task:
            self._consumer_task.cancel()  # 소비 중지
            self._consumer_task = None
                                
    async def consume(self, prim):
        consumer = AIOKafkaConsumer(
            self.configs['kafka_topic'],
            bootstrap_servers=self.configs['kafka_server'],
            group_id="kkr_omni1")
        
        # Kafka consumer 시작
        await consumer.start()
        try:
            # 무한 루프를 통해 메시지 소비
            async for msg in consumer:
                if self._stop_event.is_set():  # 소비 중지 플래그 확인
                    break                
                print("Consumed message", msg.value)
                # 메시지를 처리하고 USD Composer에 반영
                data = msg.value.decode('utf-8')
                await self.move_prim(data, prim)
        finally:
            # Kafka consumer 종료
            await consumer.stop()
            
    async def move_prim(self, data, prim):
        coordinate_json_data = json.loads(data)
        x, z = coordinate_json_data['posX'] * 100, coordinate_json_data['posY'] * 100
        translation = [-z, 93.0, x]
        
        if prim.GetTypeName() == "Xform":
            new_pos = Gf.Vec3d(translation[0], translation[1], translation[2])  # 예시 위치
            xform_api = UsdGeom.XformCommonAPI(prim)
            xform_api.SetTranslate(new_pos)