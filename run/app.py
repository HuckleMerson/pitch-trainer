import asyncio
import websockets
import multiprocessing
from multiprocessing import Queue, shared_memory
import time
from tqdm import tqdm
import threading
import random
import zlib
import orjson

from concurrent.futures import ThreadPoolExecutor
import time

from soccer.utils import (
    download_video,
    is_video_valid,
    get_range,
    MAX_FRAMES,
    WIDTH,
    HEIGHT,
    TASK_TYPE_LOAD_FRAMES,
    TASK_TYPE_PROCESS,
)
from soccer.player_tracker import Player
from soccer.pitch_tracker import Pitcher
from soccer.recognizer import Recognizer
from soccer.video_loader import VideoLoader

import numpy as np

import json


def create_shared_video_frames(
    name="shared_frames", shape=(MAX_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8
):
    size = np.prod(shape) * np.dtype(dtype).itemsize
    shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, array


class VideoProcessingClientServer:
    def __init__(self, num_players=1, num_pitchers=1, num_recogers=1):
        self.num_players = num_players
        self.num_pitchers = num_pitchers
        self.num_recogers = num_recogers

        self.loader_input_queues: list[Queue] = []
        self.loader_output_queues: list[Queue] = []
        self.loaders = []

        self.player_input_queues: list[Queue] = []
        self.player_output_queues: list[Queue] = []
        self.players = []

        self.pitcher_input_queues: list[Queue] = []
        self.pitcher_output_queues: list[Queue] = []
        self.pitchers = []

        self.recog_input_queues: list[Queue] = []
        self.recog_output_queues: list[Queue] = []
        self.recogers = []

    def new_task_arrived(
        self, video_url, challenge_id, client_id, client_count, start_frame, end_frame
    ):
        while True:
            video_path = download_video(video_url, challenge_id)
            # video_path = "/tmp/video-9.mp4"
            if is_video_valid(video_path):
                break
        print(f"[Global] Downloaded video: {video_path}")
        loader_count = len(self.loader_input_queues)
        for index, queue in enumerate(self.loader_input_queues):
            queue.put(
                (
                    video_path,
                    challenge_id,
                    client_id,
                    client_count,
                    index,
                    loader_count,
                )
            )

        hwc = self.get_loader_output()

        h, w, c = hwc

        player_count = len(self.player_input_queues)
        for index, queue in enumerate(self.player_input_queues):
            queue.put(
                (
                    TASK_TYPE_LOAD_FRAMES,
                    (
                        challenge_id,
                        client_id,
                        client_count,
                        index,
                        player_count,
                        h,
                        w,
                        c,
                    ),
                )
            )
            queue.put((TASK_TYPE_PROCESS, None))

        pitcher_count = len(self.pitcher_input_queues)
        for index, queue in enumerate(self.pitcher_input_queues):
            queue.put(
                (
                    TASK_TYPE_LOAD_FRAMES,
                    (
                        challenge_id,
                        client_id,
                        client_count,
                        index,
                        pitcher_count,
                        h,
                        w,
                        c,
                    ),
                )
            )
            queue.put((TASK_TYPE_PROCESS, None))

        for index, queue in enumerate(self.recog_input_queues):
            queue.put((TASK_TYPE_LOAD_FRAMES, (start_frame, end_frame, hwc)))

    def new_recog_task_arrived(self, task_list_data):
        recog_count = len(self.recog_input_queues)

        for i in range(recog_count):
            start, end = get_range(len(task_list_data), recog_count, i)
            chunk = task_list_data[start:end]
            self.recog_input_queues[i].put((TASK_TYPE_PROCESS, chunk))

    def get_loader_output(self):
        def get_loader_output(self):
            load_status = [None] * len(self.loader_output_queues)

            while True:
                for i, queue in enumerate(self.loader_output_queues):
                    if load_status[i] is None and not queue.empty():
                        try:
                            load_status[i] = queue.get_nowait()
                        except:
                            pass

                if None not in load_status:
                    return load_status

        start_time = time.time()

        hwc_list = get_loader_output(self)

        print(f"[Client 🔃] Loading done in {time.time() - start_time:.2f}s")

        return hwc_list[0]

    def get_obj_output(self, challenge_id, start_time):
        def get_player_output():
            objects = [None] * len(self.player_output_queues)

            while True:
                threads = []

                def get_from_queue(i, queue):
                    try:
                        output = queue.get_nowait()
                        objects[i] = output["objects"]
                    except:
                        pass

                for i, queue in enumerate(self.player_output_queues):
                    t = threading.Thread(target=get_from_queue, args=(i, queue))
                    t.start()
                    threads.append(t)

                for t in threads:
                    t.join()

                if None not in objects:
                    return objects

        play_objects = get_player_output()
        flat_objs = [item for sub in play_objects for item in sub]

        print(f"[Client 🚶] Playing done in {time.time() - start_time:.2f}s")
        recog_start_time = time.time()

        self.new_recog_task_arrived(flat_objs)
        recog_outputs = self.get_recog_output(challenge_id)

        print(f"[Client 🚶‍➡️] Recognition done in {time.time() - recog_start_time:.2f}s")
        print(f"[Client 🦘] Task done in {time.time() - start_time:.2f}s")

        s1 = time.time()

        flat_objs = [item for sub in recog_outputs for item in sub]
        print(f"[Client 🦘] Flatten done in {time.time() - s1:.2f}s")

        return flat_objs

    def get_key_output(self, start_time):
        def get_pitcher_output():
            keypoints = [None] * len(self.pitcher_output_queues)

            while True:
                threads = []

                def get_from_queue(i, queue):
                    try:
                        output = queue.get_nowait()
                        keypoints[i] = output["keypoints"]
                    except:
                        pass

                for i, queue in enumerate(self.pitcher_output_queues):
                    t = threading.Thread(target=get_from_queue, args=(i, queue))
                    t.start()
                    threads.append(t)

                for t in threads:
                    t.join()

                if None not in keypoints:
                    return keypoints

        pitch_keypoints = get_pitcher_output()
        flat_keys = [item for sub in pitch_keypoints for item in sub]

        print(f"[Client 📍] Pitching done in {time.time() - start_time:.2f}s")

        return flat_keys

    def get_recog_output(self, challenge_id):
        outputs = [None] * len(self.recog_output_queues)

        while True:
            threads = []

            def get_from_queue(i, queue):
                try:
                    outputs[i] = queue.get()
                except:
                    pass

            for i, queue in enumerate(self.recog_output_queues):
                t = threading.Thread(target=get_from_queue, args=(i, queue))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            if None not in outputs:
                return outputs

    def start_processing_pools(
        self,
        num_loaders=1,
        num_players_each=1,
        num_pitchers_each=1,
        num_recogers_each=1,
    ):
        shm, array = create_shared_video_frames()
        for i in tqdm(range(num_loaders), desc="Starting loaders"):
            iq = Queue()
            oq = Queue()
            loader = VideoLoader(shm.name, task_input_queue=iq, task_output_queue=oq)
            loader.start_processing()
            self.loader_input_queues.append(iq)
            self.loader_output_queues.append(oq)
            self.loaders.append(loader)

        for i in tqdm(range(self.num_players), desc="Starting players"):
            for _ in range(num_players_each):
                iq = Queue()
                oq = Queue()
                player = Player(
                    shm.name,
                    cuda_device_id=i,
                    task_input_queue=iq,
                    task_output_queue=oq,
                )
                player.start_processing()
                self.player_input_queues.append(iq)
                self.player_output_queues.append(oq)
                self.players.append(player)
        for i in tqdm(range(self.num_pitchers), desc="Starting pitchers"):
            for _ in range(num_pitchers_each):
                iq = Queue()
                oq = Queue()
                pitcher = Pitcher(
                    shm.name,
                    cuda_device_id=i,
                    task_input_queue=iq,
                    task_output_queue=oq,
                )
                pitcher.start_processing()
                self.pitcher_input_queues.append(iq)
                self.pitcher_output_queues.append(oq)
                self.pitchers.append(pitcher)

        for i in tqdm(range(self.num_recogers), desc="Starting Recogers"):
            for _ in range(num_recogers_each):
                iq = Queue()
                oq = Queue()
                recognizer = Recognizer(
                    shm.name,
                    cuda_device_id=i,
                    task_input_queue=iq,
                    task_output_queue=oq,
                )
                recognizer.start_processing()
                self.recog_input_queues.append(iq)
                self.recog_output_queues.append(oq)
                self.recogers.append(recognizer)


processing_server = VideoProcessingClientServer()

async def client_ws_handler(websocket):
    async def get_task_res(message):
        try:
            data = orjson.loads(message)

            if data.get("type") != "challenge":
                return {"error": "Invalid message type"}

            challenge_id = data["challenge_id"]
            video_url = data["video_url"]
            client_id = data["client_id"]
            client_count = data["client_count"]

            print(f"[WS] Received task: {challenge_id} from server")
            start_time = time.time()

            start_idx, end_idx = get_range(750, client_count, client_id)

            processing_server.new_task_arrived(
                video_url, challenge_id, client_id, client_count, start_idx, end_idx
            )
            print(f"First step done in {time.time() - start_time}s")
            s2 = time.time()

            # s2 = 0
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_obj = executor.submit(
                    processing_server.get_obj_output,
                    challenge_id,
                    s2,
                )
                future_kp = executor.submit(processing_server.get_key_output, s2)

                recog_objects = future_obj.result()
                keypoints = future_kp.result()
            print(f"Second step done in {time.time() - s2}s")

            frame_objects = [[] for _ in range(750)]
            frame_keypoints = [[] for _ in range(750)]

            recog_objects.sort(key=lambda x: x["fii"])

            for obj in recog_objects:
                del obj["fii"]
                fid = obj.pop("frame_id")
                frame_objects[fid].append(obj)

            for fid, points in keypoints:
                frame_keypoints[fid] = points.tolist()

            result = [
                {
                    "frame_number": i,
                    "objects": frame_objects[i],
                    "keypoints": frame_keypoints[i],
                }
                for i in range(start_idx, end_idx)
            ]

            result.sort(key=lambda r: len(r["objects"]))
            remove_count = int(len(result) * 0.15)
            for i in range(len(result)):
                if i < remove_count or len(result[i]["objects"]) < 2:
                    result[i].pop("objects", None)

            print(f"[Global 🚩] All done in {time.time() - start_time}s")
            return {"frames": result}

        except Exception as e:
            print(e)
            return {"frames": str(e)}

    try:
        async for message in websocket:
            result = await get_task_res(message)
            payload = zlib.compress(orjson.dumps(result))

            await websocket.send(payload)

    except websockets.exceptions.ConnectionClosed:
        print("[WS] Connection closed by server")




async def main():
    multiprocessing.set_start_method("spawn")
    processing_server.start_processing_pools()

    print("[WS] Starting WebSocket server on ws://0.0.0.0:8765")
    async with websockets.serve(
        client_ws_handler,
        "0.0.0.0",
        8765,
        ping_interval=30,
        ping_timeout=10,
        max_size=10 * 1024 * 1024,
    ):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
