import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, shared_memory

from soccer.utils import get_player_range, MAX_FRAMES, WIDTH, HEIGHT

import psutil
import os

class VideoLoader:
    def __init__(
        self,
        shared_memory_name,
        task_input_queue: Queue = None,
        task_output_queue: Queue = None,
    ):
        self.shared_memory_name = shared_memory_name
        self.task_input_queue = task_input_queue
        self.task_output_queue = task_output_queue

        self.shared_frames = None

    def start_processing(self):
        self.load_handler = Process(target=self.process_loader)
        self.load_handler.start()

    def load_video_frames_parallel(
        self, video_path, client_id, client_count, player_index, player_count
    ):
        # start_time = time.time()
        start_frame, end_frame = get_player_range(
            client_id, client_count, player_index, player_count
        )

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed to read first frame")

        h, w, c = frame.shape
        num_frames = end_frame - start_frame
        frame_batch = np.empty((num_frames, h, w, c), dtype=np.uint8)
        frame_batch[0] = frame

        i = 1
        while i < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_batch[i] = frame
            i += 1

        cap.release()

        # self.shared_frames[start_frame:start_frame+i].copy_(torch.from_numpy(frame_batch[:i]))
        # s1 = time.time()
        self.shared_frames[start_frame:start_frame+i, :h, :w, :] = frame_batch[:i]
        # print(f"{time.time() - s1}s to take np")

        # print(
        #     f"[🧘‍♂️Loader] loader_start_frame: {start_frame}, loader_end_frame: {start_frame + i} in {time.time() - start_time:.3f}s"
        # )

        return h, w, c

        # return frames, start_frame, end_frame

    def process_loader(self):
        shm = shared_memory.SharedMemory(name=self.shared_memory_name)
        self.shared_frames = np.ndarray((MAX_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
        # try:
        #     p = psutil.Process(os.getpid())
        #     p.nice(-15)
        # except Exception as e:
        #     print(e)
        #     print(f"[⚠️ Priority] Failed to set process priority: {e}")
        
        while True:
            if self.task_input_queue.empty():
                time.sleep(0.02)
                continue

            task = self.task_input_queue.get()
            (
                video_path,
                challenge_id,
                client_id,
                client_count,
                player_index,
                player_count,
            ) = task

            h, w, c = self.load_video_frames_parallel(
                video_path, client_id, client_count, player_index, player_count
            )

            self.task_output_queue.put((h, w, c))
            time.sleep(0.5)
