import cv2
import time
import json
import numpy as np
from multiprocessing import Process, Queue, shared_memory
from ultralytics import YOLO
from tqdm import tqdm

# from soccer.keypointer import recover_detected_keypoints_only
from soccer_pitch import recover_detected_keypoints_only

from concurrent.futures import ThreadPoolExecutor, as_completed

from soccer.utils import (
    get_player_range,
    MAX_FRAMES,
    WIDTH,
    HEIGHT,
    TASK_TYPE_LOAD_FRAMES,
    TASK_TYPE_PROCESS,
)


class Pitcher:
    def __init__(
        self,
        shared_memory_name,
        pitch_model_path="models/pitch-model-11n-nw2.pt",
        cuda_device_id=0,
        task_input_queue: Queue = None,
        task_output_queue: Queue = None,
    ):
        self.shared_memory_name = shared_memory_name
        self.shared_frames = None

        self.pitch_model_path = pitch_model_path
        self.cuda_device_id = cuda_device_id
        self.cuda_device = f"cuda:{cuda_device_id}"

        self.pitch_model = YOLO(self.pitch_model_path)
        self.pitch_model.to(self.cuda_device)

        self.task_input_queue = task_input_queue
        self.task_output_queue = task_output_queue

        self.frames, self.client_id, self.start_frame = None, None, None

    def start_processing(self):
        self.pitch_handler = Process(target=self.process_pitcher)
        self.pitch_handler.start()

    def is_touching_scoreboard_zone(
        self, x1, y1, x2, y2, frame_width=1280, frame_height=720
    ):
        return not (x2 < 0 or x1 > frame_width or y2 < 0 or y1 > 150)

    def load_video_frames_parallel(
        self, client_id, client_count, player_index, player_count, h, w, c
    ):
        time.sleep(0.05)
        start_frame, end_frame = get_player_range(
            client_id, client_count, player_index, player_count
        )
        frames_np = self.shared_frames[start_frame:end_frame, :h, :w, :c]
        frames_list = list(frames_np)
        return frames_list, start_frame, end_frame

    def process_pitcher(self):
        shm = shared_memory.SharedMemory(name=self.shared_memory_name)
        self.shared_frames = np.ndarray(
            (MAX_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8, buffer=shm.buf
        )

        while True:
            if self.task_input_queue.empty():
                time.sleep(0.03)
                continue
            task = self.task_input_queue.get()
            task_type, task_data = task
            if task_type == TASK_TYPE_LOAD_FRAMES:

                (
                    challenge_id,
                    client_id,
                    client_count,
                    pitcher_index,
                    pitcher_count,
                    h,
                    w,
                    c,
                ) = task_data

                frames, start_frame, _ = self.load_video_frames_parallel(
                    client_id,
                    client_count,
                    pitcher_index,
                    pitcher_count,
                    h,
                    w,
                    c,
                )

                self.frames, self.client_id, self.start_frame = (
                    frames,
                    client_id,
                    start_frame,
                )
            elif task_type == TASK_TYPE_PROCESS:

                keypoints = self.process_frames(self.frames, self.client_id, self.start_frame)

                final_output = {"challenge_id": challenge_id, "keypoints": keypoints}

                self.task_output_queue.put(final_output)
                time.sleep(0.3)

    def process_single_frame(self, frame_id, frame, pitch, start_frame):
        keypoint = None

        img_height, img_width = frame.shape[:2]

        pitch_keypoints_xy = pitch.keypoints.xy.cpu().numpy()
        pitch_keypoints_conf = pitch.keypoints.conf.cpu().numpy()

        if pitch_keypoints_xy.shape[0] == 0:
            pitch_xyv = None
        else:
            pitch_xys = pitch_keypoints_xy[0]  # (32, 2)
            pitch_conf = pitch_keypoints_conf[0]  # (32,)
            pitch_xyv = np.hstack([pitch_xys, pitch_conf[:, None]])  # (32, 3)

        detect_keypoints, _ = recover_detected_keypoints_only(
            frame_id, frame, pitch_xyv, (img_height, img_width)
        )

        keypoint = (start_frame + frame_id, detect_keypoints)
        return keypoint

    def process_frames(self, frames, client_id, start_frame):
        start_time = time.time()
        _pitchs = self.pitch_model.predict(frames, stream=True, verbose=False)
        pitchs = []
        for pitch in _pitchs:
            pitchs.append(pitch)
        
        # write the code to draw the pitch on the frame and save it
        print(
            f"[🟩 Pitcher {self.cuda_device_id} {client_id}] Processed {len(frames)} frames in {time.time() - start_time:.2f}s"
        )

        all_keypoints = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    self.process_single_frame,
                    frame_id,
                    frames[frame_id],
                    pitch,
                    start_frame,
                )
                for frame_id, pitch in enumerate(pitchs)
            ]
            for future in as_completed(futures):
                keypoint = future.result()
                all_keypoints.append(keypoint)

        return all_keypoints
