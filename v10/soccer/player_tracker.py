import cv2
import time
import json
import numpy as np
from multiprocessing import Process, Queue, shared_memory
from ultralytics import YOLO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random


from soccer.utils import (
    get_player_range,
    MAX_FRAMES,
    WIDTH,
    HEIGHT,
    MIN_PLAYER_WIDTH,
    MIN_PLAYER_HEIGHT,
    TASK_TYPE_LOAD_FRAMES,
    TASK_TYPE_PROCESS,
)


def is_bbox_large_enough(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return w >= MIN_PLAYER_WIDTH and h >= MIN_PLAYER_HEIGHT


def is_touching_scoreboard_zone(x1, y1, x2, y2, frame_width=1280, frame_height=720):
    return not (x2 < 0 or x1 > frame_width or y2 < 0 or y1 > 150)


def rects_intersect(b1, b2):
    x1a, y1a, x2a, y2a = b1
    x1b, y1b, x2b, y2b = b2
    return not (
        x2a <= x1b  # b1 is left of b2
        or x1a >= x2b  # b1 is right of b2
        or y2a <= y1b  # b1 is above b2
        or y1a >= y2b  # b1 is below b2
    )


class Player:
    def __init__(
        self,
        shared_memory_name,
        player_model_path="models/player-model.pt",
        cuda_device_id=0,
        task_input_queue: Queue = None,
        task_output_queue: Queue = None,
    ):
        self.shared_memory_name = shared_memory_name
        self.shared_frames = None
        self.player_model_path = player_model_path
        self.cuda_device_id = cuda_device_id
        self.cuda_device = f"cuda:{cuda_device_id}"

        self.player_model = YOLO(self.player_model_path).to(self.cuda_device)
        self.task_input_queue = task_input_queue
        self.task_output_queue = task_output_queue

        self.frames, self.client_id, self.start_frame = None, None, None

    def start_processing(self):
        self.play_handler = Process(target=self.process_player)
        self.play_handler.start()

    def load_video_frames_parallel(
        self, client_id, client_count, player_index, player_count, h, w, c
    ):
        start_frame, end_frame = get_player_range(
            client_id, client_count, player_index, player_count
        )
        frames_np = self.shared_frames[start_frame:end_frame, :h, :w, :c]
        frames_list = list(frames_np)
        return frames_list, start_frame, end_frame

    def process_player(self):
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
                    player_index,
                    player_count,
                    h,
                    w,
                    c,
                ) = task_data
                frames, start_frame, _ = self.load_video_frames_parallel(
                    client_id, client_count, player_index, player_count, h, w, c
                )
                self.frames, self.client_id, self.start_frame = (
                    frames,
                    client_id,
                    start_frame,
                )
            elif task_type == TASK_TYPE_PROCESS:
                objects = self.process_frames(
                    self.frames, self.client_id, self.start_frame
                )
                result = {
                    "challenge_id": 0,
                    "objects": objects,
                }
                self.task_output_queue.put(result)
                time.sleep(0.4)

    def process_single_frame(self, frame, frame_id, obj, start_frame):
        objects = []

        obj_boxes = obj.boxes
        confs = obj_boxes.conf.cpu().numpy()

        height, width, _ = frame.shape
        threshold = height * width * 0.05

        areas, boxes = [], []
        for i in range(len(obj_boxes)):
            x1, y1, x2, y2 = map(int, obj_boxes.xyxy[i].cpu().tolist())
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
            boxes.append((x1, y1, x2, y2))

        sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i])
        total_area = 0
        selected_indices = []
        selected_indices_length = 0

        cnt = random.randint(5, 9)

        for i in sorted_indices:
            x1, y1, x2, y2 = boxes[i]
            box_area = areas[i]
            total_area += box_area

            is_intersect = False
            for j in range(i):
                if rects_intersect(boxes[i], boxes[j]):
                    is_intersect = True
                    break
            if is_intersect:
                continue

            if (
                total_area < threshold
                and box_area < threshold
                and selected_indices_length < cnt
            ):
                selected_indices.append(i)
                selected_indices_length += 1

        if len(selected_indices) > 1:

            for idx, i in enumerate(selected_indices):
                x1, y1, x2, y2 = boxes[i]
                if not is_bbox_large_enough(
                    x1, y1, x2, y2
                ) or is_touching_scoreboard_zone(x1, y1, x2, y2, width, height):
                    continue

                objects.append(
                    {
                        "id": 0,
                        "fii": idx,
                        "bbox": (x1, y1, x2, y2),
                        "class_id": 1,
                        "frame_id": start_frame + frame_id,
                        "img_width": width,
                        "img_height": height,
                    }
                )

        return objects

    def process_frames(self, frames, client_id, start_frame):
        start_time = time.time()
        _detections = self.player_model.predict(
            frames, stream=True, verbose=False, batch=256, imgsz=640
        )
        detections = []
        for detection in _detections:
            detections.append(detection)
        print(
            f"[🧘‍♂️Player {self.cuda_device_id} {client_id}] Processed {len(frames)} frames in {time.time() - start_time:.2f}s"
        )

        all_objects = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    self.process_single_frame, frames[i], i, detections[i], start_frame
                )
                for i in range(len(frames))
            ]
            for future in as_completed(futures):
                objs = future.result()
                all_objects.extend(objs)

        return all_objects
