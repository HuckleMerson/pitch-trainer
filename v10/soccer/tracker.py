import cv2
import time
import json
import numpy as np
from multiprocessing import Process, Queue
from ultralytics import YOLO
from tqdm import tqdm

from soccer.keypointer import recover_detected_keypoints_only

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_range(total_frames, client_count, client_id):
    base_client_frames = total_frames // client_count
    client_remainder = total_frames % client_count

    if client_id < client_remainder:
        client_start = client_id * (base_client_frames + 1)
        client_end = client_start + (base_client_frames + 1)
    else:
        client_start = client_id * base_client_frames + client_remainder
        client_end = client_start + base_client_frames
    return client_start, client_end

class Tracker:
    def __init__(
        self,
        player_model_path="models/player-model.pt",
        pitch_model_path="models/pitch-model-11n.pt",
        cuda_device_id=0,
        task_input_queue: Queue = None,
        task_output_queue: Queue = None,
    ):
        self.player_model_path = player_model_path
        self.pitch_model_path = pitch_model_path
        self.cuda_device_id = cuda_device_id
        self.cuda_device = f"cuda:{cuda_device_id}"

        self.player_model = YOLO(self.player_model_path)
        self.player_model.to(self.cuda_device)

        self.pitch_model = YOLO(self.pitch_model_path)
        self.pitch_model.to(self.cuda_device)

        self.task_input_queue = task_input_queue
        self.task_output_queue = task_output_queue

    def start_processing(self):
        self.track_handler = Process(target=self.process_tracker)
        self.track_handler.start()

    def is_touching_scoreboard_zone(self, x1, y1, x2, y2, frame_width=1280, frame_height=720):
        return not (x2 < 0 or x1 > frame_width or y2 < 0 or y1 > 150)

    def load_video_frames_parallel(self, video_path, client_id, client_count, tracker_index, tracker_count):
        # cap = cv2.VideoCapture(video_path)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # cap.release()
        total_frames = 750

        def get_tracker_range(total_frames, client_count, client_id, tracker_count, tracker_index):
            base_client_frames = total_frames // client_count
            client_remainder = total_frames % client_count

            if client_id < client_remainder:
                client_start = client_id * (base_client_frames + 1)
                client_end = client_start + (base_client_frames + 1)
            else:
                client_start = client_id * base_client_frames + client_remainder
                client_end = client_start + base_client_frames

            client_frame_count = client_end - client_start

            # --- Distribute client frames among trackers ---
            base_tracker_frames = client_frame_count // tracker_count
            tracker_remainder = client_frame_count % tracker_count

            if tracker_index < tracker_remainder:
                tracker_start = client_start + tracker_index * (base_tracker_frames + 1)
                tracker_end = tracker_start + (base_tracker_frames + 1)
            else:
                tracker_start = client_start + tracker_index * base_tracker_frames + tracker_remainder
                tracker_end = tracker_start + base_tracker_frames
            return tracker_start, tracker_end

        tracker_start_frame, tracker_end_frame = get_tracker_range(total_frames, client_count, client_id, tracker_count, tracker_index)

        print(f"[Tracker {self.cuda_device_id}] tracker_start_frame: {tracker_start_frame}, tracker_end_frame: {tracker_end_frame}")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, tracker_start_frame)
        frames = [None] * (tracker_end_frame - tracker_start_frame)
        for i in range(tracker_start_frame, tracker_end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames[i - tracker_start_frame] = frame
        cap.release()
        return frames, tracker_start_frame, tracker_end_frame

    def process_tracker(self):
        while True:
            if self.task_input_queue.empty():
                # time.sleep(0.05)
                continue
            task = self.task_input_queue.get()

            video_path, challenge_id, client_id, client_count, tracker_index, tracker_count = task
            start_time = time.time()

            frames, start_frame, _ = self.load_video_frames_parallel(video_path, client_id, client_count, tracker_index, tracker_count)
            print(f"[Tracker {self.cuda_device_id}] Loaded {len(frames)} frames in {time.time() - start_time:.2f}s")
            

            objects, keypoints = self.process_frames(frames, client_id, start_frame)

            # final_output = {
            #     "video_path": video_path,
            #     "challenge_id": challenge_id,
            #     "frames": outputs,
            #     "start_frame": start_frame,
            #     "end_frame": end_frame,
            #     "processing_time": round(time.time() - start_time, 2)
            # }
            final_output = {
                "challenge_id" : challenge_id,
                "objects" : objects,
                "keypoints" : keypoints
            }

            self.task_output_queue.put(final_output)
            # print(f"[Tracker {self.cuda_device_id} {client_id}] Done in {time.time() - start_time:.2f}s")


    def process_single_frame(self, frame, frame_id, obj, pitch, start_frame):
        objects = []
        keypoint = None

        obj_boxes = obj.boxes
        confs = obj_boxes.conf.cpu().numpy()
        original_frame = obj.orig_img[:, :, ::-1].copy()
        img_height, img_width = original_frame.shape[:2]

        pitch_keypoints_xy = pitch.keypoints.xy.cpu().numpy()
        pitch_keypoints_conf = pitch.keypoints.conf.cpu().numpy()

        if pitch_keypoints_xy.shape[0] == 0:
            pitch_xyv = None
        else:
            pitch_xys = pitch_keypoints_xy[0]        # (32, 2)
            pitch_conf = pitch_keypoints_conf[0]     # (32,)
            pitch_xyv = np.hstack([pitch_xys, pitch_conf[:, None]])  # (32, 3)

        
        detect_keypoints, detected = recover_detected_keypoints_only(frame_id, frame.copy(), pitch_xyv, (img_height, img_width))
        # detect_keypoints, detected = recover_centered_pitch_points(original_frame)

        def draw():

            frame = obj.orig_img[:, :, ::-1].copy()
            frame = cv2.copyMakeBorder(frame, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            margin_x, margin_y = 100, 100
                # for idx, (px, py) in enumerate(projected):
            for idx, (px, py) in enumerate(detect_keypoints):
                px_shifted = int(px) + margin_x
                py_shifted = int(py) + margin_y

                # Draw the green circle
                cv2.circle(frame, (px_shifted, py_shifted), radius=2, color=(0, 255, 0), thickness=-1)

                # Draw the index number just above the point
                cv2.putText(
                    frame,
                    str(idx),
                    (px_shifted, py_shifted - 6),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
            cv2.imwrite(f"pitch_result/pitch{start_frame + frame_id}.png", frame[:, :, ::-1].copy())
        # if detected:
        #     print(start_frame + frame_id)
        # draw()
        height,width,_ = frame.shape
        small_box_indexes, small_box_count = [], 0
        threshold = height * width * 0.05

        areas = []
        posis = []

        for i in range(len(obj_boxes)):
            x1, y1, x2, y2 = map(int, obj_boxes.xyxy[i].cpu().tolist())
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
            posis.append((x1, y1, x2, y2))

        # Sort indices by area (smallest to largest)
        sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i])

        total_area = 0

        for i in sorted_indices:
            x1, y1, x2, y2 = posis[i]
            box_area = areas[i]
            total_area += box_area
            if i < 4 or (total_area < threshold and box_area < threshold and small_box_count < 10):
            # if True:
            # if small_box_count < 6:
                small_box_indexes.append(i)
                small_box_count += 1
            

        for idx, i in enumerate(small_box_indexes):
        # for i in range(min(6, len(obj_boxes))):
            # bbox = obj_boxes.xyxy[i].cpu().tolist()
            # x1, y1, x2, y2 = map(int, bbox)
            x1, y1, x2, y2 = posis[i]
            if self.is_touching_scoreboard_zone(x1, y1, x2, y2, img_width, img_height):
                continue
            cropped = original_frame[y1:y2, x1:x2].copy()
            obj_content = {
                "id": 0,
                "fii" : idx,
                "bbox": (x1, y1, x2, y2),
                "class_id": 1,
                "frame_id": start_frame + frame_id,
                "rois": cropped,
                "img_width": img_width,
                "img_height": img_height
            }
            original_frame[y1:y2, x1:x2] = 0
            objects.append(obj_content)

        # keypoint = (start_frame + frame_id, detect_pitch_lines_tophat(original_frame))
        keypoint = (start_frame + frame_id, detect_keypoints)
        return objects, keypoint
    
    def get_obj_pitch(self, frames):

        # Shared result holders
        player_result = [None]
        pitch_result = [None]

        def run_player_model():
            player_result[0] = self.player_model.predict(
                frames, stream=False, verbose=False, batch=256, imgsz=640
            )

        def run_pitch_model():
            pitch_result[0] = self.pitch_model.predict(
                frames, stream=False, verbose=False
            )

        # Create threads
        t1 = threading.Thread(target=run_player_model)
        t2 = threading.Thread(target=run_pitch_model)

        # Start threads
        t1.start()
        t2.start()

        # Wait for both to finish
        t1.join()
        t2.join()

        # Access results
        objects = player_result[0]
        pitchs = pitch_result[0]
        return objects, pitchs


    def process_frames(self, frames, client_id, start_frame):
        start_time = time.time()
        # objects, pitchs = self.get_obj_pitch(frames)
        objects = self.player_model.predict(
            frames, stream=False, verbose=False, batch=256, imgsz=640
        )
        pitchs = self.pitch_model.predict(
            frames, stream=False, verbose=False
        )
        #write the code to draw the pitch on the frame and save it
        print(f"[Tracker {self.cuda_device_id} {client_id}] Processed {len(frames)} frames in {time.time() - start_time:.2f}s")
        # objects = objects[:1]

        all_objects = []
        all_keypoints = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self.process_single_frame, frames[frame_id], frame_id, obj, pitchs[frame_id], start_frame)
                for frame_id, obj in enumerate(objects)
            ]
            for future in as_completed(futures):
                objects, keypoint = future.result()
                all_objects.extend(objects)
                all_keypoints.append(keypoint)

        return all_objects, all_keypoints


    # def process_frames(self, frames, client_id, start_frame):
    #     start_time = time.time()
    #     results = self.player_model.predict(
    #         frames, stream=False, verbose=False, batch=256, imgsz=160
    #     )
    #     print(f"[Tracker {self.cuda_device_id} {client_id}] Processed {len(frames)} frames in {time.time() - start_time:.2f}s")
    #     objects = []
    #     keypoints = []
    #     for frame_id, result in tqdm(enumerate(results)):
    #         boxes = result.boxes
    #         confs = boxes.conf.cpu().numpy()
    #         # ids = (
    #         #     boxes.id.cpu().numpy().astype(int).tolist()
    #         #     if boxes.id is not None
    #         #     else [None] * len(boxes)
    #         # )
    #         original_frame = result.orig_img[:, :, ::-1]
    #         img_height, img_width = original_frame.shape[:2]

            
    #         for i in range(min(6, len(boxes))):
    #             # if confs[i] < 0.5:
    #             #     continue
    #             bbox = boxes.xyxy[i].cpu().tolist()
    #             x1, y1, x2, y2 = map(int, bbox)
    #             if self.is_touching_scoreboard_zone(x1, y1, x2, y2, img_width, img_height):
    #                 continue
    #             cropped = original_frame[y1:y2, x1:x2]
    #             obj = {
    #                 # "id": ids[i],
    #                 "id" : 0,
    #                 "bbox": bbox,
    #                 "class_id": 1,
    #                 "frame_id": start_frame + frame_id,
    #                 "rois": cropped,
    #                 "img_width" : img_width,
    #                 "img_height" : img_height
    #             }
    #             objects.append(obj)
        
    #         keypoints.append(
    #             (start_frame + frame_id, detect_pitch_lines_tophat(original_frame))
    #         )
    #     return objects, keypoints
