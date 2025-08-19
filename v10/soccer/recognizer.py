import cv2
import time
import json
import torch

import numpy as np
from multiprocessing import Process, Queue, shared_memory
from transformers import CLIPTokenizerFast, CLIPImageProcessor, CLIPModel

from soccer.utils import (
    MAX_FRAMES,
    WIDTH,
    HEIGHT,
    TASK_TYPE_LOAD_FRAMES,
    TASK_TYPE_PROCESS,
    BoundingBoxObject,
)


class Recognizer:
    def __init__(
        self,
        shared_memory_name,
        model_name="openai/clip-vit-base-patch32",
        cuda_device_id=0,
        task_input_queue: Queue = None,
        task_output_queue: Queue = None,
    ):
        self.shared_memory_name = shared_memory_name
        self.shared_frames = None

        self.model_name = model_name
        self.cuda_device_id = cuda_device_id
        self.cuda_device = f"cuda:{cuda_device_id}"

        # Perf knobs that don't change logic
        torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(False)

        self.recog_model = CLIPModel.from_pretrained(self.model_name).to(
            self.cuda_device
        )
        self.recog_model.eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.model_name, use_fast=True
        )

        # Cache constant labels & derived indices once
        self.labels = [
            BoundingBoxObject.PLAYER.value,
            BoundingBoxObject.GOALKEEPER.value,
            BoundingBoxObject.REFEREE.value,
            BoundingBoxObject.CROWD.value,
            BoundingBoxObject.BLACK.value,
            "person",
            "grass",
        ]
        self.PERSON_INDEX = 5
        self.class_ids = [2, 1, 3]

        # Pre-tokenize constant text once (moved to device lazily in _infer_clip_batch)
        self._cached_text_inputs = self.tokenizer(
            self.labels, return_tensors="pt", padding=True, truncation=True
        )

        self.task_input_queue = task_input_queue
        self.task_output_queue = task_output_queue

        (
            self.task_list_data,
            self.client_id,
            self.start_frame,
            self.end_frame,
            self.frames,
        ) = (None, None, None, None, None)

    def start_processing(self):
        self.recog_handler = Process(target=self.process_recog)
        self.recog_handler.start()

    def _infer_clip_batch(self, texts, images, model, device):
        if len(images) == 0:
            return torch.empty((0, len(texts)), dtype=torch.float32)
        
        with torch.no_grad():
            # Tokenize text
            text_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            # Process images
            image_inputs = self.image_processor(images=images, return_tensors="pt").to(
                device
            )

            # Merge inputs
            inputs = {
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
                "pixel_values": image_inputs["pixel_values"],
            }

            outputs = model(**inputs)
            return outputs.logits_per_image.cpu()

    def batched_clip_inference(self, texts, images, batch_size=64):
        all_logits = []

        for i in range(0, len(images), batch_size):
            image_batch = images[i : i + batch_size]

            logits = self._infer_clip_batch(
                texts, image_batch, self.recog_model, self.cuda_device
            )

            all_logits.append(logits)

        # Concatenate all batch outputs into one tensor
        return torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, len(texts)))

    def load_video_frames_parallel(self, start_frame, end_frame, h, w, c):
        # Return a NumPy VIEW (no list conversion/copies)
        return self.shared_frames[start_frame:end_frame, :h, :w, :c][:, :, :, ::-1]

    def process_recog(self):
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
                start_frame, end_frame, hwc = task_data
                h, w, c = hwc

                frames = self.load_video_frames_parallel(
                    start_frame, end_frame, h, w, c
                )

                (
                    self.start_frame,
                    self.end_frame,
                    self.frames,
                ) = (start_frame, end_frame, frames)

            elif task_type == TASK_TYPE_PROCESS:
                task_list_data = task_data

                outputs = self.process_recognition(
                    task_list_data, start_frame, end_frame, frames
                )

                final_output = outputs
                self.task_output_queue.put(final_output)
                time.sleep(0.2)

    def process_recognition(self, task_list_data, start_frame, end_frame, frames):
        # Extract ROIs using NumPy views (no extra copies until needed)
        frame_cache = {}  # fi -> frames[fi] view
        rois = [
            (frame_cache.setdefault(fi := (obj["frame_id"] - start_frame), frames[fi]))[
                obj["bbox"][1] : obj["bbox"][3], obj["bbox"][0] : obj["bbox"][2]
            ]
            for obj in task_list_data
        ]

        overall_logits = self.batched_clip_inference(self.labels, rois)

        refine_logits = overall_logits[:, : self.PERSON_INDEX]
        clear_logits = overall_logits[:, self.PERSON_INDEX :]

        refine_probs = refine_logits.softmax(dim=-1)
        clear_probs = clear_logits.softmax(dim=-1)

        preds = torch.argmax(refine_probs, dim=1)

        # Vectorized mask (logic unchanged)
        # keep where predicted class is one of first three AND "person" prob > 0.08
        mask = (preds < 3) & (clear_probs[:, 0] > 0.08)

        # Build outputs for masked indices only
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
        outputs = []
        for idx in indices:
            obj = task_list_data[idx]
            pred = int(preds[idx].item())
            obj["class_id"] = self.class_ids[pred]
            outputs.append(
                {
                    "id": obj["id"],
                    "fii": obj["fii"],
                    "bbox": obj["bbox"],
                    "class_id": obj["class_id"],
                    "frame_id": obj["frame_id"],
                }
            )

        return outputs
