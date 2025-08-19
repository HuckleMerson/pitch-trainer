import os
import time
import re
import tempfile
import logging
import xmlrpc.client
import numpy as np
from soccer.pitch import SoccerPitchConfiguration
from enum import Enum
import cv2

MAX_FRAMES = 750
WIDTH = 2560
HEIGHT = 1440
MIN_PLAYER_WIDTH = 15
MIN_PLAYER_HEIGHT = 40

TASK_TYPE_LOAD_FRAMES = 0
TASK_TYPE_PROCESS = 1

logger = logging.getLogger(__name__)

import httpx
shared_client = httpx.Client(timeout=30.0, follow_redirects=True, headers={
    "Connection": "keep-alive",
})
def download_video(url:str, challenge_id:int) -> str:
    start_time = time.time()
    url = url.strip()
    if not re.match(r'^https?://', url):
        raise ValueError(f'Invalid URL: {url}')
    # url = url.replace('https://scoredata.me/chunks', 'http://103.6.170.41:9710/proxy')
    
    output_dir = tempfile.gettempdir()
    file_name = f"video-{challenge_id}.mp4"
    file_path = os.path.join(output_dir, file_name)
    
    try:
        response = shared_client.get(url)
        response.raise_for_status()
            
        if "drive.google.com" in url:
            # For Google Drive, handle the download URL specially
            if "drive.usercontent.google.com" in response.url.path:
                download_url = str(response.url)
            else:
                # If redirected to Google Drive UI, extract file ID
                file_id = url.split("id=")[1].split("&")[0]
                download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
            
            # Make the actual download request
            response = shared_client.get(download_url)
            response.raise_for_status()
            
            # Create temp file with .mp4 extension
        with open(file_path, 'wb') as f:
            f.write(response.content)
            print(url)
            print(f"Video downloaded successfully to {file_path}")
            print(f"✅  Downloaded video size: {len(response.content) / 1024 / 1024:.2f} MB in {time.time() - start_time:.2f}s")
            f.close()
        return file_path
    except Exception as e:
        print(e)
        

def get_range(total_frames, client_count, client_id):
    base = total_frames // client_count
    remainder = total_frames % client_count
    if client_id < remainder:
        start = client_id * (base + 1)
        end = start + base + 1
    else:
        start = client_id * base + remainder
        end = start + base
    return start, end

def get_player_range(client_id, client_count, player_index, player_count, total_frames = 750):
    base = total_frames // client_count
    remainder = total_frames % client_count

    if client_id < remainder:
        start = client_id * (base + 1)
        end = start + base + 1
    else:
        start = client_id * base + remainder
        end = start + base

    client_frame_count = end - start
    t_base = client_frame_count // player_count
    t_remainder = client_frame_count % player_count

    if player_index < t_remainder:
        t_start = start + player_index * (t_base + 1)
        t_end = t_start + t_base + 1
    else:
        t_start = start + player_index * t_base + t_remainder
        t_end = t_start + t_base

    return t_start, t_end

class BoundingBoxObject(Enum):
    FOOTBALL = "football"
    GOALKEEPER = "goalkeeper"
    PLAYER = "football player"
    REFEREE = "referee"
    CROWD = "crowd"
    GRASS = "grass"
    GOAL = "goal"
    BACKGROUND = "background"
    BLANK = "blank"
    OTHER = "other"
    NOTFOOT = "not a football"
    BLACK = "black shape"

def get_sample_keypoints(video_width, video_height):
    pitch_config = SoccerPitchConfiguration()
    pitch_vertices = np.array(pitch_config.vertices, dtype=np.float32)

    pitch_width = pitch_config.width
    pitch_height = pitch_config.length
    scale_x = video_width / pitch_width
    scale_y = video_height / pitch_height

    scaled_pitch_vertices = pitch_vertices * np.array([scale_y, scale_x])

    keypoints = scaled_pitch_vertices[:, [1, 0]].copy()
    return keypoints.tolist()


def is_video_valid(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False
    # Try reading the first frame to confirm validity
    ret, frame = cap.read()
    cap.release()
    return ret

