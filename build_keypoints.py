import cv2
import numpy as np
import requests
import os

def download_video(video_url):
    """Download video from URL"""
    response = requests.get(video_url, stream=True)
    temp_file = 'temp_video.mp4'
    with open(temp_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return temp_file

def merge_lines(lines, angle_threshold=2, distance_threshold=1):
    """Merge lines that are on the same line (collinear) using vector math"""
    if not lines:
        return []
    
    # Sort lines by length (longest first)
    lines = sorted(lines, key=lambda l: np.linalg.norm(
        np.array([l[2], l[3]]) - np.array([l[0], l[1]])
    ), reverse=True)
    
    merged_lines = []
    used = set()
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
        
        x1, y1, x2, y2 = line1
        
        # Calculate direction vector for line1
        v1 = np.array([x2 - x1, y2 - y1])
        v1 = v1 / np.linalg.norm(v1)  # Normalize
        
        # Find all collinear lines
        collinear_lines = [line1]
        
        for j, line2 in enumerate(lines):
            if j == i or j in used:
                continue
            
            x3, y3, x4, y4 = line2
            
            # Calculate direction vector for line2
            v2 = np.array([x4 - x3, y4 - y3])
            v2 = v2 / np.linalg.norm(v2)  # Normalize
            
            # Calculate angle between lines (using dot product)
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            angle_deg = np.degrees(angle)
            
            # Check if angles are close enough (within 2 degrees)
            if angle_deg < angle_threshold:
                # Calculate perpendicular distance from line1 to line2's endpoints
                # Line equation: Ax + By + C = 0
                A = y2 - y1
                B = x1 - x2
                C = x2*y1 - x1*y2
                
                # Distance from line to point formula
                dist1 = abs(A*x3 + B*y3 + C) / np.sqrt(A*A + B*B)
                dist2 = abs(A*x4 + B*y4 + C) / np.sqrt(A*A + B*B)
                
                # If either endpoint is close enough
                if min(dist1, dist2) < distance_threshold:
                    collinear_lines.append(line2)
                    used.add(j)
        
        # print(len(collinear_lines))
        if collinear_lines:
            # Merge all collinear lines
            all_x = [x for line in collinear_lines for x in line[::2]]
            all_y = [y for line in collinear_lines for y in line[1::2]]
            merged_line = (min(all_x), min(all_y), max(all_x), max(all_y))
            merged_lines.append(merged_line)
    
    return merged_lines

def detect_pitch_lines(frame):
    """Detect pitch lines using tophat filter"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Basic grass mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35,40,40), (85,255,255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    # Top hat filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    
    # Restrict to grass areas
    masked = cv2.bitwise_and(white_lines, grass)
    
    # Canny + HoughP
    blur = cv2.GaussianBlur(masked, (5,5), 0)
    edges = cv2.Canny(blur, 30, 100)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi/360,
                            threshold=30,
                            minLineLength=10,
                            maxLineGap=15)
    
    # Convert lines to list of tuples
    if lines is not None:
        lines = [line[0] for line in lines]
        # Merge nearby lines
        lines = merge_lines(lines)
    
    return lines

from soccer import SoccerPitchConfiguration

def create_high_score_keypoints(lines, frame_width, frame_height):
    """Create keypoints based on soccer vertices ratios"""
    if lines is None or len(lines) == 0:
        return None
    
    # Get pitch configuration
    pitch_config = SoccerPitchConfiguration()
    soccer_vertices = pitch_config.vertices
    """Create keypoints based on the longest line and soccer vertices ratios"""
    if lines is None or len(lines) == 0:
        return None
    
    # Find the longest line
    longest_line = max(lines, key=lambda line: np.linalg.norm(
        np.array([line[2], line[3]]) - np.array([line[0], line[1]])
    ))
    
    # Unpack line coordinates
    x1, y1, x2, y2 = longest_line
    
    # Get the goal box length vertex as reference (vertex 7)
    ref_vertex = soccer_vertices[5]  # First vertex with non-zero x-coordinate
    
    # Create keypoints based on soccer vertices ratios
    keypoints = []
    
    # Get the first 6 vertices from soccer configuration
    for i in range(6):
        # Get the vertex position
        vertex = soccer_vertices[i]
        # Calculate the ratio based on y-coordinate only
        ratio = vertex[1] / ref_vertex[1]  # Using y-coordinate ratio
        # Apply this ratio to our longest line
        x = int(x1 + (x2 - x1) * ratio)
        y = int(y1 + (y2 - y1) * ratio)
        keypoints.append([x, y])
    
    return keypoints

def get_keypoints_from_video(video_url):
    """Get keypoints for all frames in the video"""
    # Download video
    video_path = download_video(video_url)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process all frames
    for frame_idx in range(total_frames):
        print(f"Processing frame {frame_idx + 1}/{total_frames}")
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_idx}")
            continue
        
        # Convert frame to RGB if needed (some videos might be in BGR)
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pitch lines
        lines = detect_pitch_lines(frame)
        
        # Create keypoints
        keypoints = create_high_score_keypoints(lines, frame_width, frame_height)
        
        # Create visualization if keypoints were found
        if keypoints is not None:
            # Create visualization frame
            vis_frame = frame.copy()
            
            # # Draw lines with unique colors
            if lines is not None:
                # Generate unique colors for each line
                num_lines = len(lines)
                colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) 
                         for _ in range(num_lines)]
                
                for idx, line in enumerate(lines):
                    x1, y1, x2, y2 = line
                    color = colors[idx]
                    cv2.line(vis_frame, (x1, y1), (x2, y2), color, 2)
                    # Add line number near the line
                    cv2.putText(vis_frame, str(idx+1), 
                               ((x1+x2)//2, (y1+y2)//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw keypoints
            for i, (x, y) in enumerate(keypoints):
                # Ensure coordinates are within frame bounds
                x = max(0, min(frame_width-1, x))
                y = max(0, min(frame_height-1, y))
                color = (0, 0, 255) if i < 6 else (255, 0, 0)
                cv2.circle(vis_frame, (x, y), 5, color, -1)
                cv2.putText(vis_frame, str(i+1), (x+5, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save visualization
            os.makedirs('for_verify', exist_ok=True)
            filename = f'for_verify/frame{frame_idx}.jpg'
            cv2.imwrite(filename, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            print(f"Saved visualization to {filename}")
    
    # Clean up
    cap.release()
    os.remove(video_path)
    
    print("Processing complete. All frames have been saved in 'for_verify' folder.")

# Example usage:
video_url = "https://scoredata.me/chunks/769cdb6fa2074b219600dbbf4ccc72.mp4"
keypoints = get_keypoints_from_video(video_url)
print(keypoints)