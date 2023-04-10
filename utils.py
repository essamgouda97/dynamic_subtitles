from PIL import Image, ImageDraw, ImageFont
# import extcolors
import numpy as np
import cv2
import sqlite3
import pysrt
import os, subprocess, colorsys
from natsort import natsorted

# def get_image_colors(image: Image.Image):
#     return extcolors.extract_from_image(image, tolerance = 12, limit = 12)

def merge_audio(video_path, audio_path, output_path):
    subprocess.run(['ffmpeg', '-i', video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y',output_path], check=True)

def extract_audio(input_path, output_path):
    subprocess.run(['ffmpeg', '-i', input_path, '-vn', '-acodec', 'copy', '-y', output_path], check=True)

def get_image_colors(frame_paths, n_clusters=5, batch_size=10):
    palette = []
    for i in range(0, len(frame_paths), batch_size):
        batch_frame_paths = frame_paths[i:i+batch_size]
        batch_frames = [cv2.imread(frame_path) for frame_path in batch_frame_paths]
        combined_image = np.hstack(batch_frames)
        data = cv2.resize(combined_image, (100, 100)).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)
        cluster_sizes = np.bincount(labels.flatten())
        batch_palette = []
        for cluster_idx in np.argsort(-cluster_sizes):
            batch_palette.append(np.full((combined_image.shape[0], combined_image.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
        palette.extend(batch_palette)
    return np.hstack(palette)

def get_avg_color(list_of_colors):
    return tuple(np.average(np.array(list_of_colors), axis=0).astype(int))
def get_scene_color(scene_paths, n_clusters=5, batch_size=10):
    palette = []
    for i in range(0, len(scene_paths), batch_size):
        batch_frame_paths = scene_paths[i:i+batch_size]
        batch_frames = [cv2.imread(frame_path) for frame_path in batch_frame_paths]
        combined_image = np.hstack(batch_frames)
        data = cv2.resize(combined_image, (100, 100)).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)
        cluster_sizes = np.bincount(labels.flatten())
        batch_palette = []
        for cluster_idx in np.argsort(-cluster_sizes):
            batch_palette.append(centers[cluster_idx].astype(int))
        palette.extend(batch_palette)
    palette = np.array(palette)
    dominant_color = palette[np.argmax(np.sum(palette, axis=1)), :]
    return dominant_color

def add_text_to_image(
        font_color,
        text,
        image_path,
        font_size=18,
        text_position=None,
        output_path=None,
        font_path="config/roboto_mono.ttf",
        stroke_width_ratio=0.05,
        scene_color=None,
):
    
    # Create the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)
    
    # Adjust the vertical position of the text based on font size and actual text height
    if text_position is None:
        x = (image.width - text_width) // 2
        y = image.height - text_height - int(font_size*1.5)
    else:
        x, y = text_position
    
    # Calculate the stroke width based on the font size
    stroke_width = max(1, int(font_size * stroke_width_ratio))
    
    # Add stroke to the text
    stroke_fill = tuple([255-c for c in font_color])
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        draw.text((x+dx*stroke_width, y+dy*stroke_width), text, font=font, fill=stroke_fill)
    
    # Set the text color to a mix of the font color and the scene color
    if scene_color is not None:
        alpha = 0.5
        text_color = (
            int((1 - alpha) * font_color[0] + alpha * scene_color[0]),
            int((1 - alpha) * font_color[1] + alpha * scene_color[1]),
            int((1 - alpha) * font_color[2] + alpha * scene_color[2])
        )
    else:
        text_color = tuple(font_color) if sum(font_color) > 382 else tuple([255-c for c in font_color])
    
    draw.text((x, y), text, text_color, font=font)
    
    if output_path:
        image.save(output_path)
    
    return image, (x, y)

def save_frames_to_db(video_path, frames_folder_path, end_time=None, db_path="outputs/frames.db"):
    
    # Create folder to save frames
    if not os.path.exists(frames_folder_path):
        os.makedirs(frames_folder_path)

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS frames
                 (timestamp REAL PRIMARY KEY, path TEXT)''')

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        ret, frame = cap.read()

        if ret:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Break if timestamp exceeds end_time
            if end_time is not None and timestamp > end_time:
                break

            frame_name = str(timestamp) + ".jpg"
            frame_path = os.path.join(frames_folder_path, frame_name)

            # Save frame to file
            cv2.imwrite(frame_path, frame)

            # Insert timestamp and frame path into database
            c.execute("INSERT OR REPLACE INTO frames VALUES (?, ?)", (timestamp, frame_path))
            conn.commit()

        else:
            break

    # Close database connection
    conn.close()
    
    return fps

def get_frames_between_timestamps(start_timestamp, end_timestamp, db_path="outputs/frames.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Query the database for all frames between the start and end timestamps
    c.execute("SELECT path FROM frames WHERE timestamp BETWEEN ? AND ?", (start_timestamp, end_timestamp))
    rows = c.fetchall()

    # Extract the paths from the rows and return them as a list
    paths = [row[0] for row in rows]

    conn.close()

    return paths


def add_text_to_video(subtitles_path, stroke_width=1, font_size=20, db_path="outputs/frames.db", font_path="config/roboto_mono.ttf"):
    # Read subtitles
    subs = pysrt.open(subtitles_path)

    # Loop over each subtitle and add text to the corresponding frames
    for sub_idx in range(len(subs)):
        # Compute the start and end timestamps in milliseconds
        start_time_ms = subs[sub_idx].start.to_time().hour * 3600000 \
                        + subs[sub_idx].start.to_time().minute * 60000 \
                        + subs[sub_idx].start.to_time().second * 1000 \
                        + subs[sub_idx].start.to_time().microsecond // 1000
        end_time_ms = subs[sub_idx].end.to_time().hour * 3600000 \
                        + subs[sub_idx].end.to_time().minute * 60000 \
                        + subs[sub_idx].end.to_time().second * 1000 \
                        + subs[sub_idx].end.to_time().microsecond // 1000
        
        # Get the frames between the start and end timestamps
        frames_paths = get_frames_between_timestamps(start_time_ms, end_time_ms, db_path=db_path)
        if frames_paths:
            scene_color = get_scene_color(frames_paths)
            # Convert RGB to HSL
            h, l, s = colorsys.rgb_to_hls(*scene_color)
            # Increase lightness by 0.2
            l += 0.2
            # Convert back to RGB
            scene_color = tuple(round(c * 255) for c in colorsys.hls_to_rgb(h, l, s))
            text = subs[sub_idx].text
            for frame_path in frames_paths:
                img = cv2.imread(frame_path)
                img_with_text = add_text_to_image(
                    font_color=scene_color,
                    text=text,
                    image_path=frame_path,
                    font_size=font_size,
                    output_path=frame_path,
                    font_path=font_path,
                )

def create_video_from_frames(frames_dir, output_path, fps):
    frame_files = natsorted((os.listdir(frames_dir)))
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    size = cv2.imread(frame_paths[0]).shape[:2][::-1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()
