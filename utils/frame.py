from PIL import Image, ImageDraw, ImageFont
import extcolors
import numpy as np
import cv2
import datetime
import sqlite3
import pysrt
import os

def get_image_colors(image: Image.Image):
    return extcolors.extract_from_image(image, tolerance = 12, limit = 12)

def get_avg_color(list_of_colors):
    return tuple(np.average(np.array(list_of_colors), axis=0).astype(int))

def add_text_to_image(
        text,
        text_color,
        stroke_width,
        stroke_fill,
        image,
        font_size=18,
        text_position=None,
        output_path=None,
        font_path="config/font.ttf"
    ):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    text_position = text_position if text_position else ((image.width - draw.textsize(text, font=font)[0]) // 2, image.height - 60)
    
    # Add stroke to the text
    if stroke_fill is not None:
        draw.text((text_position[0]-stroke_width, text_position[1]), text, font=font, fill=stroke_fill)
        draw.text((text_position[0]+stroke_width, text_position[1]), text, font=font, fill=stroke_fill)
        draw.text((text_position[0], text_position[1]-stroke_width), text, font=font, fill=stroke_fill)
        draw.text((text_position[0], text_position[1]+stroke_width), text, font=font, fill=stroke_fill)
    
    draw.text(text_position, text, text_color, font=font)
    if output_path:
        image.save(output_path)
    
    return image


def save_frames_to_db(video_path, frames_folder_path, db_path="outputs/frames.db"):
    
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
def add_text_to_video(subtitles_path):
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
        frames_paths = get_frames_between_timestamps(start_time_ms, end_time_ms)
        
        # Add text to the frames and save them with the same file name
        mid_frame = frames_paths[len(frames_paths) // 2]
        colors, pixel_count = get_image_colors(Image.open(mid_frame))
        for frame_path in frames_paths:
            img = Image.open(frame_path)
            draw = ImageDraw.Draw(img)
            text = subs[sub_idx].text
            colors, pixel_count = get_image_colors(img)
            text_color = colors[3][0] # 4th most frequent color, magic number
            stroke_fill = colors[-1][0] # last color, magic number
            img_with_text = add_text_to_image(
                text=text,
                text_color=text_color,
                stroke_width=1,
                stroke_fill=stroke_fill, 
                image=img,
                font_size=20,
            )
            img_with_text.save(frame_path)

def create_video_from_frames(frames_dir, output_path, fps):
    frame_files = sorted(os.listdir(frames_dir))
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    size = cv2.imread(frame_paths[0]).shape[:2][::-1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()
