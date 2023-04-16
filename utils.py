# import extcolors
import cv2
import sqlite3
import pysrt
from natsort import natsorted

from pathlib import Path
import importlib
import subprocess
import time
import os
import re
import tqdm

try:
    from dynamic_subtitles.substyles.base import BaseSubstyle
except ModuleNotFoundError:
    from substyles.base import BaseSubstyle

class MainRunner():
    def __init__(self, 
                input_file: str,
                output_file: str,
                subtitle_file: str,
                subtyle_name: str,
                font_path: str,
                db_path: str = "outputs/db",
                end_time = None,
                frames_path: str = "outputs/frames",
                **kwargs
            ):
        self.input_file = input_file
        self.output_file = output_file
        self.subtitle_file = subtitle_file
        self.subtyle_name = subtyle_name
        self.font_path = font_path
        self.db_path = db_path
        self.end_time = end_time
        self.frames_path = frames_path
        self.project_name = self._get_project_name(self.input_file.stem)
        self.frames_dir = self.frames_path / f'{self.project_name}_frames'
        self.db_file = self.db_path / f'{self.project_name}.db'
        self.kwargs = kwargs
        
        
        self._confirm_ffmpeg_installation()
        
        # Parse args
        try:
            self._parse_validator()
        except Exception as e:
            raise Exception("[ERROR] Invalid args: ", e)
    
    def _confirm_ffmpeg_installation(self):
        """
            Confirms that ffmpeg is installed on the system
        """
        
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError:
            raise Exception("ffmpeg is not installed.")
    
    def _validate_file_path(self, file_path: str):
        try:
            subprocess.check_output(["ffmpeg", "-i", file_path])
        except subprocess.CalledProcessError:
            raise ValueError(f"[ERROR] {file_path} does not exist or is not a valid video format")
    
    def _import_substyle(self):
        substyle_module = importlib.import_module(f'substyles.{self.subtyle_name}')
        substyle_class = getattr(substyle_module, f'{self.subtyle_name.capitalize()}Substyle')
        
        if not isinstance(substyle_class, BaseSubstyle):
            raise ValueError(f"[ERROR] {self.subtyle_name} is not a valid substyle")
        
        self.substyle = substyle_class(self.font_path, **self.kwargs)
    
    def _parse_validator(self):
        """
            Args validator
        """
        self.input_file  = Path(self.input_file)
        self.output_file = Path(self.output_file)
        self.subtitle_file = Path(self.subtitle_file)
        self.font_path = Path(self.font_path)
        self.db_path = Path(self.db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self._validate_file_path(self.input_file)
        self._validate_file_path(self.output_file)
        self._validate_file_path(self.subtitle_file)
        self._validate_file_path(self.font_path)
        self._import_substyle()
    
    def _get_project_name(self, input_str: str):
        """
            Returns the project name from the input file name
        """
        return re.sub(r"\s+", "", input_str)
        
    
    def extract_audio(self):
        """
            Extracts audio from the video
        """
        print(f"[INFO] Extracting audio from {self.input_file}")
        start_time = time.time()
        self.audio_output = str(self.input_file.parent / f'{self.input_file.stem}_audio.aac')
        subprocess.run(['ffmpeg', '-i', self.input_path, '-vn', '-acodec', 'copy', '-y', self.audio_output], check=True)
        print(f"[INFO] Audio extraction completed in {time.time() - start_time} seconds")
    
    def extract_frames(self):
        """
            Extracts frames from the video
        """
        print(f"[INFO] Extracting frames from {self.input_file}")
        start_time = time.time()
    
        # Create folder to save frames
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Connect to SQLite database
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        # Create table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS frames
                    (timestamp REAL PRIMARY KEY, path TEXT)''')

        cap = cv2.VideoCapture(self.input_file)
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(frame_count), desc="Extracting frames"):
            ret, frame = cap.read()
            if ret:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Break if timestamp exceeds end_time
                if self.end_time is not None and timestamp > self.end_time:
                    break

                frame_name = str(timestamp) + ".jpg"
                frame_path = os.path.join(self.frames_dir, frame_name)

                # Save frame to file
                cv2.imwrite(frame_path, frame)

                # Insert timestamp and frame path into database
                c.execute("INSERT OR REPLACE INTO frames VALUES (?, ?)", (timestamp, frame_path))
                conn.commit()

            else:
                break

        # Close database connection
        conn.close()
        print(f"[INFO] Frame extraction completed in {time.time() - start_time} seconds")
    
    def add_text_to_video(self):
        # Read subtitles
        subs = pysrt.open(self.subtitle_file)

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
            frames_paths = self._get_frames_between_timestamps(start_time_ms, end_time_ms)
            if frames_paths:
                # Add text to the frames
                self.substyle.add_text_to_frames(frames_paths, subs[sub_idx].text)
            else:
                print(f"[WARNING] No frames found between {start_time_ms} and {end_time_ms}")
    
    def create_video_from_frames(self):
        frame_files = natsorted((os.listdir(self.frames_dir)))
        cmd = ["ffmpeg", "-r", str(self.fps)]

        # Add frames to the video clip
        for frame in tqdm(frame_files, desc="Creating video"):
            cmd.extend(["-i", f"{self.frames_dir}/{frame}"])
        cmd.extend(["-i", self.audio_output, "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", self.output_file])
        subprocess.run(cmd, check=True)

    def _get_frames_between_timestamps(self, start_time, end_time):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        # Query the database for all frames between the start and end timestamps
        c.execute("SELECT path FROM frames WHERE timestamp BETWEEN ? AND ?", (start_time, end_time))
        rows = c.fetchall()

        # Extract the paths from the rows and return them as a list
        paths = [row[0] for row in rows]

        conn.close()

        return paths
