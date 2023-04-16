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
from tqdm import tqdm

try:
    from substyles.base import BaseSubstyle
except ModuleNotFoundError:
    from dynamic_subtitles.substyles.base import BaseSubstyle

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
        self.input_file  = Path(input_file)
        self.output_file = Path(output_file)
        self.subtitle_file = Path(subtitle_file)
        self.substyle_name = subtyle_name
        self.font_path = Path(font_path)
        self.frames_path = Path(frames_path)
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.end_time = float(end_time) if end_time else None
        self.project_name = self._get_project_name(self.input_file.stem)
        self.frames_dir = self.frames_path / f'{self.project_name}_frames'
        self.db_file = self.db_path / f'{self.project_name}.db'
        self.kwargs = kwargs
        
        # Parse args
        try:
            self._parse_validator()
        except Exception as e:
            raise Exception("[ERROR] Invalid args: ", e)
        
        self._confirm_ffmpeg_installation()
        

    
    def _confirm_ffmpeg_installation(self):
        """
            Confirms that ffmpeg is installed on the system
        """
        
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError:
            raise Exception("ffmpeg is not installed.")
    
    def _validate_video_file_path(self, file_path: str):
        try:
            if self.end_time:
                subprocess.check_output(["ffmpeg", "-i", file_path, "-t", f'{int(self.end_time)/1000:.3f}', "-f", "null", "-"])
            else:
                subprocess.check_output(["ffmpeg", "-i", file_path, "-f", "null", "-"])
        except subprocess.CalledProcessError:
            raise ValueError(f"[ERROR] {file_path} does not exist or is not a valid video format")
    
    def _validate_file_path(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"[ERROR] {file_path} does not exist")
    
    def _import_substyle(self):
        substyle_module = importlib.import_module(f'substyles.{self.substyle_name}')
        substyle_class = getattr(substyle_module, f'{self.substyle_name.capitalize()}Substyle')
        
        if not issubclass(substyle_class, BaseSubstyle):
            raise ValueError(f"[ERROR] {self.substyle_name} is not a valid substyle")
        
        self.substyle = substyle_class(self.font_path, **self.kwargs)
    
    def _parse_validator(self):
        """
            Args validator
        """
        self._validate_video_file_path(str(self.input_file))
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
        if self.end_time:
            subprocess.run([
                'ffmpeg',
                '-i',
                str(self.input_file),
                '-vn',
                '-acodec',
                'copy',
                '-t',
                f'{self.end_time/1000:.3f}',
                '-y',
                self.audio_output
                ], check=True)
        else:
            subprocess.run([
                'ffmpeg',
                '-i',
                str(self.input_file),
                '-vn',
                '-acodec',
                'copy',
                '-y',
                self.audio_output
                ], check=True)
        print(f"[INFO] Audio extraction completed in {time.time() - start_time:.3f} seconds")
    
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

        cap = cv2.VideoCapture(str(self.input_file))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int((self.end_time / 1000) * self.fps) if self.end_time else frame_count
        
        for i in tqdm(range(max_frames), desc="Extracting frames"):
            ret, frame = cap.read()
            if ret:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
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
        print(f"[INFO] Frame extraction completed in {time.time() - start_time:.3f} seconds")
    
    def add_text_to_video(self):
        # Read subtitles
        subs = pysrt.open(str(self.subtitle_file))

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
        for frame in frame_files:
            cmd.extend(["-i", f"{self.frames_dir}/{frame}"])
        cmd.extend(["-i", self.audio_output, "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(self.output_file)])
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
