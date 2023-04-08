from pathlib import Path

from PIL import Image

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

try:
    from dynamic_subtitles.utils.frame import (
        save_frames_to_db,
        add_text_to_video,
        create_video_from_frames,
    )
except ImportError:
    from utils.frame import (
        save_frames_to_db,
        add_text_to_video,
        create_video_from_frames,
    )

FRAMES_DIR = "outputs/frames"

if __name__ == "__main__":
    fps = save_frames_to_db("inputs/test.mp4", FRAMES_DIR)
    add_text_to_video("inputs/test.srt")
    create_video_from_frames(FRAMES_DIR, "outputs/test.mp4", fps)