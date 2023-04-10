import os

# For files paths
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

try:
    from dynamic_subtitles.utils import (
        save_frames_to_db,
        add_text_to_video,
        create_video_from_frames,
        extract_audio,
        merge_audio,
    )
except ImportError:
    from utils import (
        save_frames_to_db,
        add_text_to_video,
        create_video_from_frames,
        extract_audio,
        merge_audio,
    )
NAME = "burning"
FRAMES_DIR = f"outputs/frames/{NAME}"

if __name__ == "__main__":
    print("Extracting audio from video...")
    extract_audio(f'inputs/{NAME}.mp4', f'inputs/{NAME}_audio.aac')
    print("Saving frames to database...")
    fps = save_frames_to_db(f"inputs/{NAME}.mp4", FRAMES_DIR, end_time=300000, db_path=f"outputs/frames_{NAME}.db")
    print(f"Adding text to frames (fps={fps})...")
    add_text_to_video(f"inputs/{NAME}.srt", font_size=40, db_path=f"outputs/frames_{NAME}.db", font_path="config/futura.ttf")
    print("Creating video from frames...")
    create_video_from_frames(FRAMES_DIR, f"outputs/{NAME}.mp4", fps)
    
    print("Adding audio to video...")
    merge_audio(f"outputs/{NAME}.mp4", f"inputs/{NAME}_audio.aac", f"outputs/{NAME}_audio.mp4")