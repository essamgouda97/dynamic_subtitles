import os
import argparse
import ast
import sys

# For files paths
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

try:
    from utils import MainRunner
except ModuleNotFoundError:
    from dynamic_subtitles.utils import MainRunner


def main():
    # Pass args
    parser = argparse.ArgumentParser()
    input_path = parser.add_argument(
        "--input", 
        "-i", 
        type=str,
        help="Path to input video file", 
        default="inputs/burning.mp4"
    )
    output_path = parser.add_argument(
        "--output",
        "-o", 
        type=str,
        help="Path to output video file",
        default="outputs/burning.mp4"
    )
    subtitle_path = parser.add_argument(
        "--subtitle",
        type=str,
        help="Path to subtitle file",
        default="inputs/burning.srt"
    )
    font_path = parser.add_argument(
        "--font",
        type=str,
        help="Path to font file",
        default="config/futura.ttf"
    )
    substyle_name = parser.add_argument(
        "--substyle",
        type=str,
        help="Name of substyle class to use in substyles/ folder",
        default="default"
    )
    db_path = parser.add_argument(
        "--db-path",
        type=str,
        help="Path to database file",
        default="outputs/db"
    )
    end_time = parser.add_argument(
        "--end-time",
        type=str,
        help="End time of clip in ms. Default is None",
        default=None,
    )
    frames_path = parser.add_argument(
        "--frames-path",
        type=str,
        help="Path to frames folder",
        default="outputs/frames"
    )
    extras = ast.literal_eval(parser.add_argument(
        '--extras',
        '-e',
        type=str, 
        help='a dictionary of extra options',
        default='{}'
    ))

    runner = MainRunner(
        input_file=input_path,
        output_file=output_path,
        subtitle_file=subtitle_path,
        subtyle_name=substyle_name,
        font_path=font_path,
        db_path=db_path,
        end_time=end_time,
        frames_path=frames_path,
        **extras,
    )
    
    runner.extract_audio()
    runner.extract_frames()
    runner.add_text_to_video()
    runner.create_video_from_frames()
            

if __name__ == "__main__":
    main()
    sys.exit(0)