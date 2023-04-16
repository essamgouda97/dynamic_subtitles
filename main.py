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
    parser.add_argument(
        "--input", 
        "-i", 
        type=str,
        help="Path to input video file", 
        default="inputs/burning.mp4"
    )
    parser.add_argument(
        "--output",
        "-o", 
        type=str,
        help="Path to output video file",
        default="outputs/burning.mp4"
    )
    parser.add_argument(
        "--subtitle",
        type=str,
        help="Path to subtitle file",
        default="inputs/burning.srt"
    )
    parser.add_argument(
        "--font",
        type=str,
        help="Path to font file",
        default="config/futura.ttf"
    )
    parser.add_argument(
        "--substyle",
        type=str,
        help="Name of substyle class to use in substyles/ folder",
        default="default"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to database file",
        default="outputs/db"
    )
    parser.add_argument(
        "--end-time",
        type=str,
        help="End time of clip in ms. Default is None",
        default=None,
    )
    parser.add_argument(
        "--frames-path",
        type=str,
        help="Path to frames folder",
        default="outputs/frames"
    )
    parser.add_argument(
        '--extras',
        '-e',
        type=str, 
        help='a dictionary of extra options',
        default='{}'
    )
    args = parser.parse_args()
    extras = ast.literal_eval(args.extras)

    runner = MainRunner(
        input_file=args.input,
        output_file=args.output,
        subtitle_file=args.subtitle,
        subtyle_name=args.substyle,
        font_path=args.font,
        db_path=args.db_path,
        end_time=args.end_time,
        frames_path=args.frames_path,
        **extras,
    )
    
    runner.extract_audio()
    runner.extract_frames()
    runner.add_text_to_video()
    runner.create_video_from_frames()
            

if __name__ == "__main__":
    main()
    sys.exit(0)