import colorsys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from typing import List

try:
    from substyles.base import BaseSubstyle
except ImportError:
    from dynamic_subtitles.substyles.base import BaseSubstyle
    

class DefaultSubstyle(BaseSubstyle):
    def __init__(self,
    font_path: str, **kwargs):
        self.font_path = font_path
        try:
            self.font_size = kwargs['font_size']
        except KeyError:
            print(f"font_size not specified, using default value of 40")
            self.font_size = 40
        
    def add_text_to_frames(self, frames_path: List[str], subtitle_text: str):
        """
        Adds subtitle text to a list of frames paths.
        """
        
        self.frames_path = frames_path
        self.subtitle_text = subtitle_text
        
        scene_color = self.get_scene_color()
        # Convert RGB to HSL
        h, l, s = colorsys.rgb_to_hls(*scene_color)
        # Increase lightness by 0.2
        l += 0.2
        # Convert back to RGB
        scene_color = tuple(round(c * 255) for c in colorsys.hls_to_rgb(h, l, s))
        text = self.subtitle_text
        for frame_path in self.frames_path:
            img_with_text = self.add_text_to_image(
                font_color=scene_color,
                image_path=frame_path,
                output_path=frame_path,
                font_path=self.font_path,
            )
    def add_text_to_image(
        self,
        font_color,
        image_path,
        text_position=None,
        output_path=None,
        font_path="config/roboto_mono.ttf",
        stroke_width_ratio=0.05,
        scene_color=None,
    ):
    
        # Create the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(font_path), self.font_size)
        text_width, text_height = draw.textsize(self.subtitle_text, font=font)
        
        # Adjust the vertical position of the text based on font size and actual text height
        if text_position is None:
            x = (image.width - text_width) // 2
            y = image.height - text_height - int(self.font_size*1.5)
        else:
            x, y = text_position
        
        # Calculate the stroke width based on the font size
        stroke_width = max(1, int(self.font_size * stroke_width_ratio))
        
        # Add stroke to the text
        stroke_fill = tuple([255-c for c in font_color])
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            draw.text((x+dx*stroke_width, y+dy*stroke_width), self.subtitle_text, font=font, fill=stroke_fill)
        
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
        
        draw.text((x, y), self.subtitle_text, text_color, font=font)
        
        if output_path:
            image.save(str(output_path))
        
        return image, (x, y)

    
    def get_scene_color(self, n_clusters=5, batch_size=10):
        palette = []
        for i in range(0, len(self.frames_path), batch_size):
            batch_frame_paths = self.frames_path[i:i+batch_size]
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

