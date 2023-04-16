from abc import ABC, abstractmethod
from typing import List

class BaseSubstyle(ABC):
    
    @abstractmethod
    def __init__(*args, **kwargs):
        """
        Substyle constructor.
        """
        pass

    @abstractmethod
    def add_text_to_frames(self, frames_path: List[str], subtitle_text: str):
        """
        Adds subtitle text to a list of frames paths.
        """
        raise NotImplementedError("Subclasses must implement this method.")

