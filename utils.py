from typing import List
import cv2
import numpy as np
from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point
import time

class TimeEstimator:
    def __init__(self):
        self.positions = []
        self.start_bottle_position = None
        self.start_bottle_line = None

    def set_start_bottle(self, player_position: Point, bottle_positions: List[np.ndarray]) -> np.ndarray:
        distances = [np.linalg.norm(np.array(player_position) - np.array([(bottle[0] + bottle[2]) / 2, (bottle[1] + bottle[3]) / 2])) for bottle in bottle_positions]
        nearest_bottle_index = np.argmin(distances)
        return bottle_positions[nearest_bottle_index]

    def set_start_bottle_position(self, bottle: np.ndarray):
        self.start_bottle_position = bottle
        bottle_center_x = (bottle[0] + bottle[2]) / 2
        self.start_bottle_line = bottle_center_x

    def update(self, player_position: Point, bottle_positions: List[np.ndarray]) -> bool:
        self.positions.append(player_position)
        return True

    def crossed_start_bottle_line(self, player_position: Point) -> bool:
        if self.start_bottle_line is None:
            return False
        
        current_player_x = player_position[0]
        return  self.start_bottle_line -3 <= current_player_x <= self.start_bottle_line+3

class TimeAnnotator:
    def __init__(
        self,
        start_time: float,
        end_time: float,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_padding: int = 10,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_padding: int = text_padding

    def annotate(self, frame: np.ndarray, time_estimator: TimeEstimator) -> np.ndarray:
        if self.start_time is not None and self.end_time is not None:
            time_text = f"Time: {self.end_time - self.start_time:.2f} seconds"
        else:
            time_text = f"Time: {time.time() - self.start_time:.2f} seconds"


        (text_width, text_height), _ = cv2.getTextSize(
            time_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        text_x = int(frame.shape[1] - text_width - 10)
        text_y = int(text_height + 10)

        cv2.rectangle(
            frame,
            (text_x - self.text_padding, text_y - text_height - self.text_padding),
            (text_x + text_width + self.text_padding, text_y + self.text_padding),
            self.color.as_bgr(),
            -1,
        )

        cv2.putText(
            frame,
            time_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
        return frame