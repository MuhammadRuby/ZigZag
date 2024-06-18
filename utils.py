from typing import List
import cv2
import numpy as np
from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point
import time
from typing import List, Tuple

class TimeEstimator:
    def __init__(self):
        self.positions = []
        self.start_bottle_position = None
        self.start_bottle_line = None
        self.n_bottle_index = None


    # def set_start_bottle(self, player_position: Point, bottle_positions: List[np.ndarray]) -> np.ndarray:
    #     distances = [np.linalg.norm(np.array(player_position) - np.array([(bottle[0] + bottle[2]) / 2, (bottle[1] + bottle[3]) / 2])) for bottle in bottle_positions]
    #     nearest_bottle_index = np.argmin(distances)
    #     self.n_bottle_index = nearest_bottle_index
    #     return bottle_positions[nearest_bottle_index]
    def set_start_bottle(self, player_position: Point, bottle_detections) -> int:
        # Calculate distances from the player to the center of each bottle
        distances = [
            np.linalg.norm(np.array(player_position) - np.array([(bottle[0][0] + bottle[0][2]) / 2, (bottle[0][1] + bottle[0][3]) / 2])) 
            for bottle in bottle_detections
        ]
        # Find the index of the nearest bottle
        nearest_bottle_index = np.argmin(distances)
        self.n_bottle_index = nearest_bottle_index
        # Return the tracker ID of the nearest bottle
        return bottle_detections.tracker_id[nearest_bottle_index]

    def update_bottle_position(self, tracker_id: int, bottle_detections: List[Tuple[List[float], float, int, int]]) -> Point:
        # Find the detection with the specified tracker ID
        for index,ID in enumerate(bottle_detections.tracker_id):
            if ID == tracker_id:
                # Calculate and return the center position of the bottle
                bottle = bottle_detections.xyxy[index]
                bottle_center_x = (bottle[0] + bottle[2]) / 2
                self.start_bottle_line= bottle_center_x
                return bottle_center_x
        # If the tracker ID is not found, return None or raise an exception
        return None

    def set_start_line_position(self,bottle_center_x ):
        self.start_bottle_line = bottle_center_x
        return self.start_bottle_line

    def update_nearest_bottle(self,bottle_positions: List[np.ndarray]):
        return bottle_positions[self.n_bottle_index]

    def crossed_start_bottle_line(self, player_position: Point) -> bool:
        if self.start_bottle_line is None:
            return False
        
        current_player_x = player_position
        return  self.start_bottle_line -5 <= current_player_x <= self.start_bottle_line+5

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
            time_text = f"Time: {(self.end_time - self.start_time)/10:.2f} seconds"
        elif self.start_time is not None:

          # time_text = f"Time: {time.time() - self.start_time:.2f} seconds"
          current_time  = (time.time() - self.start_time)/10
          time_text = f"Time: {current_time:0.2f} seconds"
        else:
          time_text = f"Time: tracking ....."


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