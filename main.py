import cv2
import mediapipe as mp
import math
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(static_image_mode=self.__mode__,
                                        max_num_hands=self.__maxHands__,
                                        min_detection_confidence=self.__detectionCon__,
                                        min_tracking_confidence=self.__trackCon__)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        self.canvas = None
        self.prev_index_pos = None
        self.line_color = (255, 255, 255)  # Default: White

    def set_line_color(self, color):
        """Update the drawing color."""
        colors = {
            "White": (255, 255, 255),
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "Yellow": (0, 255, 255),
        }
        self.line_color = colors.get(color, (255, 255, 255))

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.canvas is None:
            self.canvas = np.zeros_like(frame, dtype=np.uint8)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)

            finger_positions = self.findFingerPositions(frame)
            if finger_positions:
                fingers = self.fingersUp(finger_positions)
                gesture = self.detectHandGesture(fingers)

                if gesture == "Erase" and 9 in finger_positions:
                    cx, cy = finger_positions[9]
                    cv2.circle(frame, (cx, cy), 25, (255, 0, 0), -1)

                    if self.prev_index_pos is not None:
                        cv2.line(self.canvas, self.prev_index_pos, (cx, cy), (0, 0, 0), 50)

                    self.prev_index_pos = (cx, cy)

                elif gesture == "Draw" and 8 in finger_positions:
                    cx, cy = finger_positions[8]
                    cv2.circle(frame, (cx, cy), 10, self.line_color, -1)

                    if self.prev_index_pos is not None:
                        cv2.line(self.canvas, self.prev_index_pos, (cx, cy), self.line_color, 5)

                    self.prev_index_pos = (cx, cy)

                else:
                    self.prev_index_pos = None

        return frame

    def overlayCanvas(self, frame):
        """Overlay the drawing canvas onto the current frame."""
        return cv2.addWeighted(frame, 1, self.canvas, 0.5, 0)

    def clearCanvas(self):
        """Clear the entire canvas."""
        self.canvas = None

    def findFingerPositions(self, frame):
        """Find fingertip and PIP joint positions and return their coordinates."""
        finger_positions = {}
        if self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[0]
            h, w, c = frame.shape

            for id in self.tipIds + [6, 9, 10, 14, 18]:
                if id < len(handLms.landmark):
                    lm = handLms.landmark[id]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    finger_positions[id] = (cx, cy)

        return finger_positions

    def fingersUp(self, finger_positions):
        """Determine which fingers are up."""
        fingers = [0, 0, 0, 0, 0]

        if not finger_positions:
            return fingers

        for i, pip_id in enumerate([6, 10, 14, 18], start=1):
            tip_id = self.tipIds[i]

            if tip_id in finger_positions and pip_id in finger_positions:
                if finger_positions[tip_id][1] < finger_positions[pip_id][1]:
                    fingers[i] = 1

        return fingers

    def detectHandGesture(self, fingers):
        """Detect specific gestures based on finger positions."""
        if fingers == [0, 1, 0, 0, 0]:
            return "Draw"
        elif fingers == [0, 1, 1, 1, 1]:
            return "Erase"
        elif sum(fingers) == 0:
            return "Don't Draw"
        return "None"


### 🎨 Tkinter GUI ###
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracking Drawing App")

        # Initialize OpenCV
        self.cap = cv2.VideoCapture(0)
        self.detector = HandTrackingDynamic(detectionCon=0.8, trackCon=0.8)

        # Create a frame for controls
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        # Color selection dropdown
        self.color_var = tk.StringVar(value="White")
        self.color_menu = ttk.Combobox(self.control_frame, textvariable=self.color_var, values=["White", "Red", "Green", "Blue", "Yellow"])
        self.color_menu.pack(side=tk.LEFT, padx=10, pady=10)
        self.color_menu.bind("<<ComboboxSelected>>", self.change_color)

        # Clear canvas button
        self.clear_button = tk.Button(self.control_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Video display label
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Start video loop
        self.update_video()

    def change_color(self, event=None):
        """Change the drawing color based on user selection."""
        self.detector.set_line_color(self.color_var.get())

    def clear_canvas(self):
        """Clear the drawing canvas."""
        self.detector.clearCanvas()

    def update_video(self):
        """Capture video frames and update Tkinter display."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = self.detector.findFingers(frame)
            frame = self.detector.overlayCanvas(frame)

            # Convert frame to Tkinter format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def close(self):
        """Close the application safely."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
