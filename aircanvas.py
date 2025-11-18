"""

AirCanvas: Virtual Air Drawing Application using Hand Gestures

Gestures & Controls:
  • Draw  .................... Index finger up (only)
  • Select / Actions ......... Index + Middle fingers up (hover tip over buttons)
  • Eraser Toggle ............ Use Select mode over [ERASER]
  • Brush Size ............... Use Select mode over [–] or [+]
  • Color Change ............. Use Select mode over a color swatch
  • Clear Canvas ............. Use Select mode over [CLEAR]
  • Save PNG ................. Use Select mode over [SAVE]
  • Quit ..................... Press 'q'
AirCanvas: Virtual Air Drawing Application using Hand Gestures
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

SMOOTH_ALPHA = 0.25      # lower = smoother cursor (try 0.2–0.35)
GRACE_FRAMES = 10        # tolerate brief losses (try 8–14)
MODE_STICKY_FRAMES = 10  # keep gesture state a bit longer (try 8–12)
UI_H = 90           # Height of the top UI bar
PADDING = 12        # Padding around UI items
SWATCH_W = 60
SWATCH_H = 60
GAP = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX

COLORS = [
    (0, 0, 0),        # black
    (255, 255, 255),  # white
    (255, 0, 0),      # blue (BGR)
    (0, 255, 0),      # green
    (0, 0, 255),      # red
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
]

ACTIONS = [
    ("ERASER", (180, 180, 180)),
    ("-", (200, 200, 200)),
    ("+", (200, 200, 200)),
    ("CLEAR", (230, 230, 230)),
    ("SAVE", (230, 230, 230)),
]

DEBOUNCE_MS = 600

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def count_fingers_up(hand_landmarks, image_width, image_height, handedness_label):
    """Return list of booleans [thumb, index, middle, ring, pinky] whether each is up.
    Up is defined by tip.y < pip.y for all except thumb.
    For thumb, use x-direction depending on handedness because of mirroring.
    """
    lm = hand_landmarks.landmark

    def to_px(idx):
        return int(lm[idx].x * image_width), int(lm[idx].y * image_height)

    tips_up = [False] * 5

    # Index, middle, ring, pinky by y
    for i, (tip, pip) in enumerate(zip(FINGER_TIPS[1:], FINGER_PIPS[1:]), start=1):
        tips_up[i] = lm[tip].y < lm[pip].y

    # If handedness_label == 'Right', in a mirrored image, thumb is on viewer's left.

    if handedness_label == 'Right':
        tips_up[0] = lm[4].x < lm[3].x
    else:
        tips_up[0] = lm[4].x > lm[3].x

    return tips_up

def layout_ui(w):
    """Compute rectangles for all UI elements given frame width.
    Returns: swatch_rects, action_rects where each rect=(x1,y1,x2,y2), and labels
    """
    x = PADDING
    y = PADDING
    swatches = []
    for _ in COLORS:
        swatches.append((x, y, x + SWATCH_W, y + SWATCH_H))
        x += SWATCH_W + GAP

    action_rects = []
    for _ in ACTIONS:
        action_rects.append((x, y, x + SWATCH_W, y + SWATCH_H))
        x += SWATCH_W + GAP

    return swatches, action_rects


def draw_ui(frame, swatches, action_rects, current_color, eraser, brush_size):
    # UI background bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], UI_H + PADDING), (245, 245, 245), -1)

    for rect, color in zip(swatches, COLORS):
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
    
        if tuple(color) == tuple(current_color) and not eraser:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    for rect, (label, fill) in zip(action_rects, ACTIONS):
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.8, 2)
        tx = x1 + (SWATCH_W - tw) // 2
        ty = y1 + (SWATCH_H + th) // 2 - 6
        cv2.putText(frame, label, (tx, ty), FONT, 0.8, (40, 40, 40), 2, cv2.LINE_AA)

    bx = frame.shape[1] - 90
    by = UI_H // 2 + PADDING
    preview_color = (220, 220, 220) if eraser else tuple(int(c) for c in current_color)
    cv2.circle(frame, (bx, by), max(6, brush_size // 2), preview_color, -1)
    cv2.putText(frame, f"{brush_size}px", (bx - 30, by + 35), FONT, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(frame, "Eraser" if eraser else "Brush", (bx - 36, by - 28), FONT, 0.6, (60, 60, 60), 1, cv2.LINE_AA)




def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, frame = cap.read()
    if not ok:
        print("Could not access webcam.")
        return

    h, w = frame.shape[:2]
    swatches, action_rects = layout_ui(w)

    canvas = np.zeros((h, w, 3), dtype=np.uint8) + 255  # white canvas
    current_color = (0, 0, 0)
    brush_size = 12
    eraser = False

    last_point = None
    last_action_time = 0

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror for natural control
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            draw_ui(frame, swatches, action_rects, current_color, eraser, brush_size)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            index_tip_px = None
            fingers_up = [False] * 5
            handed_label = 'Right'

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                if results.multi_handedness:
                    handed_label = results.multi_handedness[0].classification[0].label

                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Count raised fingers
                fingers_up = count_fingers_up(hand_landmarks, w, h, handed_label)

                # Index tip location
                lm = hand_landmarks.landmark
                ix = int(lm[8].x * w)
                iy = int(lm[8].y * h)
                index_tip_px = (ix, iy)

                # Visual cursor
                if index_tip_px is not None:
                    cv2.circle(frame, index_tip_px, 8, (255, 120, 0), -1)

            now = time.time() * 1000
            in_select_mode = fingers_up[1] and fingers_up[2] and not (fingers_up[3] or fingers_up[4])  # index+middle only
            in_draw_mode = fingers_up[1] and not any(fingers_up[i] for i in [0,2,3,4])  # only index

            if in_select_mode and index_tip_px is not None:
                x, y = index_tip_px
                if y <= UI_H + PADDING:  
                    if now - last_action_time > DEBOUNCE_MS:
                        for rect, color in zip(swatches, COLORS):
                            x1, y1, x2, y2 = rect
                            if x1 <= x <= x2 and y1 <= y <= y2:
                                current_color = color
                                eraser = False
                                last_action_time = now
                                break
                        for rect, (label, _) in zip(action_rects, ACTIONS):
                            x1, y1, x2, y2 = rect
                            if x1 <= x <= x2 and y1 <= y <= y2:
                                if label == "ERASER":
                                    eraser = not eraser
                                elif label == "+":
                                    brush_size = min(80, brush_size + 2)
                                elif label == "-":
                                    brush_size = max(2, brush_size - 2)
                                elif label == "CLEAR":
                                    canvas[:] = 255
                                elif label == "SAVE":
                                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    fname = f"aircanvas_{ts}.png"
                                    out = canvas.copy()
                                    # Draw the UI strip on top of saved image as well? We'll save only artwork
                                    cv2.imwrite(fname, out)
                                    print(f"Saved {fname} in {os.getcwd()}")
                                    # flash indicator
                                    
                                    cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
                                last_action_time = now
                                break

            # Drawing mode
            if in_draw_mode and index_tip_px is not None:
                x, y = index_tip_px
                if y > UI_H + PADDING:
                    if last_point is None:
                        last_point = (x, y)
                    cv2.line(
                        canvas,
                        last_point,
                        (x, y),
                        (255, 255, 255) if eraser else current_color,
                        brush_size,
                        lineType=cv2.LINE_AA,
                    )
                    last_point = (x, y)
                else:
                    last_point = None
            else:
                
                last_point = None

            mask_area = canvas.copy()
            frame_roi = frame[UI_H + PADDING : , :]
            canvas_roi = canvas[UI_H + PADDING : , :]
            blended = cv2.addWeighted(frame_roi, 0.4, canvas_roi, 0.6, 0)
            frame[UI_H + PADDING : , :] = blended

            cv2.putText(frame, "Draw: Index up | Select: Index+Middle up | q: quit", (16, h - 16), FONT, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

            cv2.imshow("AirCanvas – Virtual Painter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break



    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
