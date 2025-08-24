# python roi_select.py rectified_id.jpg ROIs.json


import cv2
import json
import sys
import os

rois = []
drawing = False
ix, iy = -1, -1
frame = None
temp_frame = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, frame, temp_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Select ROIs', temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        temp_frame = frame.copy()
        cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Select ROIs', temp_frame)
        name = input(f"Enter ROI name for rectangle ({x1},{y1})-({x2},{y2}): ")
        rois.append({
            "name": name,
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
        })
        print(f"ROI '{name}' added -- Press 'n' to add another or 's' to save and quit.")

def main():
    global frame, temp_frame

    if len(sys.argv) < 3:
        print("Usage: python roi_selector.py <IMAGE_PATH> <OUTPUT_ROI_JSON>")
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]
    OUTPUT_ROI_PATH = sys.argv[2]

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Could not read image: {IMAGE_PATH}")
        sys.exit(1)

    temp_frame = frame.copy()
    cv2.namedWindow('Select ROIs')
    cv2.setMouseCallback('Select ROIs', draw_rectangle)

    print("Draw each ROI with your mouse (click and drag).")
    print("After each ROI, enter its name in the terminal.")
    print("Press 'n' to draw the next ROI, 's' to save and exit.")

    while True:
        cv2.imshow('Select ROIs', temp_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            temp_frame = frame.copy()
        elif key == ord('s'):
            break
        elif key == 27:  # ESC to quit without saving
            print("Exited without saving.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    with open(OUTPUT_ROI_PATH, "w") as f:
        json.dump(rois, f, indent=2)
    print(f"Saved {len(rois)} ROIs to {OUTPUT_ROI_PATH}")

if __name__ == "__main__":
    main()
