import time
import cv2
import mss
import numpy as np
from directInput import PressKey, ReleaseKey, W, A, S, D

# Define part of the screen to capture
# GTA V resolution must be 1024x768 and placed at top left of screen.
HEIGHT = 768
WIDTH = 1024
monitor = {'top': 40, 'left': 0, 'width': WIDTH, 'height': HEIGHT}

# Define a gap so that lines corresponding to window edges aren't detected.
window_gap = 10

# Defined as [X,Y] points, rather than [Y,X] as usual with images.
lane_roi_vertices = np.array([[window_gap, HEIGHT-window_gap*4], [window_gap, HEIGHT*2/3], [WIDTH*1/3, HEIGHT*1/3],
                              [WIDTH*2/3, HEIGHT*1/3], [WIDTH-window_gap, HEIGHT*2/3],
                              [WIDTH-window_gap, HEIGHT-window_gap*4]], np.int32)

# Blackens out the entire image with the exception of the region of interest specified by polygon vertices.
def img_roi(img, vertices):
    # Matrix of zeros with same size as image.
    mask = np.zeros_like(img)

    # Fill in the region of interest with white (full color).
    cv2.fillPoly(mask, vertices, [255, 255, 255])

    # Now bitwise AND the image with the given region of interest. Since the mask is white then
    # colors are preserved.
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

# Draws the lines parametrized by two points in the given image.
def draw_lines_in_img(image, lines):
    if lines is not None:
        for line in lines:
            coords = line[0] # Line format: [[x1,y1,x2,y2]]
            cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), [0, 255, 0], 4)

# Given lines parametrized by two points, calculates the pair most likely to correspond to a lane.
def get_best_lane(lines):
    pass

def process_img(img):
    # Convert image to grayscale.
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Get edges for easier line detection.
    edge_img = cv2.Canny(gray_img, 100, 150)

    # Keep edges only in region of interest (likely to correspond to lanes). Gets rid of lines corresponding to
    # electricity poles, mountains, the horizon, etc.
    lane_edges = img_roi(edge_img, [lane_roi_vertices])

    # Detect lines and draw them on the original image.
    lines = cv2.HoughLinesP(image=lane_edges,
                            rho=1, # Resolution
                            theta=np.pi/180, # Resolution
                            threshold=50,
                            minLineLength=100,
                            maxLineGap=15)
    draw_lines_in_img(img, lines)

    # Calculate which pair (if any) of lines is most likely to correspond to the lane.
    # line_l, line_r = get_best_lane(lines)

    return img

# Counts down starting from a given time.
def countdown(timer):
    for i in list(range(timer))[::-1]:
        print(i+1)
        time.sleep(1)

def main():
    with mss.mss() as sct:
        # Count down to give user time to prepare game window.
        countdown(2)

        while True:
            last_time = time.time()

            # Get raw pixels from the screen, save it to a numpy array.
            img = np.array(sct.grab(monitor))

            processed_img = process_img(img)

            # Display image.
            cv2.imshow('screenshot', processed_img)

            print('fps: {0}'.format(1 / (time.time() - last_time)))

            # Press "q" to quit (bitwise AND so that CAPS LOCK Q works as well)
            k = cv2.waitKey(10) & 0b11111111
            if k == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main()