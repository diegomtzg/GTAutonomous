import time
import cv2
import mss
import numpy as np
from directInput import PressKey, ReleaseKey, W, A, S, D

def process_img(original_img):
    # Convert image to grayscale.
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # Get edges for easier line detection.
    edge_img = cv2.Canny(gray_img, 100, 200)

    # Detect lines.
    lines = cv2.HoughLinesP(image=edge_img,
                            rho=1, # Resolution
                            theta=np.pi/180, # Resolution
                            threshold=200,
                            minLineLength=30,
                            maxLineGap=15)
    draw_lines_in_img(original_img, lines)

    return original_img

# Draws the lines parametrized by two points in the given image.
def draw_lines_in_img(image, lines):
    for line in lines:
        coords = line[0]
        cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), [0, 255, 0], 3)

# Counts down starting from a given time.
def countdown(timer):
    for i in list(range(timer))[::-1]:
        print(i+1)
        time.sleep(1)

def main():
    with mss.mss() as sct:
        # Define part of the screen to capture
        # GTA V resolution must be 1024x768 and placed at top left of screen.
        monitor = {'top': 40, 'left': 0, 'width': 1024, 'height': 768}

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