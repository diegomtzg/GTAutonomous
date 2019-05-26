import time
import cv2
import mss
import numpy as np

def process_img(original_img):
    # Convert image to grayscale.
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # Get edges.
    edge_img = cv2.Canny(gray_img, 100, 200)

    return edge_img


with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {'top': 50, 'left': 0, 'width': 1250, 'height': 1000}

    while True:
        last_time = time.time()

        # Get raw pixels from the screen, save it to a numpy array.
        img = np.array(sct.grab(monitor))

        # Process image
        processed_img = process_img(img)

        # Display image
        cv2.imshow('screenshot', processed_img)

        print('fps: {0}'.format(1 / (time.time()-last_time)))

        # Press "q" to quit (bitwise AND so that CAPS LOCK Q works as well)
        k = cv2.waitKey(10) & 0b11111111
        if k == ord('q'):
            cv2.destroyAllWindows()
            break