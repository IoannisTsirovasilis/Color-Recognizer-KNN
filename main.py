import numpy as np
from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows

# BGR
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
LIGHT_GRAY = [224, 224, 224]
GRAY = [128, 128, 128]
DARK_GRAY = [64, 64, 64]
RED = [0, 0, 255]
PINK = [208, 96, 255]
PURPLE = [255, 32, 160]
LIGHT_BLUE = [255, 208, 80]
BLUE = [255, 32, 0]
YELLOW_GREEN = [128, 255, 96]
GREEN = [0, 192, 0]
YELLOW = [32, 224, 255]
ORANGE = [16, 160, 255]
BROWN = [96, 128, 160]
PALE_PINK = [160, 208, 255]
CYAN = [255, 255, 0]

COLORS = np.array([BLACK, WHITE, LIGHT_GRAY, GRAY, DARK_GRAY, RED, PINK, PURPLE, LIGHT_BLUE, BLUE, YELLOW_GREEN,
                   GREEN, YELLOW, ORANGE, BROWN, PALE_PINK, CYAN])
LABELS = np.array(["Black", "White", "Light Gray", "Gray", "Dark Gray", "Red", "Pink", "Purple", "Light Blue",
                   "Blue", "Yellow Green", "Green", "Yellow", "Orange", "Brown", "Pale Pink", "Cyan"])

cap = VideoCapture(0)
print (cap.isOpened())
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    diff = ((frame[:, :, :, None] - COLORS.T)**2).sum(axis=2)
    index = diff.argmin(axis=2)
    index = index.flatten()
    print(LABELS[np.bincount(index).argmax()])

    # Display the resulting frame
    imshow('frame', frame)
    if waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
destroyAllWindows()
