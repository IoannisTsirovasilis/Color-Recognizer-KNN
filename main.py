import numpy as np
from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows

COLORS = []
LABELS = []
with open("colors.txt", "r") as f:
    colors = f.readlines()

for c in colors:
    tmp = np.asarray(c.split(','))
    color = np.asarray([int(x) for x in tmp[:-1]])
    label = tmp[-1].replace('\n', '')
    LABELS.append(label)
    COLORS.append(color)

COLORS = np.asarray(COLORS)
cap = VideoCapture(0)
print(cap.isOpened())
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
