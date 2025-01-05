import cv2
import time

# used this script to take 30 photo capture of my hand making sign language gesture and saved locally for training purpose

# Open the video capture
cap = cv2.VideoCapture(1)

# Set the image count limit
image_count = 0
sTime = 0

while True:
    pTime = time.time()
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)
    cv2.waitKey(1)
    if (pTime - sTime > 5):
        filename = f"image_a_{ image_count+ 1}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        image_count += 1
        sTime = pTime
    
    if (image_count == 60):
        break
cap.release()
