import cv2
import os
import handCapture as htm


class imageProcessor():
    def __init__(self, handDetector):
        self.handDetector = handDetector

    def extractLm(self, image, idx, folder):
        image = cv2.imread(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect hand and save lm drawn images
        img = self.handDetector.findHands(image_rgb)
        # extract lms
        lms = self.handDetector.findPosition(img, draw = False)

        if (lms != []):
            filename = f"detected_a_{idx}.jpg"
            output_path = os.path.join(folder, filename)
            cv2.imwrite(output_path, img)
            # print(lms)
            return True, lms

        return False, lms

# use the following to test detection
def main():
    detector = htm.handCapture(detectionCon=0.2)
    processor = imageProcessor(detector)

    input_folder = './A'    # for privacy reason photos will not be uploaded on github
    output_folder = './A_detected'
    all_lms = []

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder)]

    index = 1
    # Loop through each image in the input folder
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        detected, lms = processor.extractLm(image_path, index, output_folder)
        if (detected):
            # flatten landmarks for each sample
            flatten_lms = [item for lm in lms for item in lm]    # current flattend for MLP, will explore CNN later without flatenning due to features are highly correlated
            all_lms.append(flatten_lms)
            index += 1

    # print(len(all_lms))
if __name__ == "__main__":
    main()