import os

import cv2
from tqdm import tqdm


class VideoCreator:
    def __init__(self, imgs_dir, vid_name):
        self.imgs_dir = imgs_dir
        self.img_array = []
        self.video_filename = vid_name

    def preprocess_images(self):
        filenames = sorted(os.listdir(self.imgs_dir))
        print("Adding images...")
        for filename in tqdm(filenames):
            complete_filename = self.imgs_dir + "/" + filename
            img = cv2.imread(complete_filename)
            height, width, _ = img.shape
            size = (width, height)
            self.img_array.append(img)

        return size

    def create_video(self, fps=20):
        size = self.preprocess_images()
        out = cv2.VideoWriter(
            self.video_filename, cv2.VideoWriter_fourcc(*"MJPG"), fps, size
        )
        print("Recording video...")
        for i in tqdm(range(len(self.img_array))):
            out.write(self.img_array[i])
        out.release()
        print("Done.")


if __name__ == "__main__":
    creator = VideoCreator("/data3/airbus/onboard_vid/person0.05", "test_vid.avi")
    creator.create_video()
