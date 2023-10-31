import argparse
import os

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

from images_to_video import VideoCreator
from video_to_images import ImageCreator


class Detector:
    """
    Maximum 14 texts
    Colors in (B,G,R) format
    """

    def __init__(
        self,
        owl_model="Owl-ViT",
        imgs_dir="images",
        texts=["person"],
        thresholds=[0.1],
        box_thickness=2,
        save_model=True,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.img_dir = imgs_dir
        model_dir = f"./test/saved_models/{owl_model}"
        if owl_model == "Owl-ViT":
            from transformers import OwlViTForObjectDetection, OwlViTProcessor

            model_url = "google/owlvit-base-patch32"
        elif owl_model == "Owlv2":
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            model_url = "google/owlv2-base-patch16-ensemble"
        else:
            raise ValueError("Wrong model name.")

        if os.path.exists(model_dir):
            model_exist = True
        else:
            os.makedirs(model_dir)
            model_exist = False
        if owl_model == "Owl-ViT":
            if model_exist:
                self.processor = OwlViTProcessor.from_pretrained(model_dir)
                self.model = OwlViTForObjectDetection.from_pretrained(model_dir)
            else:
                self.processor = OwlViTProcessor.from_pretrained(model_url)
                self.model = OwlViTForObjectDetection.from_pretrained(model_url)
        elif owl_model == "Owlv2":
            if model_exist:
                self.processor = Owlv2Processor.from_pretrained(model_dir)
                self.model = Owlv2ForObjectDetection.from_pretrained(model_dir)
            else:
                self.processor = Owlv2Processor.from_pretrained(model_url)
                self.model = Owlv2ForObjectDetection.from_pretrained(model_url)
        if save_model:
            self.processor.save_pretrained(model_dir)
            self.model.save_pretrained(model_dir)
        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (0, 204, 255),
            (0, 127, 255),
            (0, 255, 0),
            (210, 242, 31),
            (138, 221, 242),
            (200, 152, 234),
            (5, 35, 84),
            (224, 26, 168),
            (193, 229, 87),
            (48, 156, 168),
            (45, 136, 255),
            (155, 155, 155),
        ]
        self.texts = texts
        self.box_colors = {texts[i]: colors[i] for i in range(len(texts))}
        self.object_threshold = {texts[i]: thresholds[i] for i in range(len(texts))}
        self.detection_count = {k: [0] for k in self.texts}
        self.box_thickness = box_thickness
        self.current = 0

    def process_image(self, img_filename):
        image_path = self.img_dir + "/" + img_filename
        image = Image.open(image_path)
        inputs = self.processor(text=self.texts, images=image, return_tensors="pt").to(
            self.device
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs

    def detect(self, results):
        logits = torch.max(results["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()

        labels = logits.indices.cpu().detach().numpy()
        boxes = results["pred_boxes"][0].cpu().detach().numpy()

        return boxes, scores, labels

    def draw_bboxes(self, img_filename, boxes, scores, labels):
        image_path = self.img_dir + "/" + img_filename
        image = cv2.imread(image_path)
        detected = 0
        dic_count = {k: 0 for k in self.texts}

        for box, score, label in zip(boxes, scores, labels):
            # box = [int(i*image.shape[0]) for i in box.tolist()] #works because square imgs
            box = box.tolist()
            box[0] *= image.shape[1]
            box[1] *= image.shape[0]
            box[2] *= image.shape[1]
            box[3] *= image.shape[0]
            if score >= self.object_threshold[self.texts[label]]:
                detected += 1
                dic_count[self.texts[label]] += 1

                x0 = int(box[0] - box[2] / 2)
                y0 = int(box[1] - box[3] / 2)
                x1 = int(x0 + box[2])
                y1 = int(y0 + box[3])

                image = cv2.rectangle(
                    image,
                    (x0, y0),
                    (x1, y1),
                    self.box_colors[self.texts[label]],
                    self.box_thickness,
                )
                image = cv2.putText(
                    img=image,
                    text=f"{self.texts[label]}: {score:1.2f}",
                    org=(x0, y1 + 15),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),  # self.box_colors[self.texts[label]],
                    thickness=1,
                )
        for key, value in dic_count.items():
            self.detection_count[key].append(value)
        return image, detected

    def save_image(self, image, save_to):
        cv2.imwrite(save_to, image)

    def detect_folder(self, save_to):
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        filenames = sorted(os.listdir(self.img_dir))
        pbar = tqdm(filenames)
        detected_data = []
        for filename in pbar:
            results = self.process_image(img_filename=filename)
            boxes, scores, labels = self.detect(results)
            image, detected = self.draw_bboxes(filename, boxes, scores, labels)
            detected_data.append(detected)
            self.save_image(image, f"{save_to}/d_{filename}")
            pbar.set_description(f"detected {detected} objects.")

        self.detection_count["total"] = detected_data

    def plot_data(self, save_to):
        directory = f"{save_to}_plots"
        if not os.path.exists(directory):
            os.makedirs(directory)
        print("Creating charts...")
        for label, count in tqdm(self.detection_count.items()):
            if label != "total":
                rgb = list(self.box_colors[label])
                rgb.reverse()
                plt.figure(figsize=(16, 9))
                plt.plot(count, c=tuple([round(x / 255, 2) for x in rgb]))
                plt.ylabel(f"Number of {label}s detected")
                plt.xlabel("Frame id")
                plt.title(label)
                plt.savefig(f"{directory}/{save_to}_{label}_plot.png")
            else:
                plt.figure(figsize=(16, 9))
                plt.plot(count)
                plt.ylabel("Total detected")
                plt.xlabel("Frame id")
                plt.title("Total")
                plt.savefig(f"{directory}/{save_to}_total_plot.png")

    def to_video(self, saved_to, video_name, fps):
        creator = VideoCreator(saved_to, video_name)
        creator.create_video(fps=fps)


def main(
    owl_model,
    images_dir,
    output_name,
    fps,
    process_vid,
    video_filename,
    image_start,
    image_end,
    texts,
    thresholds,
    box_thickness,
    save_model,
):
    if process_vid:
        if video_filename is not None:
            image_maker = ImageCreator(
                video_filename, images_dir, image_start, image_end
            )
            image_maker.get_images()
            cap = cv2.VideoCapture(video_filename)
            fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            raise Exception("Please provide a valid video filename.")
    else:
        if fps is None:
            fps = 20

    detector = Detector(
        owl_model, images_dir, texts, thresholds, box_thickness, save_model
    )
    print(
        "Detector for {} with thresholds {}".format(
            detector.texts, list(detector.object_threshold.values())
        )
    )
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    detector.detect_folder(save_to=output_name)
    detector.to_video(output_name, output_name + ".avi", fps)
    detector.plot_data(output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects in a video")
    parser.add_argument(
        "--owl_model",
        type=str,
        default="Owl-ViT",
        choices=["Owl-ViT"],  # , "Owlv2"], put back when owlv2 from HF is fixed.
        help="Which version of Owl model to use",
    )
    parser.add_argument(
        "--imgs_dir",
        type=str,
        required=True,
        help="The directory containing the images.",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        required=True,
        help="the directory in which to save the processed images",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Number of frames per second for the output video (default: None, determined automatically if --process_video).",
    )
    parser.add_argument(
        "--process_video",
        action="store_true",
        default=False,
        help="Whether to get images from a video or not (default: False).",
    )
    parser.add_argument(
        "--video_filename",
        type=str,
        default=None,
        help="Name of video file to process (default: None).",
    )
    parser.add_argument(
        "--image_start", type=int, default=0, help="Frame to start from (default:0)."
    )
    parser.add_argument(
        "--image_end", type=int, default=0, help="Frame to end with (default: last (0)."
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        type=str,
        default=[
            "person",
            "face mask",
            "backpack",
            "coat",
            "head",
            "shoe",
            "hand",
            "ear",
            "beard",
            "arm",
            "knee",
            "leg",
            "jacket",
            "train seat",
        ],
        help="A list of texts to detect in the images (default: 14 random texts).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[
            0.08,
            0.015,
            0.025,
            0.06,
            0.02,
            0.02,
            0.02,
            0.02,
            0.03,
            0.025,
            0.015,
            0.01,
            0.045,
            0.072,
        ],
        help="A list of thresholds between 0 and 1 (default: 14 low thresholds for the texts).",
    )
    parser.add_argument(
        "--box_thickness",
        type=int,
        default=2,
        help="The thickness of the bounding boxes to draw (default: 2).",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Whether to save the pretrained model locally or not (default: False).",
    )

    args = parser.parse_args()

    main(
        owl_model=args.owl_model,
        images_dir=args.imgs_dir,
        output_name=args.save_to,
        fps=args.fps,
        process_vid=args.process_video,
        video_filename=args.video_filename,
        image_start=args.image_start,
        image_end=args.image_end,
        texts=args.texts,
        thresholds=args.thresholds,
        box_thickness=args.box_thickness,
        save_model=args.save_model,
    )
