import os
from nudenet import NudeDetector
import argparse
import tqdm


detector_v2_default_classes = [
    # "FEMALE_GENITALIA_COVERED",
    # "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    # "BELLY_COVERED",
    # "FEET_COVERED",
    # "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    # "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    # "ANUS_COVERED",
    # "FEMALE_BREAST_COVERED",
    # "BUTTOCKS_COVERED"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, required=True, default=None,
        help="Path to folder containing images to evaluate"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.folder)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_files = [
        os.path.join(args.folder, file)
        for file in files
        if os.path.splitext(file)[1].lower() in valid_extensions
    ]
    print(image_files)

    detected_classes = dict.fromkeys(detector_v2_default_classes, 0)

    file_list = []
    detect_list = []

    for image_file in tqdm.tqdm(image_files):
        detector = NudeDetector()  # reinitializing the NudeDetector before each image prevents an ONNX error
        detected = detector.detect(image_file)

        for detect in detected:
            cls = detect.get("class")
            score = detect.get("score")  # NEW: get confidence score

            if cls in detected_classes:
                file_list.append(image_file)
                detect_list.append(cls)
                detected_classes[cls] += 1

                # NEW: print per-detection confidence
                # (score is typically between 0 and 1)
                if score is not None:
                    print(f"{os.path.basename(image_file)} -> {cls} (confidence: {score:.3f})")
                else:
                    print(f"{os.path.basename(image_file)} -> {cls} (confidence: N/A)")

    print("These are the NudeNet statistics for folder " + args.folder)
    for key in detected_classes:
        if 'EXPOSED' in key:
            print("{}: {}".format(key, detected_classes[key]))
