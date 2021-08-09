"""
Convert 'labelme' project's output data to a dataset for OCR model training.


## References
1. 'labelme' project: https://github.com/wkentaro/labelme


## Procedures
1. Read 'label' and 'bbox' from json file.
2. Crop the image using 'bbox' and save it.
3. Write a label for each cropped image to the label file.


## Usage example:
    python3 convert.py \
            --input_path ./input \
            --output_path ./output


## Input data structure:
    /input
    ├── image00001.png
    ├── image00001.json
    ├── image00002.png
    ├── image00002.json
    └── ...

* For the 'images.json' file structure, refer to the 'https://github.com/wkentaro/labelme'


## Output data structure:

    /output
    └── /images
        #   [filename]_[idx].[ext]
        ├── image00001_00001.png
        ├── image00001_00002.png
        ├── image00002_00001.png
        ├── image00002_00002.png
        ├── ...
        └── labels.txt


* Label file structure:

    # {filename}\t{label}\n
      image00001_00001.png	abcd
      image00001_00002.png	efgh
      image00002_00001.png	ijkl
      image00002_00002.png	mnop
      ...

"""

import os
import sys
import json
import time
import bisect
import shutil
import argparse

from PIL import Image


def run(input_path, output_path, log):
    """ Convert 'labelme' project's output data to a dataset for OCR model training. """

    if not os.path.exists(input_path):
        sys.exit(f"Can't find '{os.path.abspath(input_path)}' directory.")

    if os.path.isdir(output_path):
        sys.exit(f"'{os.path.abspath(output_path)}' directory is already exists.")
        # print(f"'{os.path.abspath(output_path)}' directory is already exists.")
        # shutil.rmtree(output_path)
    else:
        os.makedirs(output_path)

    files, count, json_files, json_count, image_files, image_count = get_files(input_path)

    if json_count != image_count:
        sys.exit(f"The number of json files and image files does not match exactly")

    start_time = time.time()
    labels = open(os.path.join(output_path, "labels.txt"), "w", encoding="utf8")

    digits = len(str(image_count))
    for ii, image_file in enumerate(image_files):
        if (ii + 1) % 100 == 0:
            print(("\r%{}d / %{}d Processing !!".format(digits, digits)) % (ii + 1, count), end="")

        filename, ext = os.path.splitext(image_file)
        # print(f"image_file: {image_file}, filename: {filename}")

        json_file = get_json_file(json_files, json_count, filename)
        if json_file is None:
            print(f"Can't find '{filename}' json file.")
            continue

        with open(os.path.join(input_path, json_file)) as f:
            json_data = json.load(f)

        with Image.open(os.path.join(input_path, image_file)) as img:
            for jj, shape in enumerate(json_data["shapes"]):
                label = shape["label"]
                bbox = get_bbox(shape["points"])

                if bbox[0] >= img.size[0] or bbox[1] >= img.size[1] or bbox[2] >= img.size[0] or bbox[3] >= img.size[1]:
                    # print(f"{image_file} {label} label's bbox error: {bbox}")
                    log.write(f"'{image_file}'s {label} bbox: '{bbox}'\n")
                    continue

                output_file = f"{filename}_{jj:03d}{ext}"
                crop_image = img.crop(bbox)
                crop_image.save(os.path.join(output_path, output_file))

                labels.write(f"{output_file}\t{label}\n")

    elapsed_time = (time.time() - start_time) / 60.
    print("\n- processing time: %.1fmin" % elapsed_time)

    labels.close()


def get_json_file(json_files, json_count, filename):
    """ Search for json file with the same name as image file """

    idx = bisect.bisect(json_files, filename)

    if idx < 0 or idx >= json_count:
        return None

    return json_files[idx]


def get_bbox(points):
    left = int(min(points[0][0], points[1][0]))
    upper = int(min(points[0][1], points[1][1]))
    right = int(max(points[0][0], points[1][0]))
    lower = int(max(points[0][1], points[1][1]))

    bbox = [left, upper, right, lower]
    # print(f"bbox: {bbox}")

    return bbox


def get_files(path, except_file=""):
    file_list = []
    json_files = []
    image_files = []

    for file in os.listdir(path):
        if file.startswith(".") or file == os.path.basename(except_file):
            print('except file: ', file)
            continue

        file_list.append(file)
        _, ext = os.path.splitext(file)
        if ext == ".json":
            json_files.append(file)
        elif ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
            image_files.append(file)
        else:
            sys.exit(f"Invalid file '{file}'.")

    file_list.sort()
    json_files.sort()
    image_files.sort()

    return file_list, len(file_list), json_files, len(json_files), image_files, len(image_files)


def create_working_directory(root, sub_dirs=None):
    dirs = [root]
    os.makedirs(root)
    for sub in sub_dirs:
        path = os.path.join(root, sub)
        dirs.append(path)
        os.makedirs(path)

    return dirs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert 'labelme' project's output data to a dataset for OCR model training")

    parser.add_argument("--group_name", type=str, help="Define the name of the group to be converted")
    parser.add_argument("--input_path", type=str, required=True, help="Data path of 'labelme' project's output data")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Data path for a datasest use in OCR model training")

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    arguments = parse_arguments()

    if arguments.group_name:
        groups = [arguments.group_name]
    else:
        groups = [g for g in os.listdir(arguments.input_path)]

    if os.path.isdir(arguments.output_path):
        sys.exit(f"'{os.path.abspath(arguments.output_path)}' directory is already exists.")
    else:
        os.makedirs(arguments.output_path)

    error_log = open(os.path.join(arguments.output_path, "error.txt"), 'a')
    dashed_line = '-' * 80

    for ii, group in enumerate(groups):
        print("\n[%d/%d] group: '%s'" % (ii+1, len(groups), group))
        print("-" * 40)

        error_log.write(f"Convert error logs of '{group}'\n")
        error_log.write(dashed_line + '\n')

        input_path = os.path.join(arguments.input_path, group)
        output_path = os.path.join(arguments.output_path, group)

        run(input_path, output_path, error_log)

        error_log.write(dashed_line + '\n')

    error_log.close()
