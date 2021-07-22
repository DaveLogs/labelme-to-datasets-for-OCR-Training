# labelme-to-datasets-for-OCR-model-training

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
    ├── /images
    │   #   [filename]_[idx].[ext]
    │   ├── image00001_00001.png
    │   ├── image00001_00002.png
    │   ├── image00002_00001.png
    │   ├── image00002_00002.png
    │   └── ...
    │
    └── labels.txt

* Label file structure:

    # {filename}\t{label}\n
      image00001_00001.png	abcd
      image00001_00002.png	efgh
      image00002_00001.png	ijkl
      image00002_00002.png	mnop
      ...
