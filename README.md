# Stopped Vehicle Detection - YOLO
Practical system using Python and OpenCV to detect stationary vehicles in a video stream. Leveraging YOLO open source models and object tracking algorithms to identify stationary vehicles by marking them with red bounding boxes while distinguishing moving vehicles with green boxes.

## How to run?
### 1. Clone this repository
git clone git@github.com:CodeRic28/stopped_vehicle_detection.git

### 2. Change directory into repository
cd stopped_vehicle_detection

### 3. Install requirements
pip install -r requirements.txt

### 4. Add videos that you want to detect in the "Videos" directory

### 5. Use the following command to run the system
python main.py --input Videos/<name-of-the-input-file.extension> --output output/<name-of-the-output-file.mp4>

#### Make sure you use ".mp4" extension for the output file.


