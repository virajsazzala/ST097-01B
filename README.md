# ST097-01B

Visual Odometry implementation with Stereo cameras for localization in Autonomous Driving

## Overview
- It uses Python and OpenCV.
- It's a minimal approach w/o Global map optimization.

## Setup and Usage

1. **Download the Dataset:**

   - Download the full dataset from [KITTI's official website](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
   - Make sure to download the greyscale images and the ground truth poses dataset.
   - Extract it to the root directory of the project.

2. **Setup Virtual Environment:**

   - Navigate to your project's root directory.
   - Create a virtual environment using `venv`. Open a terminal and run the following commands:
  ```bash
  python3 -m venv venv
  source venv/bin/activate      # For Linux/Mac
  # OR
  .\venv\Scripts\activate      # For Windows
  ```

3. **Install Requirements:**

   - With the virtual environment activated, install the required packages by using the `requirements.txt` file. Run the following command in the terminal:
  ```bash
  pip install -r requirements.txt
  ```

4. **Run the Visualization:**
   - After installing the requirements, run the `main.py` file to visualize the results. Execute the following command in the terminal:
  ```bash
  python main.py
  ```
   - This should launch the visualization.

Now, your project should be set up and running :)

> Note: This is for Educational Purposes Only
