# Parallax Effect with Depth Map

This project demonstrates a parallax effect using a depth map and either mouse movement or eye tracking. It creates an interactive visualization where different parts of an image move based on their depth when the user moves their mouse or eyes.

## Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- Pygame
- dlib

You can install the required packages using:

```bash
pip install requirments.txt
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/sidNarasimhan/parallax-effect-depth-map.git
   cd parallax-effect-depth-map
   ```

2. Add your image:
   - Place your image file (e.g., `image.jpg`) in the project directory.

3. Add a depth map:
   - Place a corresponding depth map file (e.g., `depth.png`) in the project directory.
   - The depth map should be a grayscale image where lighter pixels represent areas closer to the camera and darker pixels represent areas farther from the camera.

4. Download the shape predictor file:
   - Download the `shape_predictor_68_face_landmarks.dat` file and place it in the project directory. You can find this file in the dlib library resources.

## Generating Depth Maps

If you don't have a depth map for your image, you can generate one using the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) project. Follow these steps:

1. Clone the Depth-Anything repository:
   ```
   git clone https://github.com/LiheYoung/Depth-Anything.git
   cd Depth-Anything
   ```

2. Follow the installation instructions in their README.

3. Use their `run.py` script to generate a depth map for your image:
   ```
   python run.py --encoder vitl --img-path path/to/your/image.jpg --outdir output_directory
   ```

4. Copy the generated depth map from the output directory to your parallax effect project directory and rename it to `depth.png`.

## Usage

Run the script:
```bash
python app.py
```

- The program will open a window displaying your image with the parallax effect.
- Move your mouse to see the parallax effect in action.
- Press the spacebar to toggle between mouse control and eye tracking (requires a webcam).
- Press ESC to exit the program.

## Customization

- Adjust the `max_offset` parameter in the `apply_depth_effect` function call to control the intensity of the parallax effect.
- Modify the `zoom` factor in the `render_image` function to change the zoom level of the displayed image.

## License

This project is open source and available under the [MIT License](LICENSE).
