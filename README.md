We have put the video inputs and outputs [here](https://drive.google.com/drive/folders/1L0Fxo8HGZE2T4HVAyYEtuRa9vectINgY?usp=sharing) 

## Introduction
### Face mask detection for video stream
- The project involves detecting face mask for an inputted mp4 video.
- MTCNN is used for face detection.
- Tensorflow is used for implementing the neural network.
- Main libraries that have been used are src, opencv, tensorflow

* **Dependencies:**
	* Python 3.5+
	* Tensorflow
	* [**MTCNN**](https://github.com/davidsandberg/facenet/tree/master/src/align)
	* Scikit-learn
	* Numpy
	* Numba
	* Opencv-python
	* Filterpy

##### Instructions for executing the project -
- The videos to be inputted must be present in the directory named 'videos'
- Open the terminal in the directory named 'face_mask_detection'
- Install all the dependencies using `pip3 install -e`
- Start the project using the command 
```sh
python3 start.py
```

* The input videos should be present in `videos` folder
* The output videos will be generated in the main folder with the same name as the input video and format will be `.mp4`
* The csv and txt files for the videos are stored in the main falder with the same name as the input video and extension will be `.csv` and `.txt` respectively
