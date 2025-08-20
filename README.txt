=================================
Training Script Functionalities:
=================================
	- Automatically aquires the most recent YOLO nano model (yolo11n.pt) to retreive pre-trained weights for transfer learning (--train), 
	  or the model backbone (yolo11.yaml) for training from scratch (--scratch-train).
	  
	- Accessible model updating feature (--auto-train) that looks for the most recent
	  best.pt file and only updates with previously trained weights with new training data.
	  
	- Organized output structure based on timestamps and training type.
	
	- Inclusion of a log folder that includes a summary text document, the full
	  results.csv, and a metadata.json file for tracking model updates.
	  (...\logs\(runs / test-runs)\train (mm-dd-yyyy hh-mm-ss)
	  
	- Catered toward small object detection models.
	
	- Can be launched from an accessible .bat file.
	
	- Downloads all requirements automatically.
	
	- Test models are entirely separated from the base models that will be used.

--------------------
Training Arguments:
--------------------

## To run the validation split script:
python train_val_split.py

## To update the most recently generated best.pt file:
python train_start.py --auto-train

## To run fresh training taking advantage of transfer learning:
python train_start.py --train

## To run training completely from scratch:
python train_start.py --scratch-train

-------------------------
Test Training Arguments:
-------------------------

## To test an update to the most recently generated best.pt file:
python train_start.py --test-auto-train

## To run a test that trains fresh taking advantage of transfer learning:
python train_start.py --test-train

## To run a test that trains completely from scratch:
python train_start.py --test-scratch-train

================================
Detection Script Functionality:
================================

	- Allows for either standard detections (--detect), or option for testing (--test-detect).
		This automatically looks for the best.pt file in the appropriate runs / test-runs folder.
	  
	- Multiple source input compatibility. USB and PI cameras are supported, along with video input.
		Output recordings of processed sources are in associated model logs folder. 
		They are recorded as a timestamp from when the detection first began. (mm-dd-yyyy hh-mm-ss)
		(...\logs\(test-runs / runs)\train (timestamp)\recordings\(usb / picamera / video-input)
	
	- Supports running multiple cameras of either type alongside video input. (--sources picamera0 usb0 C:\path\to\video.type)
		Independent windows for each source, each with their own FPS, object detection, and generic interaction conters.
		Auto scales windows depending on number of sources, with a supported number of 4.
		Windows are labeled accordingly and can also be adjusted manually like most other programs.
	  
	- Class-to-class interactions are recorded in a .csv.
		Recorded as a timestamp from when the detection first began and ended. (mm-dd hh-mm-ss to mm-dd hh-mm-ss)
		Standard detections save a checkpoint every 1 hour, test detections every minute.
		Records duration of the interaction (in s), how many frames it lasted, and the timestamp from when it began to when it ended.
		A 60 frame temporal buffer to count as an interaction for standard use. (~3 sec at 20 fps)
			15 frame temporal buffer to count as an interaction for test use. (>1 sec at 20 fps)
		(...\logs\(test-runs / runs)\train (timestamp)\interaction-metrics\(usb / picamera / video-input)

	- Adjusted temporal smoothing parameters for bounding boxes.
		Ultralytics default parameters do not use temporal smoothing options.
		The included lab mode (--lab) adjusts these paremeters to be more optimal.
		Can be manually adjusted with arguments. (--smooth (0.0-1.0), --max_history (0-5), --dist-thresh (0-100))
		In-short:
			* smooth adjusts the balance between prioritizing objects on past or current frames
			* max_history adjusts how many frames are stored in smoothing process
			* dist_thresh adjusts smoothing of objects depending on relative past to current position
			
---------------------
Detection Arguments:
---------------------

## To run the detection script with preferred settings:
python detect.py --detect --lab --sources (picamera0, usb0, C:\path\to\video.type)

## To run the detection script with Ultralytics default settings:
python detect.py --detect --sources (picamera0, usb0, C:\path\to\video.type)

## To test the detection script with preferred settings:
python detect.py --test-detect --lab --sources (picamera0, usb0, C:\path\to\video.type)

-----------------
Setup Arguments:
-----------------
conda activate yolo-env
cd C:\Users\TheGo\Documents\YOLO

ultralytics>=8.0.100
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.23.0
Pillow>=9.4.0
tqdm>=4.64.0
opencv-python>=4.7.0
matplotlib>=3.7.0
pandas>=1.5.0