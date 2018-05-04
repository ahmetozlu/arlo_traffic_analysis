# VEHICLE DETECTION, TRACKING AND COUNTING
This project focuses on "Vechicle Detection, Tracking and Counting" on [Arlo Wireless & AC-Powered Security Cameras](https://www.arlo.com/en-us/) by using [Background Subtraction Algortihm](https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html) that is provided [OpenCV](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html)

This project has more than just counting vehicles, here are the additional capabilities of this project;

- Recognition of approximate vehicle color (developing is on progresss, will be available soon)
- Detection of vehicle's direction of travel
- Prediction the speed of the vehicle
- Prediction of approximate vehicle size (developing is on progresss, will be available soon)
- **The images of detected vehicles are cropped from video frame and they are saved as new images under "[detected_vehicles](https://github.com/ahmetozlu/arlo_traffic_analysis/tree/master/src/detected_vehicles)" folder path**
- **The program gives a .csv file as an output ([traffic_measurement.csv](https://github.com/ahmetozlu/arlo_traffic_analysis/blob/master/src/traffic_measurement.csv)) which includes "Vehicle Type/Size", " Vehicle Color", " Vehicle Movement Direction", " Vehicle Speed (km/h)" rows, after the end of the process for the source video file.**

The input video can be accessible by this [link](https://github.com/ahmetozlu/arlo_traffic_analysis/blob/master/src/bradley_input.mp4).

## Quick Demo

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/39646058-e4fddfb2-4fe2-11e8-99d7-72b472404112.gif">
</p>

## Theory

Background subtraction is a major preprocessing steps in many vision based applications. For example, consider the cases like visitor counter where a static camera takes the number of visitors entering or leaving the room, or a traffic camera extracting information about the vehicles etc. In all these cases, first you need to extract the person or vehicles alone. Technically, you need to extract the moving foreground from static background.

If you have an image of background alone, like image of the room without visitors, image of the road without vehicles etc, it is an easy job. Just subtract the new image from the background. You get the foreground objects alone. But in most of the cases, you may not have such an image, so we need to extract the background from whatever images we have. It become more complicated when there is shadow of the vehicles. Since shadow is also moving, simple subtraction will mark that also as foreground. It complicates things.

Several algorithms were introduced for this purpose. OpenCV has implemented three such algorithms which is very easy to use.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/39543256-5f8bfc6e-4e53-11e8-8fd6-fcf718c87d3f.gif">
</p>

## Project Demo

## Installation

**1.) Python and pip**

Python is automatically installed on Ubuntu. Take a moment to confirm (by issuing a python -V command) that one of the following Python versions is already installed on your system:

- Python 2.7

The pip or pip3 package manager is usually installed on Ubuntu. Take a moment to confirm (by issuing a *pip -V* or *pip3 -V* command) that pip or pip3 is installed. We strongly recommend version 8.1 or higher of pip or pip3. If Version 8.1 or later is not installed, issue the following command, which will either install or upgrade to the latest pip version:

    $ sudo apt-get install python-pip python-dev   # for Python 2.7
    
**2.) OpenCV**

See required commands to install OpenCV on Ubuntu in [here](https://gist.github.com/dynamicguy/3d1fce8dae65e765f7c4).

**3.) NumPy**

See required steps to install [NumPy](https://docs.scipy.org/doc/numpy-1.13.0/user/install.html)

## Citation
If you use this code for your publications, please cite it as:

    @ONLINE{vdtcbs,
        author = "Ahmet Özlü",
        title  = "Vehicle Detection, Tracking and Counting by Background Subtraction with OpenCV",
        year   = "2018",
        url    = "https://github.com/ahmetozlu/arlo_traffic_analysis"
    }

## Author
Ahmet Özlü

## License
This system is available under the MIT license. See the LICENSE file for more info.

