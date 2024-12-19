

## Set up your hardware

Before you begin, you need to
[set up your Raspberry Pi](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up)
with Raspberry Pi OS (preferably Raspberry Pi OS (Legacy, 64-bit)).

You also need to
[connect and configure the Pi Camera](https://www.raspberrypi.org/documentation/configuration/camera.md)
if you use the Pi Camera. This code also works with USB camera connect to the
Raspberry Pi.

And to see the results from the camera, you need a monitor connected to the
Raspberry Pi. It's okay if you're using SSH to access the Pi shell (you don't
need to use a keyboard connected to the Pi)—you only need a monitor attached to
the Pi to see the camera stream.

## Setup Software for Running code

Update and install necessary your Raspberry pi OS:
``` 
sudo apt-get update
sudo apt-get upgrade -y
```
Upgrade python-pip and use the latest pip:
```
sudo apt install python3-pip -y
pip install -U pip
```
Create and activate a virtualenv then install the necessary packages:

```
pip install ultralytics[export]
```
Now clone this Git repo onto your Raspberry Pi and install project specific libraries:

```
git clone https://github.com/Subhasree4921/vehicle-speed-tracker-NCNN
cd vehicle-speed-tracker-NCNN
pip install -r requirements.txt
```
###Running the code 
```
python detect.py
```
