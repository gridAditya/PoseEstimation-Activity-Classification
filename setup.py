from setuptools import setup

with open('requirements.txt') as req:
    requirements = req.read().splitlines()

setup(
   name='Pose Estimation and Activity Classification',
   version='1.0',
   python_requires='>=3.10.13',
   packages=requirements,
   author='Aditya Vikram Singh',
   description = 'Pose Estimation and Activity Classification using yolov8-nano and EfficientNetB0')