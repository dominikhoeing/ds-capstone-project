# DS-Capstone-Project

## Introduction

This repository contains the capstone project created by Tetyana Samoylenko, Alexander Simakov and Dominik Höing during the last month of the "Data Science & AI Bootcamp" by [neuefische](https://www.neuefische.de/), in Hamburg, Germany between July-September 2024.

### The Team

| Member | Background |
| ------ | ---------- |
| Tetyana Samoylenko | Mathematics |
| Alexander Simakov | Sound Engineering |
| Dominik Höing | Chemistry/Phsics |

## The topic

This project was a challenge posted by DrivenData.org called *Conser-vision Practice Area: Image Classification*. You can find the challenge [here](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/page/409/).

The project is about **image classification using artificial neural networks**. Researchers in nature reserves make extensive use of photo traps to observe the population of animals. These photo traps work autonomously and usually take a picture, if a motion sensor detects movement in front of the camera. Over the course of a study, reserachers take tens of thousands of images, which they need to sort and label in order to estimate the population of a certain kind of animal. A great introduction to these nature conservation efforts is this 4-minute video posted on YouTube by the NGO Panthera: 

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/3FdoJEVnu4A/0.jpg)](https://www.youtube.com/watch?v=3FdoJEVnu4A&t=12s&ab_channel=Panthera)

*The terms and conditions of the DrivenData.org challenge do not allow us to share the image files in this repository. If you want to look at the images yourself and run our code, you need to sign up for the challenge (for free) and download the files to the data-folder.*

## Our work

### Outline of the project

**The challenge was to build and train a neural network, which can accurately detect the correct animal in photos taken by photo traps in Tai National Park, Ivory Coast.** The data contained a total of **16,500 images** and was provided **with labels** by DrivenData.org. Each image only contained one or no animal.

The labels were:
1) Antelopes / Duikers
2) Birds
3) Blank
4) Civets / Genets
5) Hogs
6) Leopards
7) Monkeys / Prosimians
8) Rodents

### Exploring the data

We started out with exploring the data. This contained getting an overview of the label distribution, and information of the images, which included sites where the photos were taken, as well as resolution and color information of the images. You can find a Jupyter Notebook called **EDA.ipynb** we created for exploring the data in the main directory of the repository.

### Building and training a model

Our approach to building and training the best model was to **test several pre-built model architectures, which are commonly used for image classification tasks**, such as ResNet50, VGG16, or InceptionNetV3. In some cases, we used these networks as a **base model and added our own layers on top**. Additionally, there were **several preprocessing steps the images had to go through before training the model**, such as removing time stamps and logos from the images, and additional typical processing for image classification, such as resizing, controling the color channels and data augmentation. We also worked on **improving computing performance** by making use of our GPUs with **cuda**.

To write all of these functions in Python, we used three approaches:

1) Approach *(final version)*: Using **PyTorch** and a separate **self-written-functions file** to create an **all-in-one Jupyter notebook**, which we have used for laoding the data, preprocessing the images, training a model, predicting labels and evaluating the results.
2) Approach *(intermediate saving)*: Based on **PyTorch** as well, but we **separated** our costum preprocessing of the images from training and evaluating the model. The preprocessed images were saved in between. This saved us some computing time, because we did not need to preprocess the images every time we trained and evaluated a model.
3) Approach *(tensorflow)*: Since we were trained on TensorFlow during the neuefische bootcamp, we wanted to test how **Tensorflow-pipelines** work. We managed to implement pipelines for training a model, but continued in PyTorch from then on to get more experience in using that library. Hence, this solution is not complete.

The scripts and notebooks we have created can be found in the respective folders of the approaches.

## Setup

**Requirements:**

- pyenv with Python: 3.11.3

Since we were working with different operating systems, we created two separate requirement.txt files, which contain all necessary Python packages for install.

### MacOS

```BASH
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev_macos.txt
```

### Linux

```BASH
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev_windows.txt
```

### Windows

For `PowerShell` CLI :

```PowerShell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_dev_windows.txt
```

For `Git-Bash` CLI :

```BASH
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements_dev_windows.txt
```

## License
MIT License