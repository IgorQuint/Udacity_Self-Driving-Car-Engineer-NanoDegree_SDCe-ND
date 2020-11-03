# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on NVIDIA's autonomous vehicle architecture in [this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting (model.py lines 79 & 81). These layers randomly turn of inputs to neurons, this improves generalization en reduces the risk of overfitting to the data.
Furthermore, the training data is gathered by driving around the track both in the forward direction as well as reversed. This adds training data from a different perspective, helping to generalize on the driving data.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 129).

#### 4. Appropriate training data

I drove in the simulator myself to gather enough training data. I drove 2 laps in the forward direction, the attempt to recover from a bad position in the far left/right of the track. Then I proceeded to drive 2 full laps in reverse direction, to acquire
enough data from various perspectives.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try to incorporate the suggested CNN by NVIDIA[deep learning paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

My first step was to use the convolution neural network model similar from the paper. This might be appropriate as the use case is very similar.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I extended the model with two dropout layers with a keep_prob of 0.5.

Using image augmentation and reverse track driving, the training set was diverse enough to help generalize the model better. The MSE loss on the validation set dropped.

The final step was to run the simulator to see how well the car was driving around track one. This went pretty well, a couple improvement were made by improving the number of epochs. 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-83) consisted of a convolution neural network with the following layers and layer sizes:

* Lambda fucntion (normalize images)
* Cropping2D (50 px top, 20px bottom)
* Conv2D(24, (5, 5), strides=(2, 2), activation='relu')
* Conv2D(36, (5, 5), strides=(2, 2), activation='relu')
* Conv2D(48, (5, 5), strides=(2, 2), activation='relu')
* Conv2D(64, (3, 3), activation='relu')
* Conv2D(64, (3, 3), activation='relu')
* Flatten()
* Dense(100)
* Dropout(0.5)
* Dense(50)
* Dropout(0.5)
* Dense(10)
* Dense(1)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![good_driving](drivedata/IMG/center_2020_11_02_19_56_42_486.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to recover when too far left or right.
These images show what a recovery looks like:

![left_save](drivedata/IMG/left_2020_11_02_20_02_09_463.jpg)

![right_save](drivedata/IMG/center_2020_11_02_20_05_07_455.jpg)

To augment the data set, I also flipped images and angles to help the generalization of the trained model. For example, here is an image that has then been flipped:

![flipped_img](drivedata/IMG/center_2020_11_02_19_56_41_682.jpg)

After the collection process, I had 6398 number of data points. I then preprocessed this data by cropping 50 and 20 pixels respectively from the top and bottom of the image. Futhermore the images are normalized. 
Finally I randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 9 as evidenced by the very small training loss. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
