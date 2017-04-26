#**Behavioral Cloning** 
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Lane Driving"
[image2]: ./examples/recovery1.jpg "Recovery Image"
[image3]: ./examples/recovery2.jpg "Recovery Image"
[image4]: ./examples/placeholder_small.jpg "Recovery Image"
[image5]: ./examples/cropped.jpg "Cropped Image"
[image6]: ./examples/original.png "Original Camera Image"
[image7]: ./examples/flipped.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* utils.py contains code for utility functions used in the main file model.py

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I started with the basic structure used by David in the course with just one fully connected layer for initial understanding. Then I used the original nvidia model architecture described [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
Then I modified the layers,filter sizes and added maxpooling to improve the model validation accuracy.

My final model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 128 (model.py: function nvidia_model()). And then fully connected layers with final output for vehicle control, steering wheel angle.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 16). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 38). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (cell 5 in the model.ipynb). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py cell 2, line 49).

####4. Appropriate training data

Given training data along with collected simulator data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road . Also the left and right camera images were used to balance the input training data.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was for the car to be able to stay mostly in the center of the lane, finish the lap and recover at turns in the first(track 1) driving scenario.

My first step was to use the model discussed in the lecture, to get started. Then I used a convolution neural network model similar to the Nvidia end-to-end learning for self-driving car paper. I thought this model might be appropriate because it is sufficiently complex and was used for similar task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add dropout layers with convolution layers to have drop probablity of 0.2 and fully connected layers to have drop probablity of 0.52. After tuning, I realized the convolution layers did not need dropout layers, so removed them from my final model.

Then I tweaked the model by making deeper layers, because the validation loss was not dropping significantly. Making the layers deeper and adding MaxPooling helped reduce the validation loss.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I used the following steps to improve input training data. Because more than the model architecture, the training data and validation data was affecting validation loss behaviour.

At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes :


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Cropping2D Layer     	| To crop top 60 and bottom 25 rows of pixels to remove unwanted scene details	|
| Normalization Layer  	| Lambda layer to normalize all pixel values from -0.5 to 0.5	|										
| Convolution 3x3  | Conv Layer 1 (24, 3x3 filters with RELU activation, 2x2 maxpooling ) |		
| Convolution 3x3  | Conv Layer 2 (48, 3x3 filters with RELU activation, 2x2 maxpooling) |
| Convolution 3x3  | Conv Layer 3 (64, 3x3 filters with RELU activation, 2x2 maxpooling ) |
| Convolution 3x3  | Conv Layer 4 (128,3x3 filters with RELU activation, 2x2 maxpooling ) |
| Fully connected		| Output = 512, RELU activation and dropout probability of 0.52 |
| Fully connected		| Output = 256, RELU activation and dropout probability of 0.52 |
| Fully connected		| Output = 1, no activation. This final node's value is the steering angle measurement |
|						|												|


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

Then I did another lap with recovery data so that the model can tune to receover itself if it is about to hit lane boundaries or curbs.
![alt text][image2]

And trying to recover:
![alt text][image3]

STEPS:
Preprocessing:
 1. Cropping Image : Original image was cropped to remove the top sky part and the bottom car hood.
 
 ![alt text][image5]
 
 2. Data normalization: Used the normalization formula to reduce the range of the input data between -1 to 1.
 3.Reducing low steering wheel data: I was randomly dropping low steering wheel data between abs(0.05) to unbias the data for center lane driving.
 
Data Augmentation: 
 1. Using left and right camera images: The input data was augmented with left and right camera images, the left camera image ahs to move right to be at the center and the right camera image has to move  to left to be at the center, that is taken care by the correction factor of 0.18
 
 ![alt text][image6]
 
 2. Flip images: The image was flipped horizontally like a mirror and taking opposite sign of steering measurement.
 
```sh
augmented_images.append(cv2.flip(image,1))
augmented_measurements.append(measurement*-1)
```  

![alt text][image7]

After the collection process, I had 11836 number of data points. I then preprocessed this data as described above and normalized the data in the architecture itself.

I finally randomly shuffled the data set and put 0.2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I shuffled the data to begin. I used adam optimizer along with mean squared loss for training the model so that manually training the learning rate wasn't necessary. 

The ideal number of epochs was 25 as evidenced by model and the video.
