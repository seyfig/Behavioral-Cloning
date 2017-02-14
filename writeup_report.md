#**Behavioral Cloning**

**Behavioral Cloning Project**

The goals/steps of this project are the following:
* Use the simulator to collect data on good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### The [rubric points](https://review.udacity.com/#!/rubrics/432/view) are considered individually and described how they were addressed each point in the implementation.

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model architecture is based on NVIDIA architecture. In order to start with 106 x 320 image, the first convolution layer is implemented with a larger filter (11x11). The model has 5 convolution layers each followed by a RELU activation. The convolution layers are listed as follows:
* 11 x 11 filter, with 24 depth, 3 x 3 stride
* 5 x 5 filter, with 36 depth, 2 x 2 stride
* 5 x 5 filter, with 48 depth, 2 x 2 stride
* 3 x 3 filter, with 64 depth, 1 x 1 stride
* 3 x 3 filter, with 64 depth, 1 x 1 stride

The model has 3 fully connected layers with sizes 100, 50, 10, and 1 output layer of size 1.

This model was selected according to driving performance and training time. In addition to this model, networks with more convolution layers and networks with fewer convolution layers experimented. The comma.ai model was also included in the study. Since the lowest mse loss function does not guarantee the best driving model, the results are compared on the simulator.

####2. Attempts to reduce overfitting in the model

The model contains 6 dropout layers with 0.25 dropout rate in order to reduce overfitting. The are located after the convolution layers 2, 3, 4 and after flatten layer and after fully connected layers 1, 2.

The data split into training data set and validation data set with ratios 80% and 20%.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line TODO REF).

####4. Appropriate training data

In addition to the recorded data, artificial data is added by flipping images. Only the images having an absolute value of steering angle greater than rthr parameter are flipped and added to training data. For the final model, rthr is set to 0.0. Therefore, the images with zero steering angle value are added only once, and the others are added twice with one of them flipped. The training data then split into training data and validation data.

The final training data consists of four parts. Only the track 1 data is used to train the model, and it contains both clockwise and counter-clockwise driving.

The first part is the provided Udacity data. It contains 8036 images. It mainly provides center lane driving. 3675 of these images are flipped and added again. The number of samples is 11711.

The second part of the data is recorded from autonomous driving. During the study, when a model performed a successful drive it was recorded to use in later training. There four different recording created with this method. The first two recordings are successful autonomous drives of different models on track one clockwise direction. Only the Udacity data is used during the training of these models. They contain 2228 and 2057 images. The last two recordings are successful turns on the different curves of the first track in the counter-clockwise direction. They were recorded with different models. The data used in the training of these models was consist of the Udacity data and the previously recorded second part data. They have 449 and 455 images. All of the 5189 images in this part are flipped and added again, resulting in 10378 samples.

The third part of the data is recovery data and it was recorded when driving in training mode. The strategy was to drive in a curve with full speed without recording; wait until the end of the road, and only record the last minute turn. There were 327 images in this part. After adding the flipped images it has 620 samples.

The fourth part of the data is extreme recovery data. The car was set in a position that it was about to get out of the road, and it was stopped. Then the steering angle was set to 25.0 or -25.0 degrees to get the car on the road again. The images were recorded with zero or very low speeds. There were 553 images in this part. All of them flipped and added again and the total number of samples in this part is 1106.

The total number of images in the final model is 14105 with flipped images there are 23815 samples.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a convolutional neural network to predict the steering angles from camera images to keep the car on the drivable road.

My first step was to use a convolution neural network model similar to the comma.ai. It didn't work well, so I skipped to the NVIDIA model. In order to use NVIDIA model, images are cropped. Removing first 55 and last 39 rows and the first 60 and last 60 columns image were cropped to have a resolution 66 x 200. I only used Udacity data with flipping all of the images, there were 16072 samples in total. Using 90% of them in training and 10% of them in validation. Used a model similar to the final model, the only difference is the first convolutional layer has a 5 x 5 filter with 24 depth and 2 x 2 strides instead of 11 x 11 filter with 3 x 3 strides. Since the lowest mse value on validation set does not guarantee the best driving, I recorded the weights at each epoch. Test these weights on the simulator. Start to test with the epoch that has the smallest mse value on the validation set. However, when the epoch number is greater than 20, usually the set of weights perform worse in spite of lower mse values. In addition, almost always the best performing weights and the weights that have the lowest mse value are not the same. The recorded weights at epochs 9 and 11 were successful to drive autonomously on the first track. However, they were not able to drive several laps.

As the second step, I changed the cropping size. By removing the first 54 rows images were cropped to 106 x 320 resolution. Then changed the model to the final model. I have tried other models again such as comma.ai model (model.py lines 285:316), models with more convolutional layers, and models with less convolutional layers (model.py lines 237:282). I have also tried adding and removing dropout layers and changing their dropout rates. However, the best model I can come up with is the final model. With the final model and changing the cropping size, I have changed the flipping strategy. Images that has an absolute of steering angle that is greater than 1.0 (in degrees) are flipped and added again. The remaining only added as they are. I also changed the validation size to 20%. In this step, 3 successful sets of weights are saved at epochs 9, 16, 17. Only the set at 17 epoch can drive continuously.

In the third step, I added the recorded autonomous drive data, the second part of the data as mentioned in the fourth part of the previous section. There were 5 different successful weights that can complete at least one lap on the first track. I recorded one lap with each of them and added them to the training data. Changed the rthr value to 0.0, which would add all of the images twice except the ones that have a zero steering angle value. It was required to switch to generators instead of loading all of the data to memory. In the previous steps, I didn't use generators. The saved weights for several epochs were able to drive the car continuously. Then I tested them to drive counter-clockwise on the first track. They were not successful to complete the track. However the weights saved at epoch 2 and the weights saved at epoch 5 failed on only one curve, and the curves are different. Therefore, I recorded their successful turns on different curves and use this data in later training processes as second part of the data.

Then I have tried to build a model that is able to drive on both tracks. I have tried different models, added recovery data, and extreme recovery data. The final model used has a batch size of 32, the models in the previous steps used 64 batch size.

With the final model, I was able to save weights that are able to drive on both tracks in both clockwise and counter-clockwise directions. The weights are saved in the 12th epoch. The performance on the first track is a little lower than the previous successful models, but the performance on the second track is impressive. One probable reason for that is the extreme recovery data. Some of the samples possibly cause waving, removing them can increase the performance on the first track.

####2. Final Model Architecture

The final model (model.py lines 187:234) has a cropping layer and a normalization layer followed by convolutional layers and dropout layers:
* 11 x 11 filter, with 24 depth, 3 x 3 stride
* 5 x 5 filter, with 36 depth, 2 x 2 stride
    * Dropout 0.25
* 5 x 5 filter, with 48 depth, 2 x 2 stride
    * Dropout 0.25
* 3 x 3 filter, with 64 depth, 1 x 1 stride
    * Dropout 0.25
* 3 x 3 filter, with 64 depth, 1 x 1 stride

Each convolution layer has relu activation.
The model has 3 fully connected layers with sizes 100, 50, 10, and 1 output layer of size 1. Before each of the fully connected layers, there are dropout layers with 0.25 dropout rate in order to reduce overfitting.

Adam optimizer and mse loss function are used to compile the model.
The number of batch size is 32.
The number of epochs is 50, however, at every epoch weights are saved.

####3. Creation of the Training Set & Training Process

I started with the Udacity data. Two examples of center lane driving:

![Udacity Plot](https://github.com/seyfig/BehavioralCloning/blob/master/img/plot1.jpg)


![Udacity 1](https://github.com/seyfig/BehavioralCloning/blob/master/img/center_2016_12_01_13_32_54_976.jpg)


![Udacity 2](https://github.com/seyfig/BehavioralCloning/blob/master/img/center_2016_12_01_13_33_43_391.jpg)

I then recorded the vehicle when it is driving in autonomous mode.

![Autonomous Plot](https://github.com/seyfig/BehavioralCloning/blob/master/img/plot2.jpg)

![Autonomous 1](https://github.com/seyfig/BehavioralCloning/blob/master/img/center_2017_02_13_01_15_17_771.jpg)

![Autonomous 2](https://github.com/seyfig/BehavioralCloning/blob/master/img/center_2017_02_13_01_15_18_143.jpg)


Then I recorded recovery data.

![Recovery Plot](https://github.com/seyfig/BehavioralCloning/blob/master/img/plot3.jpg)

![Recovery Data](https://github.com/seyfig/BehavioralCloning/blob/master/img/center_2017_02_13_15_32_00_921.jpg)

![Recovery Data](https://github.com/seyfig/BehavioralCloning/blob/master/img/center_2017_02_13_15_32_10_175.jpg)

Then I repeated this process on last minute recoveries.

![Extreme Recovery Plot](https://github.com/seyfig/BehavioralCloning/blob/master/img/plot4.jpg)

![Extreme Recovery Data](https://github.com/seyfig/BehavioralCloning/blob/master/img/center_2017_02_13_19_25_46_253.jpg)

![Extreme Recovery Data](https://github.com/seyfig/BehavioralCloning/blob/master/img/center_2017_02_13_19_27_14_412.jpg)

After the collection process, I had 23815 number of data points. I then preprocessed this data by cropping the first 54 rows images to have 106 x 320 image resolution. Then normalized by dividing by 127.5 and subtracting 1. I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or underfitting. The ideal number of epochs was 12 as evidenced by the test on the simulator. I used an adam optimizer so that manually training the learning rate wasn't necessary.


## Results
The result drivings are listed here


First Track Clockwise

[![First Track Clockwise](http://img.youtube.com/vi/z3jbYoWCE9M/0.jpg)](http://www.youtube.com/watch?v=z3jbYoWCE9M)


First Track Counter-Clockwise

[![First Track Counter-Clockwise](http://img.youtube.com/vi/ydXteT1jcYA/0.jpg)](http://www.youtube.com/watch?v=ydXteT1jcYA)


Second Track Clockwise

[![Second Track Clockwise](http://img.youtube.com/vi/cSlGYXtiG3U/0.jpg)](http://www.youtube.com/watch?v=cSlGYXtiG3U)


Second Track Counter-Clockwise

[![Second Track Counter-Clockwise](http://img.youtube.com/vi/ECsnXxHFP58/0.jpg)](http://www.youtube.com/watch?v=ECsnXxHFP58)
