**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[train]:./output_images/train.png
[hog1]: ./output_images/hog1.png
[hog2]: ./output_images/hog2.png
[test1]: ./output_images/test1.png
[test6]: ./output_images/test6.png
[heat1]: ./output_images/heat1.png
[heat6]: ./output_images/heat6.png
[video1]: ./proccessed_project_video.mp4

[frame1]: ./output_images/frame1.png
[frame2]: ./output_images/frame2.png
[frame3]: ./output_images/frame3.png
[frame4]: ./output_images/frame4.png
[frame5]: ./output_images/frame5.png
[frame6]: ./output_images/frame6.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


My project includes the following files:
* **project.ipynb** containing the code for the project
* **project.html** the saved html version of the jupyter notebok
* **output_images** containing the output image files [ more examples in the saved notebook] 
* **writeup.md** the writeup
* **proccessed_project_video.mp4** the processed project video
* **model.p** the saved classifier model

---
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in Cell-5 of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Training images][train]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=8` and `cells_per_block=1 channnels=ALL`:

![Hog image 1][hog1]
![Hog image 2][hog2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and and then tried to see manually which ones produce more features and which did not by visualizing the hog features. I then used the feature set in the SVM training and tried to balance the features for computation time and accuracy

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a combination of the HOG histogram on `orientations=8`, `pixels_per_cell=8` and `cells_per_block=1 channnels=ALL` and combined it with the color_hist and bin_spatial with the params  `32 bins` and `spatial_size=(32,32)`
This was computed in cell 6 of the notebook and saved as a pickle dump to the file model.P also included in the submission
```
96.6 Seconds to extract HOG features...
Using: 8 orientations 8 pixels per cell and 1 cells per block
Feature vector length: 4704
12.48 Seconds to train SVC...
Test Accuracy of SVC =  0.9848
My SVC predicts:  [ 1.  1.  0.  1.  1.  0.  1.  0.  0.  0.]
For these 10 labels:  [ 1.  1.  0.  1.  1.  0.  1.  0.  0.  0.]
0.0 Seconds to predict 10 labels with SVC
Saved the new model!
```
###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to just try one scale in the sliding window of 1.5 and that seemed work well enough, and limited the search to ymin=400 and ymax= 656 based on the examples in the lesson.

![test image 1][test1]

![test image 2][test6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on just one scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, 
I used a heat map + thresholded the image ..which provided a nice result.  Here are some example images:


![Pipeline 1][heat1]
![Pipeline 2][heat6]
---

### Video Implementation

####1. Provide a link to your final video output.  
Here's a [link to my video result](./proccessed_project_video.mp4)

![My processed video][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video as seen in the images below:

### Here are six frames and their corresponding heatmaps:

![Frame 1][frame1]
![Frame 2][frame2]
![Frame 3][frame3]
![Frame 4][frame4]
![Frame 5][frame5]
![Frame 6][frame6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The compuation time of the pipeline was large  and still produced false positives. The pipeline also identified the cars on the road on the other side with traffic in the opposite direction.
Overall it doesnt seem to be something that can be used for real time detection... perhaps using a CNN might be something to explore if that is our goal

