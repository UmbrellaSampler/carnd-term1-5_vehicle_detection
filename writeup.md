## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Car-NotCar.jpg
[image2]: ./output_images/HogFeatures_Car_YUV.jpg
[image3]: ./output_images/HogFeatures_NotCar_YUV.jpg
[image4]: ./output_images/ColorHist_Car_YUV.jpg
[image5]: ./output_images/ColorHist_NotCar_YUV.jpg
[image6]: ./output_images/Channels_Car_YUV.jpg
[image7]: ./output_images/Channels_NotCar_YUV.jpg
[image8]: ./output_images/WindowImage.jpg
[image9]: ./output_images/WindowImage_test1.jpg
[image10]: ./output_images/WindowImage_test3.jpg
[image11]: ./output_images/WindowImage_test4.jpg
[image12]: ./output_images/WindowImage_test5.jpg
[image13]: ./output_images/carpos-heatmap.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Code
All code is contained in the Jupyter Notebook `vehicle_detection.ipynb`.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook. Here I define all relevant function to extract HOG features (`get_hog_features()`, line 11) or features from binned colors (`bin_spatial()`, line 35) and color histograms (`color_hist()`, line 43).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

##### HOG Car
![alt text][image2]

##### HOG Not Car
![alt text][image3]

I also experimented with color histograms: 

##### Color Histograms Car
![alt text][image4]

##### Color Histograms Not Car
![alt text][image5]


These are the individual Channels:

##### Channels Histograms Car
![alt text][image6]

##### Channels Histograms Not Car
![alt text][image7]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried many various combinations of parameters... many ... See cells in section "Feature extraction" in the code.
First off all, for allmost all combinations, the training of the classifier did not converge. So my goal was to make the training converge.

Unfortunately, only one combination of parameters made it. Here is the set of paramters I used for feature extraction. 

```python
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 15  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In section "Training a linear SVC" in the Jupyter notebook I trained a linear SVM using a stack of extracted car features and non car features and a prepared label data set. For feature extration I used color binning and histograms, and the HOG features, as already described above. I splitted the features in a training and test set using the function `sklearn.model_selection.train_test_split()` with also includes a random shuffling. Then I scaled the data using a `StandardScalar` in order to improve the training quality. The training was performed using a `LinearSVC` model. The model performed optimization on a feature vector of length 11988 with test accuracy of 99.1%. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code that implements the sliding window search is defined in the first cell of section "Sliding Window Search" in the Jupyter notebook. As first step all possible window candidates are determined in the function `slide_window()`. In my case I decided to perform two searches with different window dimension: `xy_windows = [(96, 96), (64, 64)]` (line 12-24, 2nd cell in section "Sliding Window Search" ) which significantly improved the car finding performance. In the function `search_windows()` all images found in these windows are predicted using the SVC classifier. Positivley classified (we found a car) windows are collected and returned. An example result can be seen in this image:

![alt text][image8]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some more example images:

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./vehicle_detection_submission.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video.

### Here are six frames and their corresponding heatmaps and the labels of `scipy.ndimage.measurements.label()`:

![alt text][image13]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One problem I had was to find the final parameter set for extracting image features. The main problem was that in most cases the training of the SVC model did not converge. So I had spend most of the time finding the right parameters.

Another problem was that I sliding window search output contained to many outliers and sometimes the cars were not recognized appropriately. To overcome this problem I introduced a second sliding window search with a different window size. This improved the quality significantly. However, this increased the required processing time for the video by a factor of two.

To futher improve the pipeline one could implemented a tracking technique to follow already found cars. One could then concentrate on detecting the cars in those area which could lead to less false positives.

