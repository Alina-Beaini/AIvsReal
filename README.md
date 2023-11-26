# Detecting Images Generated by Neural Networks
[Erdős Institute](https://www.erdosinstitute.org/) [Data Science Boot Camp](https://www.erdosinstitute.org/programs/fall-2023/data-science-boot-camp), Fall 2023.

[//]:(We_need_to_add_our_presentation_link_here.)

## Team Members:
- [Alina Al Beani](www.linkedin.com/in/alina-beaini)
- [Amanda Pan](https://github.com/appandaman)
- [Berrin Senturk](https://www.linkedin.com/in/berrin-senturk-1b8a8a117/)
- [Cemile Kurkoglu](https://www.linkedin.com/in/cemile-kurkoglu)
- [Hasan Saad](https://github.com/HasanSaad2)

# Project Description

The recent advances in deep learning, neural networks, and the advent of hardware to support it has provided fertile ground for creating fake images.  This new technology, if left unchallenged, creates the risk of undermining journalism, and therefore, democracy. In the words of the KPMG chair in organizational trust at the University of Queensland , Nicole Gillespie:

>“How do we know what’s true or not true any more? That’s what’s at stake.”</br>

We tackle this problem by constructing two multi-classification models (single-channel and dual-channel) to discern between images which are real, that is, not generated by AI, and those which are generated by AI, and to determine which generative algorithm was used.  Our model is trained on a publicly available dataset  [[2]](https://github.com/PeterWang512/CNNDetection)  of ≈90000 images, which contains 13 CNN-based generative algorithms.</br>

# Precision and Recall

Our precision and recall on validation sets for the dual-channel model are shown below:

![image](https://raw.githubusercontent.com/Alina-Beaini/AIvsReal/main/Presentation_Images/Precision_Validation.png)

![image](https://raw.githubusercontent.com/Alina-Beaini/AIvsReal/main/Presentation_Images/Recall_Validation.png)
# Model Details
Our models are a single-channel and dual-channel Convolutional Neural Networks. 

## Model Architecture

### Single-Channel Convolutional Neural Network

Here the architecture of the single-channel CNN is outlined.

![image](https://raw.githubusercontent.com/Alina-Beaini/AIvsReal/main/Presentation_Images/single_channel_model.png)
- Activation Function for Hidden Layers: ReLU
- Activation Function for Output Layer: Softmax
- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy
- Filter Applied to Input: High Pass Filter Using Gaussian Blur (see below)

### Dual-Channel Convolutional Neural Network

Here the architecture of the dual-channel CNN is outlined.

![image](https://raw.githubusercontent.com/Alina-Beaini/AIvsReal/main/Presentation_Images/dual_channel_model.png)

- Activation Function for Hidden Layers: ReLU
- Activation Function for Output Layer: Softmax
- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy
- Filter Applied to Input: One copy of the image has a High Pass Filter while the other copy has a Log-Scale and Normalized Discrete Cosine Transform

## Filters

Three filters were used in this project. The code for the filters is found in
[`Filters.py`](??)

[//]:(We_need_to_add_link_to_filters.)

## Miscellany 

### To process our images:

We use a python script which discards images below a certain dimension as well as grayscale images. Furthermore, it saves the files into a new folder which contains the newly processed images, with filenames that include the necessary output material.

[`Preprocessor.py`](https://raw.githubusercontent.com/Alina-Beaini/AIvsReal/main/Standalone_Modules/Preprocessor.py)

### Loading our images

Since loading all the images into memory before training the neural network is not feasible, we create a custom Sequence element which reads the paths, and puts the images in memory on a "need-to-know" basis. In other words, on any given batch, only that batch is loaded into memory.

[`KerasCustomSequence.py`](https://raw.githubusercontent.com/Alina-Beaini/AIvsReal/main/Standalone_Modules/KerasCustomSequence.py)

### Multiclass Precision and Recall Metrics

Since Keras does not natively support precision and recall metrics when more classes than two are involved, we write our own metrics.

[`KeywordsMetrics.py`](https://raw.githubusercontent.com/Alina-Beaini/AIvsReal/main/Standalone_Modules/KeywordMetrics.py)

# Model Interface

We've written a [web app](??) that showcases our model. 

Here is an example run.

![image](...)
