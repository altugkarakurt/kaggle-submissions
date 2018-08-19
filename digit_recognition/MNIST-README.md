## Description
This is the Hello World challenge for image classification where we are given the classic MNIST dataset of 60K 28x28 grayscale handwritten digit pictures and try to classify them.  

## Data & Preprocessing
Since I've been down this road before during my undergrad I knew the fancy tricks like chasing rotational invariance or bootstrapping do small improvements only. I only did scaling to map all pixel values in [0,1] instead of [0,255] and reshped them from 1D vectors to the original square images.

## Learning Model
Instead of using such feature engineering methods to improve a dumb learning methdos like fully connected MLP (which I did back in the day and learned a lot), I just followed Yann LeCun's classic convolutional neural network.