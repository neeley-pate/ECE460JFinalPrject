# Classifying Images: Artwork, Real World Images, Digital Renderings

Authors: Boting Lu, Chloe Tang, Dario Jimenez, Neeley Pate, Yuri Rykhlo, Joon Song

## Goal
Images originate from many mediums, whether it be from a portrait painting or a picture taken of a beach. Our goal with the Spring 2023 Data Science Laboratory Project is to determine if a model can discern between 1) works of art from different time periods, 2) images taken with a camera in the “real world”, and 3) digitally generated images using stable diffusion. 

## Significance
Having models that can recognize the difference between works of art, images taken with a camera, and digitally created images has the potential to aid in the effort to detect deep fakes. As deep fakes become more sophisticated, it is important to have accurate detection mechanisms to help debunk them.

## Previous Work
Ahmad and Khursheed (May 2021) published an in-depth introduction and review of existing works in digital image manipulation. Some of the methods included applying PCA to the data and convolutional neural networks.
[Link to Paper](https://link.springer.com/chapter/10.1007/978-981-33-4604-8_70#Sec9)

Lee et. al. (June 2021) discusses that while there are many uses for real image detection, often “unusual images” such as pieces of art go unlearned and act as a limitation for some image processing models. Their study finds that it is possible to detect real world features from pieces of art, such as hands and shoulders from children’s drawings.
[Link to Paper](https://link.springer.com/chapter/10.1007/978-3-030-79474-3_7)

## Data Collection
Appropriate datasets were found for each of the classes, mostly from Papers with Code and various Kaggle competitions. For the Artwork class, datasets contained different types of paintings and collectively consisted of 29 different genres, ranging from impression and portraits, to realism and romanticism. For the Real-World class, the data ranged from images consisting of faces, food, landscapes and bodies of water, and other miscellaneous ones taken by a camera. For the Digital Renderings class, the images were created using stable diffusion, a deep-learning text-to-image model.

## Data Visualization
TODO

## Pre-Processing
For all models, 1,000 images were used from each class, or 3,000 total for train and test. Despite having 15,000 total images, this proved to take too much processing power to use all images on a given model, so the images were sampled from this dataset. The images were 128 x 128 pixels, and were RGB encoded.

## Implementation
### Custom Made Models
#### Convolutional Neural Network
For the CNN, the Keras library was used to build a custom model. The model saved 25% for testing. The model then has two convolutional layers followed by a ReLU activation. Finally, the results from those layers are flattened and condensed into 3 outputs that are run through a softmax. The loss measure was evaluated using cross entropy, and early stopping was used to prevent overfitting.

#### Logistic Regression
The Logistic Regression model also uses the Keras library. This model also saved 25% of the images for testing. The logistic regression model is very similar to the CNN without the convolutional layers. It takes the images, condenses them, then runs it through a layer that gives 3 outputs that are run through a softmax. The loss for this model

#### Multi Layer Perceptron
TODO

### Transfer Learning
#### VGG-16
VGG-16 is a popular pre-trained image classification model from the Keras library. The pre-trained model is a large CNN with 16 weight layers and approximately 138 million parameters. VGG-16 needed to be fine tuned for classification of 128x128 and 64x64 images using the Keras library. To fine tune the model for our task, an additional dense layer with ReLU activation was stacked on top of the output. Another dropout layer was included before running a softmax in order to account for overfitting. Overall, the 128x128 model performed better on the testing set.

#### ResNet50
ResNet50 is a variant of the ResNet model, which is a type of Artificial Neural Network (ANN) that stacks residual blocks, and is used for image recognition tasks. It is a 50 convolutional neural network (CNN) layers, composed of 48 convolutional layers, 1 MaxPool layer, and 1 average pool layer. The pre-trained model contains over 23 trainable parameters, and displays higher performance by creating shorter connections through residual mapping (as opposed to direct mapping). ResNet50 specifically implements a Bottleneck design, in which it contains 1x1 convolution layers in the residual blocks that allows for faster training.  

![alt text](https://www.researchgate.net/publication/349717475/figure/fig4/AS:996933933993986@1614698980245/The-architecture-of-ResNet-50-model.ppm)

#### Inceptionv3
TODO

#### EfficientNet
EfficientNet is one of the newest image classification models from Google. It uses a scaling method called Compound scaling, which is essentially scaling the dimensions of parameters by a fixed number at the same time and uniformly. This model uses this scaling technique and multiple convolutional layers to achieve a very high performance with only 5.3 million parameters. We used the output of this model as input to a fully connected layer of 1024 neurons with ReLu activation. We then passed the output of this layer to a dropout layer to decrease overfitting, then finally to a softmax layer.

## Results
| Model         | Accuracy          |
| ------------ | ------------- |
| Keras CNN     | 53.2%     |
| Logistic Regression   | 53.7%      |
| MLP | 33.6 - 37.8% |
| VGG 16     | 89.9%     |
| ResNet50     | 90%     |
| Inceptionv3   | 47.1%      |
| EfficientNet | 93%|

## Next Steps
Since we were limited on the amount of data we used to train our models, our models did not perform as well on types of images not used in training as it did on our validation set. For instance, all the computer generated images we used for training and validation were generated from stable diffusion, which means that our model would be very good at detecting stable diffusing images but not as good at identifying images rendered by other means, such as game engines, simulations, cgi, etc. Same goes for paintings and real-world images whose style and type were not represented in our training dataset. Our next steps here is to attempt to capture as many types of images from each class as possible, in hopes to increase our model’s performance on all types of images. For computer-generated images, we want to capture more images from video-games, computer-generated imagery (cgi), simulation renderings, computer-generated cartoons, photo-realistic renderings, and non-photorealistic renderings, graphical user interfaces,  and any other types of computer-generated images. For paintings, we want to capture more styles of painting and paintings using different types of paint. For real-world images, we want to capture images of everything there is such as landscapes, oceans, animals, food, cars, buildings, etc.
