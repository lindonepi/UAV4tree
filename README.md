# uav4tree
Deep Learning based tree species classification from UAV images

script enables the training of a Vision Transformer B16 neural network for image recognition from patches of size 256 x 256 pixels. 

In our case, the dataset consists of images of 9 species of trees. The images are made with a drone (AUV). Each image before being inserted into the dataset was divided into images of size 256 x256 . This avoids losing details compared to other solutions in which the image is resized before training.

you can edit the script to change the number of training epochs, hyper parameters, or dataset directories 
