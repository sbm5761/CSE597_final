# CSE597_Final: DeeCap

# Data
To run the code, annotations and images for the COCO dataset are needed. Please download the zip files including the images ([train2014.zip](http://images.cocodataset.org/zips/train2014.zip), [val2014.zip](http://images.cocodataset.org/zips/val2014.zip)), the zip file containing the annotations [annotations_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) and extract them. Then, to create a training set and testing set of your desired length, you will need to run dataset.py to create .pkl files that will be used to extract the data during training. Within dataset.py, you will need to change the paths to the annotations and images accordingly as well as where you would like the final .pkl file stored. 

# Training
After you have your train.pkl and test.pkl files generated, you will run python train_deecap.py to begin the training process. You will need to change the paths accordingly to where the .pkl dataset files are saved. Feel free to experiment with parameters such as epochs, learning rate, batch size, etc. Once training is done, the model's state dict will be saved to a .pth file in which you can easily load for later use in evaluation.

# Testing and Evaluation
Once you trained your model, you can run python test.py to evaluate its performance. Again, you will have to change the paths to direct it to the testing dataset and the saved model. You can also go into ./evaluation/__init__.py and change the metrics to evaluate the model on (the options available are given). 

Acknowledgement to: https://github.com/feizc/DeeCap/tree/main 
