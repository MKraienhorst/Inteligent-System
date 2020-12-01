README for mainDeepLearning.py

python script to classify the ISIC images with deep learning.

Uses:
torch
torchvision
PIL
sklearn
(configparser) -> not yet

Parameters to declare atm (at the moment):
_lr: 		learning rate of the optimizer
_numModel: 	which cnn model is used (only densenet working untill now)
_bin:		set _bin to 0 to run classification of 7 classes, any other value: run binary classification
_onlyEval:  set to anything else than 0 -> no model is not trained -> old model is loaded and predicts the validation-set once + saves the prediction

Used CNN Model:
Atm only a pretrained Densenet Model is used. Tried to use VGG or Resenet they have errors in the last layer, eventhough pytorch said they need 244x244 images.
To load the data into the model, the images are loaded with the ISICDataset class and preprocessed.
Preprocessing includes:
1. scaling the RGB images from 600x450 to 256 x 256
2. cropping the image to 244 x 244
3. some other augumentation-> depending on train or learn
4. transform to a tensor.
5. normalize the tensor.

These CNN models are trained on the "imagenet" to predict 1000 classes.
The last layer of all moduls have to be replaced by a layer, which downsizes from XXX (depends on model) Neurons to only 7 or 2 classes (depends on binary or class classification).
The last layer of the moduls have different names, some are called "model.classifier" or "model.fc" -> printing the model might be helpful to see what is the last layer, or look in online code.
