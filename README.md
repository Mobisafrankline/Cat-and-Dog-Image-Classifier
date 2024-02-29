# Cat-and-Dog-Image-Classifier
This project aims to develop an image classification model to distinguish between images of cats and dogs using deep learning techniques in Python.


Features

Data preprocessing using TensorFlow's ImageDataGenerator.
Model architecture based on Convolutional Neural Networks (CNNs).
Data augmentation to improve model generalization.
Model evaluation and performance metrics computation.

Dataset
The dataset used for training and testing the model consists of images containing both cats and dogs. It can be found here.

Requirements
Python 3.x
TensorFlow
NumPy
Matplotlib
Installation

Clone this repository:
bash
Copy code
git clone https://github.com/your_username/cat_dog_classifier.git

Navigate to the project directory:
bash
Copy code
cd cat_dog_classifier
Install the required dependencies:
Copy code
pip install -r requirements.txt
Usage
Prepare your dataset by organizing cat and dog images into separate directories.
Update the paths to your training and testing data directories in the code.
Run the training script:
Copy code
python train.py
Once training is complete, evaluate the model's performance:
Copy code
python evaluate.py
To make predictions on new images, use the provided inference script:
Copy code
python predict.py path_to_image
Model
The trained model is saved as cat_dog_classifier_model.h5 in the project directory.

Credits
The dataset used in this project is sourced from https://www.kaggle.com/datasets/eduardofv/cat-or-dog-petfinder-pawpularity-competition.
Inspiration and guidance from various online tutorials and resources.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any inquiries or feedback, please contact mobisafrankline@gmail.com
