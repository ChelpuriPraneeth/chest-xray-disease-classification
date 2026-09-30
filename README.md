# AI-Based Chest X-ray Disease Classification Using Deep Learning


## Overview


This project is a deep learning-based system for classifying chest X-ray images into two classes:


- NORMAL

- PNEUMONIA


The project compares multiple CNN-based approaches and uses MobileNetV3Small with transfer learning.


## Project Workflow


```text

Chest X-ray

     ↓

Image Preprocessing

     ↓

Deep Learning Model

     ↓

Prediction

     ↓

NORMAL / PNEUMONIA

```


## Models Evaluated


1. Basic CNN

2. Class-Weighted CNN

3. Improved CNN with Data Augmentation and Dropout

4. MobileNetV3Small Transfer Learning


## Results


 Model                Test Accuracy 
-------------------------------------

Basic CNN                    70.83% 

Class-Weighted CNN           74.52% 

Improved CNN                 82.37%

MobileNetV3Small             87.50% 


## Final Model Metrics


Class    Precision    Recall    F1-score 

--------------------------------------------

NORMAL        0.93      0.72        0.81 

PNEUMONIA     0.85      0.97        0.91 


**Test Accuracy: 87.50%**


**Macro F1-score: 0.86**


## Technologies Used


- Python

- TensorFlow

- Keras

- CNN

- MobileNetV3Small

- Transfer Learning

- Flask

- HTML

- CSS

- NumPy

- Pandas

- Scikit-learn

- Matplotlib

- Jupyter Notebook


## Web Application


The project includes a Flask-based web interface where a user can upload a chest X-ray image and receive a model prediction.


## Dataset


The project uses the Chest X-Ray Images (Pneumonia) dataset.


The dataset is not included in this repository because of its large size.


## Project Structure


```text

chest-xray-disease-classification/

│

├── app.py

├── Minor_Project.ipynb

├── chest_xray_mobilenetv3.keras

├── model_comparison.png

├── requirements.txt

├── .gitignore

│

├── templates/

│   └── index.html

│

└── static/

    └── style.css

```


## Limitations


- The dataset contains class imbalance.

- Performance may vary on images from different datasets or clinical environments.

- The model should not be considered a replacement for professional medical diagnosis.

- Further validation on diverse external datasets is required.


## Disclaimer


This project is developed for educational and research purposes. It is not a medical diagnostic system and should not be used to make clinical decisions.


## Future Scope


- Improve model generalization using larger and more diverse datasets.

- Add explainable AI such as Grad-CAM.

- Improve the web interface.

- Add support for additional chest diseases.

- Deploy the application using a cloud platform.