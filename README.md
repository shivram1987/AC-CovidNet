# AC-CovidNet
This repo contains the code used in AC-CovidNet paper.

## Steps to use:
1. Download data using covidx_download.ipynb
2. Preprocess using preprocessing.ipynb
3. Train the model using supcon.py
4. Use that trained model and test using main.py
(main.py can also be used to train and test covidnet)

# References
Code used from:

Supcon - https://github.com/keras-team/keras-io/blob/master/examples/vision/supervised-contrastive-learning.py
covinet keras implementation - https://github.com/busyyang/COVID-19
covidx data generation and preprocessing - https://github.com/lindawangg/COVID-Net
