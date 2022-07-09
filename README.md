# Multicharacter Alphanumeric Handwriting Detection using Tensorflow

Handwritten text recognition model built using tensorflow and the EMNIST dataset that can be found [here](https://www.nist.gov/itl/products-and-services/emnist-dataset). The EMNIST dataset contains 62 classes containing the digits 0-9 and A-Z characters in both upper and lowercase. A model to detect handwriting was generated from the 'byclass' type of this dataset, which contains 814,255 images, using a Convolutional Neural Network. User input is exported from a tkinter canvas where it is then split into characters from left to right and analyzed by our model.

<a href="https://github.com/mpeng72/handwritingdetection">
    <img src="/example.png" alt="Logo" width="400" height="400">
  </a>

This project also contains models generated from the MNIST digit dataset and an A-Z uppercase letter dataset which can be found [here](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)

## Tensorflow

[Tensorflow](https://www.tensorflow.org/api_docs/python/tf) is an open-source library that supports the development of neural networks and large-scale machine learning

## Installation
Use the file requirements.txt to install the necessary packages

```bash
pip install -r requirements.txt
```

## Usage
To run the code, type 

```python
python main.py
```

