# Classifying Color Images

This repo uses the MNIST database to train a convolution neural network in Tensorflow to determine if a specific image is of a dog or a cat. It uses images of varying sizes found in [this dataset](https://www.kaggle.com/c/dogs-vs-cats/data) provided by Kaggle

# Recommended usage


Clone the repo, setup a Python 3 virtual environment and install the required packages found in `requirements.txt` in the virtual environment.

```
git clone https://github.com/NormanBenbrahim/predict-clothing.git
cd predict-clothing
python3 -m venv env_python
source env_python/bin/activate
pip install -r requirements.txt
```

Then run `python cat_or_dog.py`