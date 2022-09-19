FastText Classifier for Text Classification - Base problem category as per Ready Tensor specifications.

- text classification
- fasttext
- quantization
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker

fastText is a library for efficient learning of word representations and sentence classification.

## See these for source:

Bag of Tricks for Efficient Text Classification
Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov
arXiv:1607.01759
https://doi.org/10.48550/arXiv.1607.01759

FastText.zip: Compressing text classification models
Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hérve Jégou, Tomas Mikolov
arXiv:1612.03651
https://doi.org/10.48550/arXiv.1612.03651

The trained classifier is also quantized with minimal loss in performance to create a storage efficient

The data preprocessing step includes

- lower casing all tokens
- remove tall words less than 2 characters
- remove stop words (list of stop words is in this repo)
- Porter stemming

In regards to processing the labels, a label encoder is used to turn the string representation of a class into a numerical representation.

Hyperparameter Tuning (HPT) is setup to fine-tune this parameters:

- learning rate of the algorithm
- Embedding size of word vectors
- word n-grams

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as clickbait, drug, and movie reviews as well as spam texts and tweets from Twitter.

This Text Classifier is written using Python as its programming language. fasttext pypi package is used to implement the main algorithm and evaluate the model. Sklearn, numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.
