# Machine Learning Examples

## Requirements
+ Docker
+ Docker Compose
+ Jupyter Notebook(https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
You are likely to use a `jupyter/scipy-notebook` image.(https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-scipy-notebook)
+ Python 3.6

## Sections & Collection Resources
### Data Preprocessing
+ Missing data
+ Endcoding categorical data
+ Feature Scaling


### Regression
+ Linear Regression
+ Multiple Linear Regressionb
+ Decision Tree Regression
+ Random Forest Regression


### Classification
+ Logistic Regression
+ Classification
+ K Nearest Neighbors
+ SVM
+ Naive Bayes
+ Decision Tree Classification
+ Random Forest Classification


### Clustering
+ K Means


### Natural Language Processing
[Understand Sentiment](https://towardsdatascience.com/making-computers-understand-the-sentiment-of-tweets-1271ab270bc7)

[TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder-large/3)

[Notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb#scrollTo=MSeY-MUQo2Ha)

[TensorFlow Hub Github Examples](https://github.com/tensorflow/hub/tree/master/examples)

[NLP Transfer learning techniques to predict Tweet stance](https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde)

[NLP Transfer Github](https://github.com/prrao87/tweet-stance-prediction)


## Devops
### Create base image for Machine Learning
```
$cd devops
$docker build -t nhatthai/machine-learning-python3.6 .
$docker push nhatthai/machine-learning-python3.6
```

### Using Jupyter Notebook in docker
```
$docker run --rm -p 8888:8888 jupyter/scipy-notebook:17aba6048f44
```

```
$cd devops
$docker-compose up
```

Import libraries
```
$docker exec [container_id] pip install nltk
```

## Reference
[Docker for Data Science](https://www.dataquest.io/blog/docker-data-science/)

[Jupyter Image](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)

[Machine Learning A-Z](https://www.superdatascience.com/machine-learning/)