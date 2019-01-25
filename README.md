# Machine Learning Examples

## Requirements
+ Docker
+ Docker Compose
+ Jupyter Notebook(https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
You are likely to use a `jupyter/scipy-notebook` image.(https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-scipy-notebook)
+ Python 3.6

## Sections & Collection Resources
+[Overview ML Beginner](https://towardsdatascience.com/introduction-to-machine-learning-for-beginners-eed6024fdb08)

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
+ [Logistic Regression](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)
+ Classification
+ [K Nearest Neighbors and Naive Bayes](https://towardsdatascience.com/playlist-classification-on-spotify-using-knn-and-naive-bayes-classification-35a279b7e255)
+ SVM
+ Decision Tree Classification
+ Random Forest Classification
+ [Stack for Classification](https://towardsdatascience.com/automate-stacking-in-python-fc3e7834772e)

### Clustering
+ [K Means](https://towardsdatascience.com/k-means-clustering-implementation-2018-ac5cd1e51d0a)
+ [Var Model](https://towardsdatascience.com/prediction-task-with-multivariate-timeseries-and-var-model-47003f629f9)

### Natural Language Processing
+ [Understand Sentiment](https://towardsdatascience.com/making-computers-understand-the-sentiment-of-tweets-1271ab270bc7)
+ [TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder-large/3)
+ [Notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb#scrollTo=MSeY-MUQo2Ha)
+ [TensorFlow Hub Github Examples](https://github.com/tensorflow/hub/tree/master/examples)
+ [NLP Transfer learning techniques to predict Tweet stance](https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde)
+ [NLP Transfer Github](https://github.com/prrao87/tweet-stance-prediction)
+ [Text Preprocessing Methods](https://towardsdatascience.com/nlp-learning-series-part-1-text-preprocessing-methods-for-deep-learning-20085601684b)

### Deep Learning
+ [Counting Parameters](https://towardsdatascience.com/counting-no-of-parameters-in-deep-learning-models-by-hand-8f1716241889)
+ FFNNs(Feed Forward Neural Network)
+ RNNs(Recurrent Neural Network)
+ [CNNs(Convolutional Neural Network)](https://towardsdatascience.com/understanding-convolutional-neural-networks-through-visualizations-in-pytorch-b5444de08b91)
+ [Mask RCNN](https://towardsdatascience.com/mask-r-cnn-for-ship-detection-segmentation-a1108b5a083)


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
+ [Docker for Data Science](https://www.dataquest.io/blog/docker-data-science/)
+ [Jupyter Image](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
+ [Machine Learning A-Z](https://www.superdatascience.com/machine-learning/)