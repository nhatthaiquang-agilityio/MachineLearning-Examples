# Machine Learning Examples

## Requirements
+ Docker
+ Docker Compose
+ Jupyter Notebook(https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
You are likely to use a `jupyter/scipy-notebook` image.(https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-scipy-notebook)
+ Python 3.6

### Data Preprocessing
+ Missing data
+ Endcoding categorical data
+ Feature Scaling


### Regression
+ Liner Regression
+ Multiple Liner Regression
+ Decision Tree Regression
+ Random Forest Regression


### Classification
+ Classification
+ K Nearest Neighbors
+ SVM
+ Naive Bayes
+ Decision Tree Classification
+ Random Forest Classification


### Clustering
+ K Means


### Natural Language Processing


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

[Machine Learning A-Z](https://www.superdatascience.com/machine-learning/)

[Jupiter Image](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
