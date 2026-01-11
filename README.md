# Project Overview

This repository contains a machine learning pipeline developed as part of an advanced ML coursework at UCL.

The goal was to explore various techniques on real-world biomedical datasets, with an emphasis on interpretability and methodology.



# Problem and Dataset

 - **Dataset:** The Wisconsin diagnostic dataset for breast cancer was used for this analysis.
 - **Task:** To see whether a binary classification (healthy vs unhealthy) can be made despite the high-dimensional feature space relative to the relatively small sample size.

# Methodology

The dataset was approached in a "blind" fashion, assuming no prior knowledge to bias the analysis. Any features that emerge must do so organically through our pipeline.

## Pipeline overview

1. Data import, cleaning, uni-/bi-variate analysis
2. Dimensional reduction using PCA
3. Unsupervised clustering using GMM and K-means
    - comparison between raw features vs PCA eigenbasis
4. Supervised classification (MLP, k-NN, AdaBoost,...)
5. Evaluation and comparison

#### Data cleaning

The data is checked after import to ensure that it is clean, consistent, and does not contain missing values or type mismatches.

#### Principal component analysis

The Wisconsin dataset contains 31 unique features. To identify features of importance classically would be time consuming, difficult to interpret for humans, and not digestible for most.

Ideally, we would extract "prime features" that heavily correlate with the diagnosis, or better yet, combination of features that yield a strong correlation with the diagnosis. 

Using PCA, a linear combination of all features (bar the `diagnosis` feature to avoid biasing) is formed to create a new eigenbasis. Although human interpretability suffers a bit, the new basis might lead to better classification of the diagnosis.


#### Clustering

Unsupervised clustering methods (here Gaussian Mixture Models and K-Means), were used to compare clustering of PCA-informed "prime features" against clustering in the (heavily reduced) principal component feature space.

This is discover any structures that we may not fully understand and to quantify whether we can separate the labels from either a select few features or the new eigenbasis.

#### Supervised classification

Here, some of `sklearn`'s models are leveraged to test and quantify the structure revealed via PCA + unsupervised clustering. At this stage, we try to see the extent at which are able to identify the correct diagnosis based on the features used to train the models.

**The models used are:**

- Multi-layer Perceptron
- k-nearest neighbours
- AdaBoost (Random Forest)
- SVC (RBF)
- Random Forest
- SVC (Linear)
- Grandient Boost (Random Forest)
- Decision Tree

# Models and Results


### Preprocessing and PCA

Starting with a basic correlation matrix, the data shows high correlation between features of similar type (i.e. perimeter correlates strongly with radius)

<img width="628" height="706" alt="image" src="https://github.com/user-attachments/assets/0166904c-c3b0-45a3-823d-456bb9a7c7db" />


This points to multiple features being either highly redundant or highly correlated to the extent where it causes compounding bias towards certain types (length) and may shadow other features of different dimensional type such as "texture".

In such datasets, the smarter approach would be to standardise the data to avoid emphasis on specific types, however in this example pipeline this was not done deliberately to provide a simpler output in terms of how the data is thought about and interpreted. 

PCA was used to see whether a rotated basis might be more informative that the total feature space. In the non-standardised dataset, the results showed the total variance in the system is dominated by a single **linear** combination of features where a single principal component accounts for >98% of the total variance.  This effect is label-independent.

<img width="789" height="390" alt="image" src="https://github.com/user-attachments/assets/c6a06289-31fe-488c-bd20-0de3dc071761" />


Given that the cummulative variance exceeds the typical values of 80-90% with a single PC, a liberal choice of 3 PCs to describe the reduced dataset is already plenty. Their scatterplots show that indeed the variance is dominated by PC1.

<img width="1489" height="789" alt="image" src="https://github.com/user-attachments/assets/52bfc474-ac49-48c2-9935-7940c7d7a26b" />


Based on the loadings of PC1 and PC2 (which combined account for more than 99% of the total variance) we can see that the most impactful features are those of **tumour size** (area, perimeter, radius...). This is of course fully expected since we (deliberately) did not standardise the data but also fully reassuring that have a metric that is fully interpretable.


<img width="629" height="470" alt="image" src="https://github.com/user-attachments/assets/c0a5eab0-cf08-49d0-8b23-d07b068ac8a5" />


A naive plotting of the PCA-informed feature space where data are coloured based on their TRUE labels shows a relatively neat separation validating the rationale so far.

<img width="1414" height="495" alt="image" src="https://github.com/user-attachments/assets/cc11e5dd-ffda-46ac-9acc-d55d8693455f" />


### Clustering using GMM and K-Means

<img width="989" height="495" alt="image" src="https://github.com/user-attachments/assets/513d90b8-2e54-49a4-a98e-b4d671bf172d" />


A basic clustering with 2 components (reflecting the label binary of "healthy" vs "cancerous") shows that GMMs are slightly better are identifying the correct group based on the adjusted random score (ARI) when performed on the raw features.

On the other hand, GMMs increased their accuracy by 50% when operating on the PCA eigenbasis:

<img width="634" height="475" alt="image" src="https://github.com/user-attachments/assets/5ae342b8-551c-4e50-9253-4143dc1c1781" />



### Classification using supervised models

In this case, Random Forests performed best if given access to the full raw dataset reaching accuracies of ~97%, with MLP being a close contender achieving 96% with just 4 PCA-informed features. The reduced PCA eigenbasis also did remarkably well scoring consistently close to 94% across all models and using just 3 PCs.

<img width="765" height="436" alt="image" src="https://github.com/user-attachments/assets/8561a4fe-6eb3-4bf9-8cfb-7594424f123b" />

The mismatch between the feature importances between supervised and PCA is fully expected given that the data was not standardised at the start and highlights that other factors such as `concave points_worst` being a powerful determinant of the true diagnostic label.


<img width="989" height="1475" alt="image" src="https://github.com/user-attachments/assets/4eba5cf8-c7e6-454c-9ebe-6fe022ecd862" />


# Limitations 



# What I learned from this

Labelled data are incredibly powerful for classification. However, labels are not always true and may lead to false conclusions. It is always important to have a pipeline that involves unsupervised learning to check whether the structure emerging is supported by the highlighted labels from a supervised model.

On the flip side, unsupervised modelling alone can be harmful, as clusters and "structure" will always be found no matter what dataset is used. Even there, interpretation is not always possible as it is not always possible to extract whether there is an important feature or whether a combination of features matter.

Unsupervised methods help define a hypothesis, which is then validated/tested in a supervised model.

The most interesting was how sensitive the results were to the preprocessing of the data. Whether the data are standardised or not can dramatically affect how models perceive the data and their respective feature importances. 
