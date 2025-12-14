# Quick ML Refresher
This just the from the scratch implementation of fundamental ML concepts.

## Linear Algebra

For a Quick Refresher of Linear Algebra I have implemented, PCA, followed the traditional steps,
1) scale the input data
2) compute its covariance matrix
3) find the EigenValues and Eigenvectors of the covariance matrix
4) using Eigenvalues, create Projection Matrix and pick the top-k eigenvectors
5) transform the scaled input using the top k-projection matrix


## Probability

for a Probability quick refresher, I have concentrated on Bayesian probability and implemented MultiNomial Naive Bayes for email spam detection and its corresponding TextVectorizer
1) we find the P("spam") and P("not spam"), store it log(class_priori)
2) found its P(w_i | class) and stored it as log_word_per_class_prob
3) performed Laplace smoothing
4) and for predicting I added log_class_priori with the likelihood to predict it

and built a very simple Text Vectorizer

## Calculus

for the calculus quicj refresher, I have built a basic micrograd-autograd, followed by a simple MLP to test its implementation.
