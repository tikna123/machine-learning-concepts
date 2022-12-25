# Machine Learning Concepts 
It contains details about the generic concepts in Machine Learning. Below is the list of concepts:
* Regularization
* Bias vs Variance Tradeoff
* Handle overfitting
* Handle imbalance datasets
* Loss functions
* Evaluation Metrics
* Model caliberation
* Normalization
    * Batch Normalization
    * Layer normalization
* Hyperparameter tuning
* Optimization Algorithms
* Early stopping criteria
* Activation Functions
* Data Drifting
* Data leakage
* Handle outliers
* Data Augmentation
* K-cross validation
* Sampling of data
* Feature Engineering(in predictive modelling)
    * Handle missing values(in train & test)
    * Handle numerical values(Feature Scaling)
    * Handle categorical features
    * Feature Hashing
* Distributed training
    * Model ||ism
    * Data ||ism
* AB tests
    * p-value
    * How to decide sample size(when to stop experiments?)
* Knowledge distillation

## Regularization
* Regularization is one of the solution for the overfitting problem. This technique prevents the model from overfitting by adding extra information to it.
* In the Regularization technique, we reduce the magnitude of the independent variables by keeping the same number of variables.
* There are 2 techniques of regularization:
    ## Ridge Regression: 
        * It is also called L2-norm.
        * It is one of the types of linear regression in which we introduce a small amount of bias, known as Ridge regression penalty so that we can get better long-term predictions.
        * In this technique, the cost function is altered by adding the penalty term (shrinkage term), which multiplies the lambda with the squared weight of each individual feature. Therefore, the optimization function(cost function) becomes:
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im1.png) <br/>
        In the above equation, the penalty term regularizes the coefficients of the model, and hence ridge regression reduces the magnitudes of the coefficients that help to decrease the complexity of the model.
        ### Application
            * When we have the independent variables which are having high collinearity (problem of ) between them, at that time general linear or polynomial regression will fail so to solve such problems, Ridge regression can be used.
            * If we have more parameters than the samples, then Ridge regression helps to solve the problems.  
        ### Limitations
            * Not helps in Feature Selection: It decreases the complexity of a model but does not reduce the number of independent variables since it never leads to a coefficient being zero rather only minimizes it. Hence, this technique is not good for feature selection.
            * Model Interpretability: Its disadvantage is model interpretability since it will shrink the coefficients for least important predictors, very close to zero but it will never make them exactly zero. In other words, the final model will include all the independent variables, also known as predictors.

    ## Lasso Regression
        * It stands for Least Absolute and Selection Operator. It is called L1-norm.
        * It is similar to the Ridge Regression except that the penalty term includes the absolute weights instead of a square of weights. Therefore, the optimization function becomes:
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im2.png) <br/>
        * In this technique, the L1 penalty has the eﬀect of forcing some of the coeﬃcient estimates to be exactly equal to zero which means there is a complete removal of some of the features for model evaluation when the tuning parameter λ is suﬃciently large. Therefore, the lasso method also performs Feature selection and is said to yield sparse models.
        ### Limitation of Lasso Regression:
            * Problems with some types of Dataset: If the number of predictors is greater than the number of data points, Lasso will pick at most n predictors as non-zero, even if all predictors are relevant.
            * Multicollinearity Problem: If there are two or more highly collinear variables then LASSO regression selects one of them randomly which is not good for the interpretation of our model.
    ## Difference between Ridge and lasso regression
        * Ridge regression helps us to reduce only the overfitting in the model while keeping all the features present in the model. It reduces the complexity of the model by shrinking the coefficients whereas Lasso regression helps in reducing the problem of overfitting in the model as well as automatic feature selection.
        * Lasso Regression tends to make coefficients to absolute zero whereas Ridge regression never sets the value of coefficient to absolute zero.
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im3.png) <br/>
* Details: https://www.analyticsvidhya.com/blog/2021/05/complete-guide-to-regularization-techniques-in-machine-learning/

# Bias vs Variance Tradeoffs
* Beyond intrinsic uncertainty/noise in the data, any learning algorithm has error that comes from two sources:
    1. Bias
    2. Variance

    error(X) = noise(X) + bias(X) + variance(X)
* Bias is the algorithm's tendency to consistently learn the wrong thing by not taking into account all the information in the data (underfitting).
*  Variance is the algorithm's tendency to learn random things irrespective of the real signal by fitting highly flexible models that follow the error/noise in the data too closely (overfitting).
* Bias and variance are also measures of these tendencies. So, 'bias' is also used to denote by how much the average accuracy of the algorithm changes as input/training data changes. Similarly, 'variance' is used to denote how sensitive the algorithm is to the chosen input data.
* Ideally, we want to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Let's use the below showing different curve-fits to the same set of points to see what this means.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im4.png) <br/>
    We see that the linear (degree = 1) fit is an under-fit:
    1) It does not take into account all the information in the data (high bias), but
    2) It will not change much in the face of a new set of points from the same source (low variance).

    The high degree polynomial (degree = 20) fit, on the other hand, is an over-fit:
    1) The curve fits the given data points very well (low bias), but
    2) It will collapse in the face of subsets or new sets of points from the same source because it intimately takes all the data into account, thus losing generality (high variance).
    
    The ideal fit, naturally, is one that captures the regularities in the data enough to be reasonably accurate and generalizable to a different set of points from the same source. Unfortunately, in almost every practical setting, it is nearly impossible to do both simultaneously. Therefore, to achieve good performance on data outside the training set, a tradeoff must be made. This is referred to as the bias-variace trade-off.

    Details: 
    * http://scott.fortmann-roe.com/docs/BiasVariance.html
    * https://www.quora.com/What-is-an-intuitive-explanation-for-bias-variance-tradeoff
# Handle Overfitting
* Regularization
* Dataset Augmentation(will explain later)
* Parameter sharing
* Adding noise to input
* Adding noise to output
* Early stopping
* Ensemble method
* Dropout

# Handle Imbalance datasets
* There are different methods to handle imbalance datasets
    * Resample differently. Over-sample from your minority class and under-sample from your majority class, so you get a more balanced dataset.
    * Try different metrics other than correct vs wrong prediction. Try Confusion Matrix or ROC curve. Accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of the values. PR curve is still better choice than ROC curve for imbalance dataset.
    * Use Penalized Models. Like penalized-SVM and penalized-LDA. They put additional cost on the model for making classification mistakes on the minority class during training. These penalties can bias the model towards paying attention to minority class.
    * Try Anomaly Detection techniques and models often used there. Although that would probably be necessary if your data was even more Imbalanced.
    * We can also use One-class classifier. What you are doing here is that you are considering the smaller class an outlier and confidently learn the decision boundary for ONLY the larger class.
    * Details
        * http://www.chioka.in/class-imbalance-problem/
        * https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/
        * https://classeval.wordpress.com/simulation-analysis/roc-and-precision-recall-with-imbalanced-datasets/
        * https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba
        * https://towardsdatascience.com/class-imbalance-in-machine-learning-problems-a-practical-guide-4fb81eee0041 (best one)

# Loss functions
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im5.png) <br/>
* Loss functions define an objective against which the performance of your model is measured, and the setting of weight parameters learned by the model is determined by minimizing a chosen loss function. There are many loss functions such as:
    ## Mean Squared Loss(MSE)
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im6.png) <br/>
    ## cross-entrophy loss
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im7.PNG) <br/>
    ## Hinge loss
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im8.PNG) <br/>
* Given a particular model, each loss function has particular properties that make it interesting for example, the (L2-regularized) hinge loss comes with the maximum-margin property, and the mean-squared error when used in conjunction with linear regression comes with convexity guarantees.
    
# Evaluation Metrics
We can use different metrics to evaluate machine learning models. The choice of metric completely depends on the type of model and the implementation plan of the model.Here, we are going to only focus on classification evaluation metrics. Following are the list of important evaluation metrics:
* Confusion metrics
* Accuracy
* Precision/Recall
* F1-Score
* Area under ROC-Curve
* PR Curve

    ## Confusion metrics
    * The Confusion matrix is one of the most intuitive and easiest (unless of course, you are not confused)metrics used for finding the correctness and accuracy of the model.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im8.PNG) <br/>
    The Confusion matrix in itself is not a performance measure as such, but almost all of the performance metrics are based on Confusion Matrix and the numbers inside it.
        * Terms associated with Confusion metrics
            * **True Positives (TP)**: True positives are the cases when the actual class of the data point was 1(True) and the predicted is also 1(True)
            * **True Negatives (TN)**: True negatives are the cases when the actual class of the data point was 0(False) and the predicted is also 0(False
            * **False Positives (FP)**: False positives are the cases when the actual class of the data point was 0(False) and the predicted is 1(True). False is because the model has predicted incorrectly and positive because the class predicted was a positive one. (1)
            * **False Negatives (FN)**: False negatives are the cases when the actual class of the data point was 1(True) and the predicted is 0(False). False is because the model has predicted incorrectly and negative because the class predicted was a negative one. (0)
        * When to minimise what?
        We know that there will be some error associated with every model that we use for predicting the true class of the target variable. This will result in False Positives and False Negatives(i.e Model classifying things incorrectly as compared to the actual class). 
        There’s no hard rule that says what should be minimised in all the situations. It purely depends on the business needs and the context of the problem you are trying to solve. Based on that, we might want to minimise either False Positives or False negatives. <br/>
    
    1. **Minimising False Negatives**: Let’s say in our cancer detection problem example, out of 100 people, only 5 people have cancer. In this case, we want to correctly classify all the cancerous patients as even a very BAD model(Predicting everyone as NON-Cancerous) will give us a 95% accuracy(will come to what accuracy is). But, in order to capture all cancer cases, we might end up making a classification when the person actually NOT having cancer is classified as Cancerous. This might be okay as it is less dangerous than NOT identifying/capturing a cancerous patient since we will anyway send the cancer cases for further examination and reports. But missing a cancer patient will be a huge mistake as no further examination will be done on them.
    
    2. **Minimising False Positives**: For better understanding of False Positives, let’s use a different example where the model classifies whether an email is spam or not
    Let’s say that you are expecting an important email like hearing back from a recruiter or awaiting an admit letter from a university. Let’s assign a label to the target variable and say,1: “Email is a spam” and 0:”Email is not a spam”
    Suppose the Model classifies that important email that you are desperately waiting for, as Spam(case of False positive). Now, in this situation, this is pretty bad than classifying a spam email as important or not spam since in that case, we can still go ahead and manually delete it and it’s not a pain if it happens once a while. So in case of Spam email classification, minimising False positives is more important than False Negatives.
    
    ## Accuracy
    * Accuracy in classification problems is the number of correct predictions made by the model over all kinds of predictions made. 


    





