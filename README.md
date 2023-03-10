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
    * In this technique, the L1 penalty has the e???ect of forcing some of the coe???cient estimates to be exactly equal to zero which means there is a complete removal of some of the features for model evaluation when the tuning parameter ?? is su???ciently large. Therefore, the lasso method also performs Feature selection and is said to yield sparse models.
    ### Limitation of Lasso Regression:
    * Problems with some types of Dataset: If the number of predictors is greater than the number of data points, Lasso will pick at most n predictors as non-zero, even if all predictors are relevant.
    * Multicollinearity Problem: If there are two or more highly collinear variables then LASSO regression selects one of them randomly which is not good for the interpretation of our model.
## Difference between Ridge and lasso regression
* Ridge regression helps us to reduce only the overfitting in the model while keeping all the features present in the model. It reduces the complexity of the model by shrinking the coefficients whereas Lasso regression helps in reducing the problem of overfitting in the model as well as automatic feature selection.
* Lasso Regression tends to make coefficients to absolute zero whereas Ridge regression never sets the value of coefficient to absolute zero.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im3.png) <br/>
* References: https://www.analyticsvidhya.com/blog/2021/05/complete-guide-to-regularization-techniques-in-machine-learning/

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

    References: 
    * http://scott.fortmann-roe.com/docs/BiasVariance.html
    * https://www.quora.com/What-is-an-intuitive-explanation-for-bias-variance-tradeoff
# Handle Overfitting
* Regularization
* Dataset Augmentation
* Parameter sharing
* Adding noise to input
* Adding noise to output
* Early stopping
* Ensemble method
* Dropout
* k-cross validation(detect overfitting)
* Treat the problem as a anamoly detection

# Handle Imbalance datasets
* There are different methods to handle imbalance datasets
    * Resample differently. Over-sample from your minority class and under-sample from your majority class, so you get a more balanced dataset.
    * Try different metrics other than correct vs wrong prediction. Try Confusion Matrix or ROC curve. Accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of the values. PR curve is still better choice than ROC curve for imbalance dataset.
    * Use Penalized Models. Like penalized-SVM and penalized-LDA. They put additional cost on the model for making classification mistakes on the minority class during training. These penalties can bias the model towards paying attention to minority class.
    * Try Anomaly Detection techniques and models often used there. Although that would probably be necessary if your data was even more Imbalanced.
    * We can also use One-class classifier. What you are doing here is that you are considering the smaller class an outlier and confidently learn the decision boundary for ONLY the larger class.
    * References
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
        There???s no hard rule that says what should be minimised in all the situations. It purely depends on the business needs and the context of the problem you are trying to solve. Based on that, we might want to minimise either False Positives or False negatives. <br/>
    
    1. **Minimising False Negatives**: Let???s say in our cancer detection problem example, out of 100 people, only 5 people have cancer. In this case, we want to correctly classify all the cancerous patients as even a very BAD model(Predicting everyone as NON-Cancerous) will give us a 95% accuracy(will come to what accuracy is). But, in order to capture all cancer cases, we might end up making a classification when the person actually NOT having cancer is classified as Cancerous. This might be okay as it is less dangerous than NOT identifying/capturing a cancerous patient since we will anyway send the cancer cases for further examination and reports. But missing a cancer patient will be a huge mistake as no further examination will be done on them.
    
    2. **Minimising False Positives**: For better understanding of False Positives, let???s use a different example where the model classifies whether an email is spam or not
    Let???s say that you are expecting an important email like hearing back from a recruiter or awaiting an admit letter from a university. Let???s assign a label to the target variable and say,1: ???Email is a spam??? and 0:???Email is not a spam???
    Suppose the Model classifies that important email that you are desperately waiting for, as Spam(case of False positive). Now, in this situation, this is pretty bad than classifying a spam email as important or not spam since in that case, we can still go ahead and manually delete it and it???s not a pain if it happens once a while. So in case of Spam email classification, minimising False positives is more important than False Negatives.
    
    ## Accuracy
    * Accuracy in classification problems is the number of correct predictions made by the model over all kinds of predictions made.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im9.png) <br/>
    In the Numerator, are our correct predictions (True positives and True Negatives)(Marked as red in the fig above) and in the denominator, are the kind of all predictions made by the algorithm(Right as well as wrong ones).
    **When to use accuracy**: Accuracy is a good measure when the target variable classes in the data are nearly balanced.
    **When NOT to use Accuracy**: Accuracy should NEVER be used as a measure when the target variable classes in the data are a majority of one class.
    
    ## Precision
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im10.png) <br/>
    ## Recall
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im11.png) <br/>
    
    ## F1-Score: 
    Harmonic mean of precision and recall.
    F1 Score = 2 * Precision * Recall / (Precision + Recall)
    If one number is really small between precision and recall, the F1 Score kind of raises a flag and is more closer to the smaller number than the bigger one, giving the model an appropriate score rather than just an arithmetic mean.
    
    ## Area under ROC curve
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im12.png) <br/>
    * The AUC ROC value is between 0 to 1.
    * For a model which gives class as output, will be represented as a single point in ROC plot.
    * In case of probabilistic model, we were fortunate enough to get a single number which was AUC-ROC. But still, we need to look at the entire curve to make conclusive decisions. It is also possible that one model performs better in some region and other performs better in other.
    * Each point in the ROC curve corresponds to model result(TPR and FPR) given some threshold value.
    * AUC is scale-invariant. It measures how well predictions are ranked, rather than their absolute values.
    * AUC is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.
    * References
        * https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
        * https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
        * https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
        * https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab (how to calculate AUC using python)
        * https://blog.revolutionanalytics.com/2016/08/roc-curves-in-two-lines-of-code.html
        * https://blog.revolutionanalytics.com/2016/11/calculating-auc.html
        
    ## PR curve
     ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im13.png) <br/>
    * PR curve is the graph between precision and recall where precision is on y axis and recall is on the x axis. A good PR curve has greator AUC.
    * The Precision-Recall AUC is just like the ROC AUC, in that it summarizes the curve with a range of threshold values as a single score. The score can then be used as a point of comparison between different models on a binary classification problem where a score of 1.0 represents a model with perfect skill.
    
    ## Tips about when to use which metrics
    * For Imbalance dataset never used accuracy.
    * When you have imbalance datasets with +ve class as minority class, better to use PR curve as compare to ROC, as ROC doesn't give much information about the minority class prediction, even if some classifier gives lot of FP for minority class, the ROC will look just fine. ROC cuver is better than PR cuver when you have imbalance datasets.
    * Use ROC when the positives are the majority or switch the labels and use precision and recall.

# Model Caliberation
* Let???s consider a binary classification task and a model trained on this task. Without any calibration, the model???s outputs cannot be interpreted as true probabilities. For instance, for a cat/dog classifier, if the model outputs that the prediction value for an example being a dog is 0.4, this value cannot be interpreted as a probability. To interpret the output of such a model in terms of a probability, we need to calibrate the model. 
* Surprisingly, most models out of the box are not calibrated and their prediction values often tend to be under or over confident. What this means is that, they predict values close to 0 and 1 in many cases where they should not be doing so.
* To better understand why we need model calibration, let???s look into the previous example whose output value is 0.4 . Ideally, what we would want this value to represent is the fact that if we were to take 10 such pictures and the model classified them as dogs with probabilities around 0.4 , then in reality 4 of those 10 pictures would actually be dog pictures. This is exactly how we should interpret outputs from a calibrated model.
However, if the model is not calibrated, then we should not expect that this score would mean that 4 out of the 10 pictures will actually be dog pictures.
## Reliability curve
* The reliability curve is a nice visual method to identify whether or not our model is calibrated. First we create bins from 0 to 1. Then we divide our data according to the predicted outputs and place them into these bins. For instance if we bin our data in intervals of 0.1, we will have 10 bins between 0 and 1. Say we have 5 data points in the first bin, i.e we have 5 points (0.05,0.05,0.02,0.01,0.02) whose model prediction range lies between 0 and 0.1. Now on the X axis we plot the average of these predictions i.e 0.03 and on the Y axis, we plot the empirical probabilities, i.e the fraction of data points with ground truth equal to 1. Say out of our 5 points, 1 point has the ground truth value 1. In that case our y value will be 1/5 = 0.2. Hence the coordinates of our first point are [0.03,0.2]. We do this for all the bins and connect the points to form a line. We then compare this line to the line
* y = x and assess the calibration. When the dots are above this line the model is under-predicting the true probability and if they are below the line, model is over-predicting the true probability.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im14.png) <br/>
As you can see the model is over-confident till about 0.6 and then under-predicts around 0.8
* References:
    * https://towardsdatascience.com/a-comprehensive-guide-on-model-calibration-part-1-of-4-73466eb5e09a
    * https://wttech.blog/blog/2021/a-guide-to-model-calibration/
    * https://www.analyticsvidhya.com/blog/2022/10/calibration-of-machine-learning-models/
    * https://neptune.ai/blog/brier-score-and-model-calibration
# Normalization
   ## Batch Normalization
   * Batch-Normalization (BN) is an algorithmic method which makes the training of Deep Neural Networks (DNN) faster and more stable.
   * It consists of normalizing activation vectors from hidden layers using the first and the second statistical moments (mean and variance) of the current batch. This normalization step is applied right before (or right after) the nonlinear function.
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im15.png) <br/>
   * The distribution of the inputs to layers deep in the network may change after each mini-batch when the weights are updated. This can cause the learning algorithm to forever chase a moving target. This change in the distribution of inputs to layers in the network is referred to the technical name ???internal covariate shift.???
   * Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.
   * Batch normalization allow suboptimal start. We don't need to worry too much about the weight initialization.
   * **Batch normalization at training**
        At each hidden layer, Batch Normalization transforms the signal as follow :
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im16.png) <br/>
        The BN layer first determines the mean ???? and the variance ???? of the activation values across the batch.
        It then normalizes the activation vector. That way, each neuron???s output follows a standard normal distribution across the batch.
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im17.png) <br/>
   * **Batch Normalization at evaluation**
        * Unlike the training phase, we may not have a full batch to feed into the model during the evaluation phase.
            To tackle this issue, we compute (????_pop , ??_pop), such as :
            ????_pop : estimated mean of the studied population ;
            ??_pop : estimated standard-deviation of the studied population.
            Those values are computed using all the (????_batch , ??_batch) determined during training, and directly used at evaluation.
   * References:
        * https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
        * https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338
        * https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/
## Layer Normalization
   * Unlike batch normalization, Layer Normalization directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer so the normalization does not introduce any new dependencies between training cases. It works well for RNNs and improves both the training time and the generalization performance of several existing RNN models. More recently, it has been used with Transformer models.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im18.png) <br/>
    We compute the layer normalization statistics over all the hidden units in the same layer as follows:
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im19.PNG) <br/>
    where  denotes the number of hidden units in a layer. Under layer normalization, all the hidden units in a layer share the same normalization terms  and , but different training cases have different normalization terms. Unlike batch normalization, layer normalization does not impose any constraint on the size of the mini-batch and it can be used in the pure online regime with batch size 1.
   * **why   NLP uses layer normalization**
        * In NLP, sentence length often varies, it is more suitable to use layer normalization(average across feature dimension)
        * In Layer normalization, it is calculated for each instance independently exactly same
        computation at training & test time.
   * Order of batch norm: FC layer -> BN -> RELU -> Dropout -> next FC

# Hyperparameter tuning
* The best way to think about hyperparameters is like the settings of an algorithm that can be adjusted to optimize performance, just as we might turn the knobs of an AM radio to get a clear signal (or your parents might have!). While model parameters are learned during training ??? such as the slope and intercept in a linear regression ??? hyperparameters must be set by the data scientist before training.
* Common examples of hyperparameters are penalties for an algorithm (l1 or l2 or elastic net), a number of layers for neural networks, number of epochs, batch size, activation functions, learning rate, optimization algorithms (SGD, adam, etc), loss functions, and many more.
* Grid search is one Hyperparameters tuning algorithm where we try all possible combinations of hyperparameters. Trying all possible combinations of hyperparameters can take a lot of time (sometimes even days if there is a lot of data) even on powerful computers.
* Python has libraries like Optuna, scikit-optimize, hyperopt, keras-tuner, bayes_opt, etc specifically designed for faster hyperparameters tuning.
* References
    * https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning
    * https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Optimization Algorithms
* In Machine learning, for any learning task we need an algorithm that maps the examples of inputs to that of the outputs and an optimization algorithm. An optimization algorithm finds the value of the parameters(weights) that minimize the error when mapping inputs to outputs. These optimization algorithms or optimizers widely affect the accuracy of the ML model. They as well as affect the speed training of the model(specially when you have large amount of training data in deep learning).
* While training the deep learning optimizers model, we need to modify weights in each epoch and minimize the loss function. An optimizer is a function or an algorithm that modifies the attributes of the neural network, such as weights and learning rate. Thus, it helps in reducing the overall loss and improve the accuracy.
* Following are the important Optmization algorithms
    ## Gradient Descent
    * Gradient descent is a first-order optimization algorithm which is dependent on the first order derivative of a loss function.
    * It finds the local minima of a differentiable funciton(loss funciton). It is simply used to find the values of a function's parameters (coefficients) that minimize a cost function as far as possible.
    * We can start by defining the initial parameter's values and from there gradient descent uses calculus to iteratively adjust the values so they minimize the given cost-function. 
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im20.png) <br/>
        The above equation computes the gradient of the cost function J(??) w.r.t. to the parameters/weights ?? for the entire training dataset:
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im21.png) <br/>
        Our aim is to get to the bottom of our graph(Cost vs weights), or to a point where we can no longer move downhill???a local minimum.
    * Code snippet for Gradient descent
        ```python
        for i in range(nb_epochs):   
            params_grad = evaluate_gradient(loss_function, data, params)           
            params = params - learning_rate * params_grad
        ```
    * ***Advantages***
        * Easy to compute, implement and understand
    * ***Disadvantages***
        * Weights are changed after calculating the gradient on the whole dataset. So, if the dataset is too large then this may take years to converge to the minima.
        * may trap into local minima
        * Requires large memory to calculate the gradient on the whole dataset.
    
    ## Stochastic Gradient Descent (SGD)
    * Gradient Descent has a disadvantage that it requires a lot of memory to load the entire dataset of n-points at a time to compute the derivative of the loss function.
    * In SGD, model parameters are altered after computation of loss on each training example. So, if the dataset contains 1000 rows SGD will update the model parameters 1000 times in one cycle of dataset instead of one time as in Gradient Descent. 
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im22.png) <br/>
    In the diagram, we can see that there are more oscillation in SGD as compare to GD. But each step is lot faster to compute for SGD as compare to GD.
    * Code snippet for Stochastic Gradient descent(SGD)
         ```python
        for i in range(nb_epochs):
            np.random.shuffle(data)
            for example in data:
                params_grad = evaluate_gradient(loss_function, example, params)
                params = params - learning_rate * params_grad
        ```
    * ***Advantages***
        * Memory requirement is less compared to the GD algorithm as the derivative is computed taking only 1 point at once.
    * ***Disadvantages*** 
        * The time required to complete 1 epoch is large compared to the GD algorithm
        * Takes a long time to converge.
        * May stuck at local minima.

    ## Mini Batch Stochastic Gradient Descent (MB-SGD)
    * MB-SGD overcomes the drawbacks of both SGD and GD. Only a subset of the dataset is used for calculating the loss function. Since we are using a batch of data instead of taking the whole dataset, fewer iterations are needed. That is why the mini-batch gradient descent algorithm is faster than both stochastic gradient descent and batch gradient descent algorithms. 
    * It needs a hyperparameter that is ???mini-batch-size???, which needs to be tuned to achieve the required accuracy. Although, the batch size of 32 is considered to be appropriate for almost every case.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im23.png) <br/>

    * Code snippet for Mini Batch Stochastic Gradient Descent
         ```python
        for i in range(nb_epochs):
            np.random.shuffle(data)
            for batch in get_batches(data, batch_size=50):
                params_grad = evaluate_gradient(loss_function, batch, params)
                params = params - learning_rate * params_grad
        ```
    * ***Advantages***
        * Less time complexity to converge compared to standard SGD algorithm.
    * ***Disadvantages*** 
        * The update of MB-SGD is much noisy compared to the update of the GD algorithm.
        * Take a longer time to converge than the GD algorithm.
        * May stuck at local minima.
    
    ## SGD with Momentum
    * SGD has trouble navigating ravines, i.e. areas where the surface curves much more steeply in one dimension than in another [4], which are common around local optima. In these scenarios, SGD oscillates across the slopes of the ravine while only making hesitant progress along the bottom towards the local optimum
    * It calculates the exponential weighting average of the updates to give more weightage to recent updates compared to the previous update.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im124.png) <br/>
    * It helps accelerate SGD in the relevant direction and dampens oscillation as shown in the image.It does this by adding a fraction ?? of the update vector of the past time step to the current update vector:
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im25.png) <br/>
    Note: Some implementations exchange the signs in the equations. The momentum term ??
    is usually set to 0.9 or a similar value. 
    * Essentially, when using momentum, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way (until it reaches its terminal velocity if there is air resistance.The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.
    * ***Advantages***
        * Has all advantages of the SGD.
        * Converges faster than the GD.
    * ***Disadvantages***
        * We need to compute one more variable for each update.

    ## Nesterov Accelerated Gradient (NAG)
    * The idea of the NAG algorithm is very similar to SGD with momentum with a slight variant.
    * Momentum may be a good method but if the momentum is too high the algorithm may miss the local minima and may continue to rise up. So, to resolve this issue the NAG algorithm was developed. It is a look ahead method. We know we???ll be using ??.V(t???1) for modifying the weights so, ???????V(t???1) approximately tells us the future location. Now, we???ll calculate the cost based on this future parameter rather than the current one.
        V(t) = ??.V(t???1) + ??. ???(J(?? ??? ??V(t???1)))/?????
    and then update the parameters using ?? = ?? ??? V(t)
    * Again, we set the momentum term ???? to a value of around 0.9. While Momentum first computes the current gradient (small brown vector in Image 4) and then takes a big jump in the direction of the updated accumulated gradient (big brown vector), NAG first makes a big jump in the direction of the previously accumulated gradient (green vector), measures the gradient and then makes a correction (red vector), which results in the complete NAG update (red vector). This anticipatory update prevents us from going too fast and results in increased responsiveness, which has significantly increased the performance of RNNs on a number of tasks.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im26.png) <br/>
    Both NAG and SGD with momentum algorithms work equally well and share the same advantages and disadvantages.
    * It checks the lookahead gradient and update based on that. Reduce the oscillations. With vanilla momentum, it can be difficult to converge(overshoot) at minima.


    ## Adaptive Gradient Descent(AdaGrad)
    * For all the previously discussed algorithms the learning rate remains constant. So the key idea of AdaGrad is to have an adaptive learning rate for each of the weights.
    * It performs smaller updates for parameters associated with frequently occurring features, and larger updates for parameters associated with infrequently occurring features.
    * For brevity, we use gt to denote the gradient at time step t. gt,i is then the partial derivative of the objective function w.r.t. to the parameter ??i at time step t, ?? is the learning rate and ????? is the partial derivative of loss function J(??i)
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im27.png) <br/>
    In its update rule, Adagrad modifies the general learning rate ?? at each time step t for every parameter ??i based on the past gradients for ??i:
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im28.png) <br/>
    where Gt is the sum of the squares of the past gradients w.r.t to all parameters ??.
    The benefit of AdaGrad is that it eliminates the need to manually tune the learning rate; most leave it at a default value of 0.01.
    Its main weakness is the accumulation of the squared gradients(Gt) in the denominator. Since every added term is positive, the accumulated sum keeps growing during training, causing the learning rate to shrink and becoming infinitesimally small and further resulting in a vanishing gradient problem.
    * ***Advantages***
        * No need to update the learning rate manually as it changes adaptively with iterations.
    * ***Disadvantages***
        * As the number of iteration becomes very large learning rate decreases to a very small number which leads to slow convergence.
    
    ## Ada delta
    * The problem with the previous algorithm AdaGrad was learning rate becomes very small with a large number of iterations which leads to slow convergence. To avoid this, the AdaDelta algorithm has an idea to take an exponentially decaying average.
    * Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done. Compared to Adagrad, in the original version of Adadelta, you don't have to set an initial learning rate.
    * Instead of inefficiently storing w previous squared gradients, the sum of gradients is recursively defined as a decaying average of all past squared gradients. The running average E[g2]t at time step t then depends only on the previous average and current gradient:
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im29.png) <br/>
    With Adadelta, we do not even need to set a default learning rate, as it has been eliminated from the update rule.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im30.png) <br/>

    ## RMSprop
    * RMSprop in fact is identical to the first update vector of Adadelta that we derived above:
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im31.png) <br/>
    * RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients. Hinton suggests ?? be set to 0.9, while a good default value for the learning rate ?? is 0.001.
    * RMSprop and Adadelta have both been developed independently around the same time stemming from the need to resolve Adagrad's radically diminishing learning rates

    ## Adaptive Moment Estimation (Adam)
    * Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum.
    * Adam computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients vt like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients mt, similar to momentum. Whereas momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction, which thus prefers flat minima in the error surface.
    * Hyper-parameters ??1, ??2 ??? [0, 1) control the exponential decay rates of these moving averages. We compute the decaying averages of past and past squared gradients mt and vt respectively as follows:
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im32.png) <br/>
    mt and vt are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively, hence the name of the method.
    * Adam is considered the best algorithm amongst all the algorithms discussed above.
    * Summary: 
        * Combination of momentum and rmsprop
        * RMSProp: adapt the learning rate. Divide the learning rate by accumulated history of gradients(handle sparse feature). Exponential weightage of second moment.
        * Momentum: Use accumulated history of gradients.
        * Adam is the best optimizer which works well for most of the practical applications. It takes care of sparse feature(RMSprop). Reduce oscillations. It updates more for sparse features and less for dense features.  When doing the update checks the history of updates by using exponentially weighted average. It reduces the oscillations and very fast compare to vanilla GD. In Adam, we have different learning rates for different features.

    * References(for all optimization algorithms)
        * https://www.kdnuggets.com/2020/12/optimization-algorithms-neural-networks.html
        * https://ruder.io/optimizing-gradient-descent/
        * https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
        * https://d2l.ai/chapter_optimization/
        * https://towardsdatascience.com/complete-guide-to-adam-optimization-1e5f29532c3d

# Early stopping Criteria
* A problem with training neural networks is in the choice of the number of training epochs to use. Too many epochs can lead to overfitting of the training dataset, whereas too few may result in an underfit model. Early stopping is a method that allows you to specify an arbitrary large number of training epochs and stop training once the model performance stops improving on a hold out validation dataset. 
* ***Drawbacks***
    * We need validation dataset to use early stopping.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im33.png) <br/>
* References
    * https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html(pytorch library for early stopping)
    * https://jeande.medium.com/early-stopping-explained-62eebce1127e
    * https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/#:~:text=There%20are%20three%20elements%20to,choice%20of%20model%20to%20use.

# Activation Functions
* The purpose of activation functions in the neural network is to introduce non-linearity when modelling any learning function. It defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network.
* All hidden layers typically use the same activation function. The output layer will typically use a different activation function from the hidden layers and is dependent upon the type of prediction required by the model.
* Activation functions are also typically differentiable, meaning the first-order derivative can be calculated for a given input value. This is required given that neural networks are typically trained using the backpropagation of error algorithm that requires the derivative of prediction error in order to update the weights of the model.
* Following are the imortant activation functions that is used in neural network.
## Sigmoid or logistic function
* It converts or squeezes the hidden layer output between 0 to 1. Therefore, it is used in the models
where we have to predict the probability as an output. The function is differentiable.
* It is rarely used nowadays.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im34.png) <br/>
* ***Drawbacks***
    * Gradient vanishes if z>>0 or z<<0, resulted in no updates in parameters
    * Sigmoids are computational heavy.
    * It is not zero-centered. i.e. if there are two parameters then the updates in the parameters is either both +ve or -ve.

## tanh
* The range of tanh is between -1 to 1, hence not zero-centered. The advante is that the negative inputs will be maped strongly negative and the zero inputs will be mapped near zero in the tanh graph. The function is differentiable.
* It is mostly used in NLP.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im35.png) <br/>
* ***Drawbacks***
    * It still suffers from vanishing gradient problem.
    * Computation heavy.

## RELU
* It is the most used activation function currently.
* It is mainly used in CNN.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im36.png) <br/>
* As you can see, the ReLU is half rectified (from bottom). f(z) is zero when z is less than zero and f(z) is equal to z when z is above or equal to zero.

* ***Drawbacks***
* If the input value(z) is -ve, then neuron is die and once a relu neuron die it will be always 0.

## Leaky RELU
* It is an attempt to solve the dying ReLU problem
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im37.png) <br/>
* The leak helps to increase the range of the ReLU function. Usually, the value of a is 0.01 or so.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im38.png) <br/>
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im39.png) <br/>

* References
    * https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
    * https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

# Data drifting
* Machine learning creates static models from the historical data. But, once deployed in production, ML models become unreliable and obsolete and degrade with time. There might be changes in the data distribution in production, thus causing biased predictions. User behavior itself might have changed compared to the baseline data the model was trained on, or there might be additional factors in real-world interactions which would have impacted the predictions. Data drift is a major reason model accuracy decreases over time.
* Types of data drift: Let???s call the inputs to a model X and its outputs Y. We know that in supervised learning, the training data can be viewed as a set of samples from the joint distribution P(X, Y) and then ML usually models P(Y|X). This joint distribution P(X, Y) can be decomposed in two ways:
P(X, Y) = P(Y|X)P(X)
P(X, Y) = P(X|Y)P(Y)
P(Y|X) denotes the conditional probability of an output given an input ??? for example, the probability of an email being spam given the content of the email. P(X) denotes the probability density of the input. P(Y) denotes the probability density of the output. Label shift, covariate shift, and concept drift are defined as follows.
    * ***Covariate shift*** is when P(X) changes, but P(Y|X) remains the same. This refers to the first decomposition of the joint distribution.
    * ***Label shift*** is when P(Y) changes, but P(X|Y) remains the same. This refers to the second decomposition of the joint distribution.
    * ***Concept drift*** is when P(Y|X) changes, but P(X) remains the same. This refers to the first decomposition of the joint distribution.
* References
    * https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html
    * https://www.explorium.ai/blog/understanding-and-handling-data-and-concept-drift/
    * https://www.meesho.io/blog/what-is-data-drift
    * https://www.analyticsvidhya.com/blog/2021/10/mlops-and-the-importance-of-data-drift-detection/

# Data Leakage
It can be defined in different ways:
* Data leakage is when information from outside the training dataset is used to create the model. This additional information can allow the model to learn or know something that it otherwise would not know and in turn invalidate the estimated performance of the mode being constructed.
* If any other feature whose value would not actually be available in practice at the time you???d want to use the model to make a prediction, is a feature that can introduce leakage to your model.
* When the data you are using to train a machine learning algorithm happens to have the information you are trying to predict
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im40.png) <br/>
***how to detect it***
* In general, if we see that the model which we build is too good to be true (i.,e gives predicted and actual output the same), then we should get suspicious and data leakage cannot be ruled out. At that time, the model might be somehow memorizing the relations between feature and target instead of learning and generalizing it for the unseen data. So, it is advised that before the testing, the prior documented results are weighed against the expected results.
* While doing the Exploratory Data Analysis (EDA), we may detect features that are very highly correlated with the target variable. Of course, some features are more correlated than others but a surprisingly high correlation needs to be checked and handled carefully. We should pay close attention to those features. So, with the help of EDA, we can examine the raw data through statistical and visualization tools.
* After the completion of the model training, if features are having very high weights, then we should pay close attention. Those features might be leaky.
* References
    * https://www.analyticsvidhya.com/blog/2021/07/data-leakage-and-its-effect-on-the-performance-of-an-ml-model/
    * https://machinelearningmastery.com/data-leakage-machine-learning/
    * https://towardsdatascience.com/data-leakage-in-machine-learning-how-it-can-be-detected-and-minimize-the-risk-8ef4e3a97562

# Handle Outliers
* An outlier is an observation that lies an abnormal distance from other values in a random sample from a population. There is, of course, a degree of ambiguity. Qualifying a data point as an anomaly leaves it up to the analyst or model to determine what is abnormal???and what to do with such data points.
* ***Causes of outliers***
    * Data entry errors
    * Measurement errors or instrument errors
    * Sampling errors
    * Data processing error
    * Natural novelties in data
* ***Methods for Detecting outliers***
    * ***Z-score***
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im41.png) <br/>
    define a threshold value of 3 and mark the datapoints whose absolute value of Z-score is greater than the threshold as outliers.

    * ***Detecting outliers using the Inter Quantile Range(IQR)***  
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im42.png) <br/>
    data points that lie 1.5 times of IQR above Q3 and below Q1 are outliers. 

    * ***Box plot***
    Some of the dots on the upper end are a bit further away. You can consider them outliers. It is not giving you the exact points that are outliers but it shows that there are outliers in this column of data.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im43.png) <br/>


* ***Dealing with outliers***
    * Deleting the values: You can delete the outliers if you know that the outliers are wrong or if the reason the outlier was created is never going to happen in the future. For example, there is a data set of peoples ages and the usual ages lie between 0 to 90 but there is data entry off the age 150 which is nearly impossible. So, we can safely drop the value that is 150.
    * Data transformation: Data transformation is useful when we are dealing with highly skewed data sets. By transforming the variables, we can eliminate the outliers for example taking the natural log of a value reduces the variation caused by the extreme values. This can also be done for data sets that do not have negative values.
    * Using different analysis methods: You could also use different statistical tests that are not as much impacted by the presence of outliers ??? for example using median to compare data sets as opposed to mean or use of equivalent nonparametric tests etc.
    * Valuing the outliers: In case there is a valid reason for the outlier to exist and it is a part of our natural process, we should investigate the cause of the outlier as it can provide valuable clues that can help you better understand your process performance. Outliers may be hiding precious information that could be invaluable to improve your process performance. You need to take the time to understand the special causes that contributed to these outliers. Fixing these special causes can give you significant boost in your process performance and improve customer satisfaction. For example, normal delivery of orders takes 1-2 days, but a few orders took more than a month to complete. Understanding the reason why it took a month and fixing this process can help future customers as they would not be impacted by such large wait times.
    * Mean/Median imputation: As the mean value is highly influenced by the outliers, it is advised to replace the outliers with the median value.
    * Quantile based flooring and capping: In this technique, the outlier is capped at a certain value above the 90th percentile value or floored at a factor below the 10th percentile value.
    * Trimming/Remove the outliers: we remove the outliers from the dataset. Although it is not a good practice to follow.
* References:
    * https://www.sigmamagic.com/blogs/how-to-handle-outliers/
    * https://cxl.com/blog/outliers/
    * https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/
    * https://statisticsbyjim.com/basics/remove-outliers/
    * https://regenerativetoday.com/a-complete-guide-for-detecting-and-dealing-with-outliers/


# Data Augmentation
* Data augmentation is the technique of increasing the size of data used for training a model. For reliable predictions, the deep learning models often require a lot of training data, which is not always available. Therefore, the existing data is augmented in order to make a better generalized model.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im44.png) <br/>
* Data augmention can be used in Computer vision or Natural language processing.
* ***Data augmentation techinques in Computer vision***
    * For data augmentation, making simple alterations on visual data is popular. In addition, generative adversarial networks (GANs) are used to create new synthetic data. Classic image processing activities for data augmentation are:
        * padding
        * random rotating
        * re-scaling
        * vertical and horizontal flipping
        * translation ( image is moved along X, Y direction)
        * cropping
        * zooming
        * darkening & brightening/color modification
        * grayscaling
        * changing contrast
        * adding noise
        * random erasing

* ***Data augmentation techinques in NLP***
    * Data augmentation is not as popular in the NLP domain as in the computer vision domain. Augmenting text data is difficult, due to the complexity of a language. Common methods for data augmentation in NLP are:
        * EDA methods such as Synonym replacement, Text Substitution (rule-based, ML-based, mask-based and etc.), Random insertion, Random swap, Random deletion and Word & sentence shuffling.
        * Back Translation: A sentence is translated in one language and then new sentence is translated again in the original language. So, different sentences are created.
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im45.png) <br/>
        * Text Generation: A generative adversarial networks (GAN) is trained to generate text with a few words.
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im46.png) <br/>
        Developers can optimize natural language models by training them on web data which contains large volumes of human speech, languages, syntaxes, and sentiments.
    * Libraries for text augmentation
        * Textattack
        * Nlpaug
        * Googletrans
        * NLP Albumentations
        * TextAugment
* References:
    * https://research.aimultiple.com/data-augmentation-techniques/
    * https://research.aimultiple.com/data-augmentation/
    * https://iq.opengenus.org/data-augmentation/
    * https://neptune.ai/blog/data-augmentation-nlp
    * https://www.analyticsvidhya.com/blog/2022/02/text-data-augmentation-in-natural-language-processing-with-texattack/

# K-cross validation
* Cross validation is an evaluation method used in machine learning to find out how well your machine learning model can predict the outcome of unseen data. It is a method that is easy to comprehend, works well for a limited data sample and also offers an evaluation that is less biased, making it a popular choice.
* The data sample is split into ???k??? number of smaller samples, hence the name: K-fold Cross Validation. In each set (fold) training and the test would be performed precisely once during this entire process. It helps us to avoid overfitting. To resist this k-fold cross-validation helps us to build the model is a generalized one.
* To achieve this K-Fold Cross Validation, we have to split the data set into three sets, Training, Testing, and Validation, with the challenge of the volume of the data.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im47.png) <br/>
* Let???s have a generalised K value. If K=5, it means, in the given dataset and we are splitting into 5 folds and running the Train and Test. During each run, one fold is considered for testing and the rest will be for training and moving on with iterations, the below pictorial representation would give you an idea of the flow of the fold-defined size.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im48.png) <br/>


![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im49.png) <br/>
In the above set, 5- Testing 20 Training. In each iteration, we will get an accuracy score and have to sum them and find the mean. Here we can understand how the data is spread in a way of consistency and will make a conclusion whether to for the production with this model (or) NOT.
* References:
    * https://www.kdnuggets.com/2022/07/kfold-cross-validation.html
    * https://www.analyticsvidhya.com/blog/2022/02/k-fold-cross-validation-technique-and-its-essentials/
    * https://machinelearningmastery.com/k-fold-cross-validation/
    * https://towardsdatascience.com/what-is-k-fold-cross-validation-5a7bb241d82f

# Sampling of data
* When we train a model, we don???t have access to all the corresponding possible data in the world; the data to be used is a subset created by some sampling methods. Data is full of potential biases, possibly due to some human factors during collecting, sampling, or labeling of the data. Since the biases are perpetuated in the models trained on this data, sampling appropriate subsets that can reduce the biases in the data is crucial.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im51.png) <br/>
* During data sampling, in order to train reliable models, the sampled subsets should represent the real-world data to reduce the selection biases. Following are some of the important probabilistic data sampling methods in model training:
    * ***Simple Random Sampling***
        * All the samples in the population have the same chance of being sampled, thus their probabilities form a uniform distribution. For example, if you want to sample 5 out of a 10 population, the probability of every element being selected is 0.5.
        * The rare classes in the population might not be sampled in the selection. Suppose you want to sample 1% from your data, but a rare class appears only in 0.01% of the population: samples of this rare class might not be selected. In this condition, models trained with the sampled subsets might not know the existence of the rare class.
    * ***Stratified Sampling***
        * To avoid the drawbacks of simple random sampling, you can divide the population to several groups according to your requirements, for example the labels, and sample from each group separately. Each group is called a stratum and this method is called stratified sampling.
        * For example, to sample 1% of a population that has classes A and B, you can divide the population to two groups and sample 1% from the two groups, respectively. In this way, no matter how rare A or B is, the sampled subsets are ensured to contain both of the two classes.
        * However, a drawback of stratified sampling is that the population is not always dividable. For example, in a multi-label learning task in which each sample has multiple labels, it is challenging to divide the population according to different labels.
    * ***Weighted Sampling***
        * In weighted sampling, each sample is assigned a weight???the probability of being sampled. For example, for a population containing classes A and B, if you assign weights of 0.8 to class A and 0.2 to class B, the probabilities of being sampled for class A and B are 80% and 20%, respectively.
        * Weight sampling can leverage domain expertise, which is important for reducing sampling biases. For example, in training some online learning models, recent data is much more important than old data. Through assigning bigger weights to recent data and smaller weights to old data, you can train a model more reliably.
    * ***Reservoir Sampling***
        * It is used to deal with streaming data in online learning models, which is quite popular in products.   
        * Suppose the data is generated in a sequential streaming manner, for example, a time series, and you can???t fit all the data to the memory, nor do you know how much data will be generated in the future. You need to sample a subset with k samples to train a model, but you don???t know which sample to select because many samples haven???t been generated yet.
        * Reservoir sampling can deal with this problem that 1) all the samples are selected with equal probability and 2) if you stop the algorithm at any time, the samples are always selected with correct probability. The algorithm contains 3 steps:
            1. Put the first k samples in a reservoir, which could be an array or a list
            2. When the nth sample is generated, randomly select a number m within the range of 1 to n. If the selected number m is within the range of 1 to k, replace the mth sample in the reservoir with the nth generated sample, otherwise do nothing.
            3. Repeat 2 until the stop of algorithm.
        * We can easily prove that for each newly generated sample, the probability of being selected to the reservoir is k/n. We can also prove that for each sample that is already in the reservoir, the probability of not being replaced is also k/n. Thus when the algorithm stops, all the samples in the reservoir are selected with correct probability.
    * ***Importance Sampling***
        *  It allows us to sample from a distribution when we only have access to another distribution.
        * For example, we want to sample from a distribution P(x), but can???t access it. However, we can access another distribution Q(x). The following equation shows that, in expectation, x sampled from P(x) equals to x sampled from Q(x) weighted by P(x)/Q(x).
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im50.png) <br/>
        Therefore, instead of sampling from P(x), we can alternatively sample from Q(x) which is accessible, and weight the sampled results by P(x)/Q(x). The results are the same as we directly sample from P(x).
    * ***Understanding data sampling errors***
        * When data sampling occurs, it requires those involved to make statistical conclusions about the population from a series of observations. Because these observations often come from estimations or generalizations, errors are bound to occur. The two main types of errors that occur when performing data sampling are:
            1. Selection bias: The bias that???s introduced by the selection of individuals to be part of the sample that isn???t random. Therefore, the sample cannot be representative of the population that is looking to be analyzed.
            2. Sampling error: The statistical error that occurs when the researcher doesn???t select a sample that represents the entire population of data. When this happens, the results found in the sample don???t represent the results that would have been obtained from the entire population.
        The only way to 100% eliminate the chance of a sampling error is to test 100% of the population. Of course, this is usually impossible. However, the larger the sample size in your data, the less extreme the margin of error will be.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im52.png) <br/>
    * References:
        * https://towardsdatascience.com/5-probabilistic-training-data-sampling-methods-in-machine-learning-460f2d6ffd9
        * https://medium.com/analytics-vidhya/sampling-statistical-approach-in-machine-learning-4903c40ebf86
        * https://medium.com/analytics-vidhya/sampling-statistical-approach-in-machine-learning-4903c40ebf86

# Feature Engineering(in predictive modelling)
## Handle Missing values
* ***Deleting rows with missing values***
    * Missing values can be handled by deleting the rows or columns having null values. If columns have more than half of the rows as null then the entire column can be dropped. The rows which are having one or more columns values as null can also be dropped.
    * A model trained with the removal of all missing values creates a robust model but it can results in loss of a lot of information.
* ***Impute missing values with Mean/Median***
    * Columns in the dataset which are having numeric continuous values can be replaced with the mean, median, or mode of remaining values in the column. This method can prevent the loss of data compared to the earlier method. Replacing the above two approximations (mean, median) is a statistical approach to handle the missing values.
    * It prevent data loss which results in deletion of rows or columns.
    * Works only with numerical continuous variables.
    * Can cause data leakage
* ***Imputation method for categorical columns***
    * When missing values is from categorical columns (string or numerical) then the missing values can be replaced with the most frequent category. If the number of missing values is very large then it can be replaced with a new category.
    * It prevent data loss which results in deletion of rows or columns.
    * Negates the loss of data by adding a unique category.
    * Addition of new features to the model while encoding, which may result in poor performance
* ***Other Imputation Methods***
    * Depending on the nature of the data or data type, some other imputation methods may be more appropriate to impute missing values. For example, for the data variable having longitudinal behavior, it might make sense to use the last valid observation to fill the missing value. This is known as the Last observation carried forward (LOCF) method.
    * For the time-series dataset variable, it makes sense to use the interpolation of the variable before and after a timestamp for a missing value.
* ***Using Algorithms that support missing values***
    * All the machine learning algorithms don???t support missing values but some ML algorithms are robust to missing values in the dataset. The k-NN algorithm can ignore a column from a distance measure when a value is missing. Naive Bayes can also support missing values when making a prediction. These algorithms can be used when the dataset contains null or missing values.
    * All tree-based algorithms(like XGBoost) automatically handles missing values.
* ***Prediction of missing values***
    * In the earlier methods to handle missing values, we do not use the correlation advantage of the variable containing the missing value and other variables. Using the other features which don???t have nulls can be used to predict missing values.
    * The regression or classification model can be used for the prediction of missing values depending on the nature (categorical or continuous) of the feature having missing value.
* ***Add separate column or feature(binary column: 1 for missing value otherwise 0) for missing value***
* ***Create a new category for NAN values(for cateogrical column)***
* References
    * https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e
    * https://www.analyticsvidhya.com/blog/2021/10/handling-missing-value/
    * https://medium.com/geekculture/how-to-deal-with-missing-values-in-machine-learning-98e47f025b9c
    * https://www.freecodecamp.org/news/how-to-handle-missing-data-in-a-dataset/
    * https://medium.com/analytics-vidhya/ways-to-handle-categorical-column-missing-data-its-implementations-15dc4a56893

## Handle numerical values(feature scaling or Normalization)
* Similarly, the goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. For machine learning, every dataset does not require normalization. It is required only when features have different ranges.
* For example, consider a data set containing two features, age, and income(x2). Where age ranges from 0???100, while income ranges from 0???100,000 and higher. Income is about 1,000 times larger than age. So, these two features are in very different ranges. When we do further analysis, like multivariate linear regression, for example, the attributed income will intrinsically influence the result more due to its larger value. But this doesn???t necessarily mean it is more important as a predictor. So we normalize the data to bring all the variables to the same range.
* There are different normalization techniques:
    * ***Standard Scalar or Z-score normalization***
        * It transforms each feature to a normal distribution with a mean of 0 and standard deviation of 1. May also be referred to as Z-score transformation.
        * Formula: x??? = (x ??? ??) / ?? , where ?? is the mean and ?? is the standard deviation
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im53.png) <br/>
    * ***MinMax Scalar***
        * linear transformation of data that maps the minimum value to maximum value to a certain range (e.g. 0 to 1)
        * x' = (x-x_min)/(x_max-x_min)
        * Scaling to a range is a good choice when both of the following conditions are met:
            * You know the approximate upper and lower bounds on your data with few or no outliers.
            * Your data is approximately uniformly distributed across that range.
        In contrast, you would not use scaling on features such as income, because only a few people have very high incomes. The upper bound of the linear scale for income would be very high, and most people would be squeezed into a small part of the scale.
    * ***Clipping***
        * If your data set contains extreme outliers, you might try feature clipping, which caps all feature values above (or below) a certain value to fixed value. For example, you could clip all temperature values above 40 to be exactly 40.
        * Formula: if x > max, then x??? = max else if x < min, then x??? = min
        * You may apply feature clipping before or after other normalization.
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im54.png) <br/>

    * ***Log Scaling***
        * Log scaling computes the log of your values to compress a wide range to a narrow range.
        * x???=log(x)
        * Log scaling is helpful when a handful of your values have many points, while most other values have few points. This data distribution is known as the power law distribution. Movie ratings are a good example. In the chart below, most movies have very few ratings (the data in the tail), while a few have lots of ratings (the data in the head). Log scaling changes the distribution, helping to improve linear model performance.
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im55.png) <br/>
    * ***Tips***
        * Min-Max scaling is sensitive to outliers. So, it is used when the feature is more-or-less uniformly distributed across a fixed range
        * Clipping is preferred to remove some outliers. It can be used in conjunction with other normalization techniques
        * Log scaling is preferred when there are extreme values. In short, when data distribution is skewed
        * Z-score or Standardization is useful when there are a few outliers, but not so extreme that you need clipping
    * References
        * https://medium.com/analytics-vidhya/data-transformation-for-numeric-features-fb16757382c0
        * https://towardsai.net/p/data-science/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff#:~:text=Normalization%3A,when%20features%20have%20different%20ranges.
        

## Handle categorical features
* Most machine learning algorithms cannot handle categorical variables unless we convert them to numerical values
* Many algorithm???s performances even vary based upon how the categorical variables are encoded
* A categorical or discrete variable is one that has two or more categories (values). There are two different types of categorical variables:
    * ***Nominal***: A nominal variable has no intrinsic ordering to its categories. For example, gender is a categorical variable having two categories (Male and Female) with no inherent ordering between them. Another example is Country (India, Australia, America, and so forth).
    * ***Ordinal***: An ordinal variable has a clear ordering within its categories. For example, consider temperature as a variable with three distinct (but related) categories (low, medium, high). Another example is an education degree (Ph.D., Master???s, or Bachelor???s).
* Different approaches to handle categorical data:
    * ***One Hot Encoding***
        * This technique is applied for nomial categorical features. In one Hot Encoding method, each category value is converted into a new column and assigned a value as 1 or 0 to the column.
        * This will be done using the pandas get_dummies() function and then we will drop the first column in order to avoid dummy variable trap.
        * Issue: A high cardinality of higher categories will increase the feature space, resulting in the curse of dimensionality.
    * ***Ordinal Number Encoding or Label Encoding***
        * It is used for used for ordinal categorical features. In this technique, each unique category value is given an integer value. For instance, ???red??? equals 1, ???green??? equals 2 and ???blue??? equals 3.
        * Domain information can be used to determine the integer value order. For example, we people love Saturday and Sundays, and most hates Monday. In this scenario the mapping for weekdays goes ???Monday??? is 1, ???Tuesday??? is 2, ???Wednesday??? is 3, ???Thursday??? is 4, ???Friday??? is 5,???Saturday??? is 6,???Sunday??? is 7.
    * ***Target Encoding***
        * A lesser known, but very effective way of handling categorical variables, is Target Encoding. It consists of substituting each group in a categorical feature with the average response in the target variable.
        ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im56.png) <br/>
        * The process to obtain the Target Encoding is relatively straightforward and it can be summarised as:
            1. Group the data by category
            2. Calculate the average of the target variable per each group
            3. Assign the average to each observation belonging to that group
        * Target Encoding is a powerful solution also because it avoids generating a high number of features, as is the case for One-Hot Encoding, keeping the dimensionality of the dataset as the original one.
    * ***Tips***:
        * For non-ordinal categories, Label Encoding, which consists of substituting a category with a relatively random integer, should be avoided at all costs.
        * Instead, One-Hot-Encoding and Target Encoding are preferable solutions. One-Hot Encoding is probably the most common solution, performing well in real-life scenarios. Target Encoding is a lesser-known but promising technique, which also keeps the dimensionality of the dataset consistent, improving performance.
    * References:
        * https://www.kdnuggets.com/2021/05/deal-with-categorical-data-machine-learning.html
        * https://medium.com/analytics-vidhya/how-to-handle-categorical-features-ab65c3cf498e#:~:text=1)%20Using%20the%20categorical%20variable,category%20with%20a%20probability%20ratio.
        * https://towardsdatascience.com/handling-categorical-data-the-right-way-9d1279956fc6

## Feature Hashing
* The Feature Hashing is used to transform a stream of English text into a set of integer features. You can then pass this hashed feature set to a machine learning algorithm to train a text analytics model.
* Feature hashing works by converting unique tokens into integers. It operates on the exact strings that you provide as input and does not perform any linguistic analysis or preprocessing.
* References
    * https://learn.microsoft.com/en-us/azure/machine-learning/component-reference/feature-hashing

# Distributed Training
## Data Parallelism
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im57.png) <br/>
* In data parallelism, the dataset is split into ???N??? parts, where ???N??? is the number of GPUs. These parts are then assigned to parallel computational machines. Post that, gradients are calculated for each copy of the model, after which all the models exchange the gradients. In the end, the values of these gradients are averaged. 
* For every GPU or node, the same parameters are used for the forward propagation. A small batch of data is sent to every node, and the gradient is computed normally and sent back to the main node. There are two strategies using which distributed training is practised called synchronous and asynchronous. 
    1. Synchronous training
        * As a part of sync training, the model sends different parts of the data into each accelerator. Every model has a complete copy of the model and is trained solely on a part of the data. Every single part starts forward pass simultaneously and computes a different output and gradient. 
        * Synchronous training uses an all-reduce algorithm which collects all the trainable parameters from various workers and accelerators. 
    2. Asynchronous training
        * While synchronous training can be advantageous in many ways, it is harder to scale and can result in workers staying idle at times. In asynchronous, workers don???t have to wait on each other during downtime in maintenance or a difference in capacity or priorities. Especially if devices are smaller, less reliable and more limited, asynchronous training may be a better choice. If the devices are stronger and with a powerful connection, synchronous training is more suited. 
## Model parallelism
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im58.png) <br/>
* In model parallelism, every model is partitioned into ???N??? parts, just like data parallelism, where ???N??? is the number of GPUs. Each model is then placed on an individual GPU. The batch of GPUs is then calculated sequentially in this manner, starting with GPU#0, GPU#1 and continuing until GPU#N. This is forward propagation. Backward propagation on the other end begins with the reverse, GPU#N and ends at GPU#0. 
* Model parallelism has some obvious benefits. It can be used to train a model such that it does not fit into just a single GPU. But when computing is moving in a sequential fashion, for example, when GPU#1 is in computation, the others simply lie idle. This can be resolved by shifting to an asynchronous style of GPU functioning. 
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im59.png) <br/>
There are multiple mini-batches in progress in the pipeline; first the initial mini-batches update weights, the mini-batches next in the pipeline adopt stale weights to derive gradients. In model parallelism, staleness leads to instability and low model accuracy. A study titled ???Efficient and Robust Parallel DNN Training through Model Parallelism as Multi-GPU Platform??? that tested model parallelism against data parallelism showed that models using data parallelism increase in their accuracy as training proceeds, but the accuracy starts fluctuating with model parallelism. 
* The study also demonstrated that data parallelism is expected to have a scalability issue. Model parallelism seemed more apt for DNN models as a bigger number of GPUs was added. 
In a recent and prominent instance, Google AI???s large language model PaLM or Pathways Language Model used a combination of data and model parallelism as a part of its state-of-the-art training. The model was scaled using data parallelism at the Pod level across two Cloud TPU v4 Pods while each Pod used model parallelism with standard data.
* References:
    * https://analyticsindiamag.com/data-parallelism-vs-model-parallelism-how-do-they-differ-in-distributed-training/
    * https://medium.com/@esaliya/model-parallelism-in-deep-learning-is-not-what-you-think-94d2f81e82ed

# AB tests
## p-value
* All statistical tests have a null hypothesis. For most tests, the null hypothesis is that there is no relationship between your variables of interest or that there is no difference among groups. While the alternative hypothesis means there is a difference between groups.
* The null hypothesis states that there is no relationship between the two variables being studied (one variable does not affect the other). It states the results are due to chance and are not significant in terms of supporting the idea being investigated. Thus, the null hypothesis assumes that whatever you are trying to prove did not happen.
* ***P-value***: A p-value measures the probability of obtaining the observed results, assuming that the null hypothesis is true.
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im60.png) <br/>
* The level of statistical significance is often expressed as a p-value between 0 and 1. The smaller the p-value, the stronger the evidence that you should reject the null hypothesis.
* A p-value less than 0.05 (typically ??? 0.05) is statistically significant. It indicates strong evidence against the null hypothesis, as there is less than a 5% probability the null is correct (and the results are random). Therefore, we reject the null hypothesis, and accept the alternative hypothesis.
* However, if the p-value is below your threshold of significance (typically p < 0.05), you can reject the null hypothesis, but this does not mean that there is a 95% probability that the alternative hypothesis is true. The p-value is conditional upon the null hypothesis being true, but is unrelated to the truth or falsity of the alternative hypothesis.
* A p-value higher than 0.05 (> 0.05) is not statistically significant and indicates strong evidence for the null hypothesis. This means we retain the null hypothesis and reject the alternative hypothesis. You should note that you cannot accept the null hypothesis, we can only reject the null or fail to reject it.
* References
    * https://www.simplypsychology.org/p-value.html
    * https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/p-value/
    * https://www.youtube.com/watch?v=-FtlH4svqx4
    * https://www.youtube.com/watch?v=KS6KEWaoOOE
## How to decide sample size(when to stop experiments?)
* If you run experiments: the best way to avoid repeated significance testing errors is to not test significance repeatedly. Decide on a sample size in advance and wait until the experiment is over before you start believing the ???chance of beating original??? figures that the A/B testing software gives you. ???Peeking??? at the data is OK as long as you can restrain yourself from stopping an experiment before it has run its course. I know this goes against something in human nature, so perhaps the best advice is: no peeking!
* Since you are going to fix the sample size in advance, what sample size should you use? This formula is a good rule of thumb:
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/snip1.PNG) <br/>
Where ?? is the minimum effect you wish to detect and ??2 is the sample variance you expect. Of course you might not know the variance, but if it???s just a binomial proportion you???re calculating (e.g. a percent conversion rate) the variance is given by:
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/snip2.png) <br/>
* References
    * https://www.evanmiller.org/how-not-to-run-an-ab-test.html
    * https://medium.com/swlh/the-ultimate-guide-to-a-b-testing-part-1-experiment-design-8315a2470c63
Committing to a sample size completely mitigates the problem described here.
# Knowledge distillation
* Knowledge distillation refers to the process of transferring the knowledge from a large unwieldy model or set of models to a single smaller model that can be practically deployed under real-world constraints.
* Knowledge distillation is performed more commonly on neural network models associated with complex architectures including several layers and model parameters.
* The challenge of deploying large deep neural network models is especially pertinent for edge devices with limited memory and computational capacity. To tackle this challenge, a model compression method was first proposed to transfer the knowledge from a large model into training a smaller model without any significant loss in performance. This process of learning a small model from a larger model was formalized as a ???Knowledge Distillation???
![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im61.png) <br/>
As shown in Figure, in knowledge distillation, a small ???student??? model learns to mimic a large ???teacher??? model and leverage the knowledge of the teacher to obtain similar or higher accuracy.
* ***Major parts of the technique***
    * ***Teacher Model*** : An ensemble of separately trained models or a single very large model trained with a very strong regularizer such as dropout can be used to create a larger cumbersome model. The cumbersome model is the first to be trained.
    * ***Student model***: A smaller model that will rely on the Teacher Network???s distilled knowledge. It employs a different type of training called ???distillation??? to transfer knowledge from the large model to the smaller Student model. The student model is more suitable for deployment because it will be computationally less expensive than the Teacher model while maintaining the same or better accuracy.
* References
    * https://neptune.ai/blog/knowledge-distillation
    * https://analyticsindiamag.com/a-beginners-guide-to-knowledge-distillation-in-deep-learning/
