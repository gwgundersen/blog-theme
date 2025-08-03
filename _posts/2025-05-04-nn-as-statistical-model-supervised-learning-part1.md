---
title: The conditional probability distribution induced by a given network and loss in supervised learning
subtitle: Any neural network and a loss function in a supervised learning settings induces a conditional probability density. Explicitly working with the induced conditional probability is beneficial.
layout: default
date: 2025-08-02
keywords: supervised learning, neural networks, loss functions, conditional probability, parameteric models, information theory, kullback-leibler divergence.
published: true
---

{% katexmm %}
An Artificial neural network can be described as a function $f(x)$ over some input set $\mathcal{X}$ with values in some output set $\mathcal{Y}$. In supervised learning, $\mathcal{X}$ is different from $\mathcal{Y}$ while in unsupervised learning they are the same. The activity of the network is carried out by many layers of activation units linked together using weights {% cite goodfellow:deep-learning:2016 %}. Denote the collection of all the weights of the network by the symbol $w$. In large models, $w$ is a very large tuple. $w$ is an element in $W$, the set of all possible weight values the network can take at any one time. To reflect that for different values of $w\in \mathcal{W}$ we have different network realizations $f(x)$, we will adopt the notation $f(x|w)$.

Any neural network also defines a conditional probability density or mass function $p(y|x,w)$. To see that, suppose a loss function $\mathscr{L}(y,f(x|w))$ was given. We can define the following probability density (or mass) function

$$ \tag{1}
p(y|x,w) = \frac{e^{-\mathscr{L}(y,f(x|w))}}{Z}
$$

where $Z$ is a normalizing constant such that $e^{-\mathscr{L}(y,f(x|w))}$ integrates to 1 for any network realization $f(x|w)$. 

Equation (1) above is the well known Gibbs density where $\mathscr{L}$ is playing the role of the energy of the configuration $y$. For this to be a well defined probability density (or mass) function, we need the normalizing constant $Z$ to be finite for all $w\in \mathcal{W}$. This is indeed the case by the usual assumptions that $\mathscr{L}$ is integrable and $\mathscr{L}(y,f(x|w)) \geq 0$ for arbitrary $y, x, w$. 

In this post, I show in detail how a neural network and an associated loss function induces a conditional probability distribution and briefly discuss the benefits of this view. In future posts, I extend this probabilistic view to other elements of supervised learning and show in details how this view is important to understanding deep learning.

In supervised learning, one is given a data set $\{(x_1, y_1),\cdots, (x_n, y_n)\}$ and our objective is to construct a neural network one can use to predict future unseen value $y_{n+1}$ given future seen or unseen value $x_{n+1}$ {% cite friedman:an-overview-of-predictive-learning-and-fn-approx:1994 %}. Based on the nature of the observables $x, y$ one constructs an appropriate neural network and chooses a loss function $\mathscr{L}$ that is deemed appropriate. In the following examples we drive the conditional probability densities (or mass) functions associated with a given network and loss function.

#### Example 1: Squared error loss
When $y$ is on a continuous scale (i.e. stock price, air temperature ...etc), a typical loss is any function $\mathscr{L}$ such that:
1. $\mathscr{L}(y, f(x|w)) \geq 0$ for all $y, f(x|w)\in\mathcal{Y}$, with equality if $y=f(x|w)$.
2. $\mathscr{L}$ is an integrable function (w.r.t the unknown distribution that generated the data). This is needed to guarantee that the expected loss (which is the limit of the training loss as $n\rightarrow \infty$) is finite; otherwise, one cannot speak of minimizing the expected loss.

While condition 2 cannot be verified in practice because the distribution that generated the data is unknown, typical losses like *squared error loss*, *absolute error loss*, and *Huber loss* satisfy these properties under reasonable distributional assumptions.

If $\mathcal{Y}$ is a subset of $\mathbb{R}^k$, we could use the squared euclidean norm $|\cdot|^2_2$ that $\mathbb{R}^k$ is often equipped with as a loss function.

$$
\begin{aligned}
\mathscr{L}_{\text{SE}}(y, f(x|w)) &\triangleq |y-f(x|w)|_2^2\\
    &= \sum_{i=1}^k(y_i-f(x|w)_i)^2
\end{aligned}
$$
with $y=(y_1, \cdots, y_k)$, and $f(x|w)_i$ is the $i$-th entry of $f(x|w)$. The neural network $f(\cdot|w)$ and the loss $\mathscr{L}_{\text{SE}}(y,f(x|w))$ induces the parameteric conditional probability density $\frac{1}{Z}e^{-|y-f(x|w)|_2^2}$ which one can immediately recognize as the $k$ dimensional normal distribution 
$$
p(y|x,w) = \frac{1}{\sqrt{2\pi}|\Sigma|^{k/2}}e^{\frac{-1}{2}(y-f(x|w))^T\Sigma^{-1} (y-f(x|w))}
$$
with mean $f(x|w)$ and variance $\Sigma=2\mathbb{I}_{k\times k}$ where $(y-f(x|w))^T$ is the transpose of the column vector $(y-f(x|w))$, and $|\Epsilon|$ is the determinant of $\Epsilon$.

#### Example 2: Absolute error loss
Another loss function used in practice is the $L_1$ norm.

$$
\begin{aligned}
\mathscr{L}_{\text{AE}}(y, f(x|w)) &\triangleq L_1(y, f(x|w))\\
    &=\sum_{i=1}^k|y_i-f(x|w)_i|
\end{aligned}
$$

The induced conditional probability density is

$$
\begin{aligned}
p(y|x,w) &= \frac{1}{Z}e^{-\sum_{i=1}^k|y_i-f(x|w)_i|}\\
    &= \frac{1}{Z}\prod_{i=1}^k e^{-|y_i-f(x|w)_i|}
\end{aligned}
$$
which is the product of $k$ independent laplace probability densities with parameter $1$.

#### Example 3: Cross entropy loss
When $y$ is on a categorical scale (i.e. dog vs cat vs bird, happy vs sad, a number in the set $\{0, \cdots, 9\}$), one typically uses a network with number of output units matching the cardinality of $\mathcal{Y}$ and the cross entropy loss

$$
\mathscr{L}_{\text{CE}}(y, f(x|w)) = -\sum_{i\in \mathcal{Y}} \big\{\delta[y=i]\log{\text{softmax}[f(x|w)]}_i\big\}
$$
where $\delta[y=i]$ is the dirac delta, and $\text{softmax}(f(x|w))_i$ is the $i$-th component of $\text{softmax}(f(x|w))$. The associated conditional probability mass function is in fact explicit

$$
P(y=i|x,w) = \text{softmax}[f(x|w)]_i \quad i\in\mathcal{Y}
$$
When the cardinality of $\mathcal{Y}$ is $2$, the cross-entropy reduces to the binary cross-entropy.

Now that we have a good handle on the conditional distribution induced by our choice of the loss function and the nature of the output layer of the network, one can apply the tools of frequentist statistics such as maximum likelihood, hypothesis testing, and asymptotic theory for analyzing supervised learning methods. For instance, under the assumption that the probability of any pair $(x_i,y_i)$ is independent, we can write the log likelihood of the dataset (under our model) for different parameters $w$ as 

$$\tag{2}
\log{\prod_{i=1}^n p(y_i|x_i,w)} = \sum_{i=1}^n \log{p(y_i|x_i,w)}
$$

yielding the following maximum likelihood parameter everyone is familiar with from 2nd year undergraduate statistics.

$$
\hat{w}_{\text{mle}} = \underset{w\in W}{\text{argmax}}\bigg\{\sum_{i=1}^n \log{p(y_i|x_i,w)}\bigg\}
$$
Maximizing this likelihood equation (2) is equivalent to minimizing the empirical loss since $\log{p(y|x,w)}=-\mathscr{L}(y,f(x|w))+\text{log(Z)}$. As a result, one might argue that we did not gain much by characterizing the conditional probability density (or mass) associated with a given loss and network. 

This is not exactly true. First, making the nature of the assumed noise in the model explicit provide us with more information about the nature of our model and ways to change it. For instance, in the probabilistic characterization one could choose a non-diagnoal matrix to represent known correlations in our noise. Even more, one could compute the observed errors and check if they indeed conform with the assumed noise distribution (a statistical technique for measuring model fit).

Second, one can apply tools of information theory to formally characterize what it means to be surprised when using our model to stand in for the unknown distribution $q(y|x)$ that generated the data. Many notions of surprise are available in practice. If we subscribe to the infamous notion of Shannon surprise, then the average surprise when using the model $p(y|x, w_{\text{mle}})$ instead of the true unknown distribution that generates the data is defined by the cross entropy

$$\tag{2}
H(q,p) \triangleq -\int_{\mathcal{X}\times\mathcal{Y}} q(y|x)\log{p(y|x,w_{\text{mle}})}\,dy\,dx
$$

The minimum average (Shannon) surprise when observing samples from $q(y|x)$ is 
$$
H(q) \triangleq -\int_{\mathcal{X}\times\mathcal{Y}}q(y|x)\log{q(y|x)}\,dx\,dy,
$$
the entropy of $q(y|x)$. As a result, the average excess surprise when using our model $p(y|x,w)$ instead of the true distribution $q(y|x)$ is

$$
\begin{aligned}\tag{3}
D_{\text{KL}}(q,p) &\triangleq \int_{\textstyle\mathcal{X}\times\mathcal{Y}} q(y|x)\frac{\log{q(y|x)}}{\log{p(y|x,w_{\text{mle}})}}\,dy\,dx\\
    &= H(q,p) - H(q)
\end{aligned}
$$
which is the Kullback-Leibler divergence {% cite kullback:on-info-and-suff:1951 %}. This is one important reason why machine learning minimizes the cross entropy $H(q,p)$ (which is equivalent to minimizing the KL divergence from $q$ to $p$) since if our model is any good it should stand in for $q$ when making decisions related to our observables $x$ and $y$.

Last but not least, by understanding the conditional probabilistic distribution induced by a network and loss, one can use laws that are unique to probability theory such as conditional expectation, and bayes' rule to study complex models and building powerful learning machines.

{% endkatexmm %}