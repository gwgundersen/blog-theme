---
title: Neural networks of supervised learning as bayesian statistical models
subtitle: Here is some extra detail about the post.
layout: default
date: 2025-4-5
keywords: statistical models, information theory, artificial neural networks, differential geometry, statistical inference 
published: false
---

{% katexmm %}
An Artificial neural network can be described as a function $f(x)$ over some input set $\mathcal{X}$ with values in some output set $\mathcal{Y}$. In supervised learning, $\mathcal{X}$ is different from $\mathcal{Y}$ while in unsupervised learning they are the same.

The activity of the network is carried out by many layers of activation units linked together using weights. Denote the collection of all the weights of the network by the symbol $w$. In large models, $w$ is a very large tuple. $w$ is an element in $W$, the set of all possible weight values the network can take at any one time. To reflect that for different values of $w\in \mathcal{W}$ we have different network realizations $f(x)$, we will adopt the notation $f(x|w)$.

Any neural network also defines a conditional probability density or mass function $p(y|x,w)$. To see that, suppose a loss function $\text{Loss}(y,f(x|w))$ was given. We can define the following probability density (or mass) function


\begin{equation} \label{eq:gibbs-density-representation}\tag{1}
p(y|x,w) = \frac{e^{-\text{Loss}(y,f(x|w))}}{Z}
\end{equation}

where $Z$ is a normalizing constant such that $e^{-\text{Loss}(y,f(x|w))}$ integrates to 1 for any network realization $f(x|w)$. Equation \eqref{eq:gibbs-density-representation} above is the well known Gibbs density where $\text{Loss}$ is playing the role of the energy of the configuration $y$.

For this to be a well defined probability density, we need $Z$ to be finite for all $w\in \mathcal{W}$. This is indeed the case by the usual assumptions that $\text{Loss}$ is integrable and $\text{Loss}(y,f(x|w)) \geq 0$ for arbitrary $y, x, w$. 

In this post, I show in detail how a neural network and an associated loss function induces a conditional probability distribution and briefly discuss the benefits of this view. In future posts, I extend this probabilistic to other elements of supervised learning and show in details how neural networks in supervised learning are bayesian statistical models.

In supervised learning, one is given a data set $\{(x_1, y_1),\cdots, (x_n, y_n)\}$ and our objective is to construct a neural network one can use to predict future unseen value $y_{n+1}$ given future seen or unseen value $x_{n+1}$. Based on the nature of the observables $x, y$ one constructs an appropriate neural network and chooses a loss function $\text{Loss}$ that is deemed appropriate. In the following examples we drive the conditional probability densities associated with a given network and loss function.

### Example 1.1: Squared error loss
When $y$ is on a continuous scale (i.e. stock price, air temperature ...etc), a typical loss is any function $\text{Loss}$ such that:
1. $\text{Loss}(y, f(x|w)) \geq 0$ for all $y, \hat{y}=f(x|w)\in\mathcal{Y}$, with equality if $y=\hat{y}$.
2. $\text{Loss}$ is an integrable function (w.r.t the unknown distribution that generated the data). This is needed to guarantee that the expected loss (which is the limit of the training loss) is finite; otherwise, one cannot speak of minimizing L.

While condition 2 cannot be verified in practice because the distribution that generated the data is uknown, typical losses like Squared Error Loss, Absolute Error Loss, and Huber loss satisfy these properties under reasonable distributional assumptions.

If $\mathcal{Y}$ is a subset of $\mathbb{R}^k$, we could use the squared euclidean norm $|\cdot|^2_2$ that $\mathbb{R}^k$ is often equipped with as a loss function.

$$
\begin{aligned}
\text{Loss}_{\text{SE}}(y, f(x|w)) &\triangleq |y-f(x|w)|_2^2\\
    &= \sum_{i=1}^k(y_i-f(x|w)_i)^2
\end{aligned}
$$
with $y=(y_1, \cdots, y_k)$, and $f(x|w)_i$ is the $i$-th entry of $f(x|w)$. The neural network $f(\cdot|w)$ and the loss $\text{Loss}_{\text{SE}}(y,f(x|w))$ induces the parameteric conditional probability density $\frac{1}{Z}e^{-|y-f(x|w)|_2^2}$ which one can immediately recognize as the $k$ dimensional normal distribution 
$$
p(y|x,w) = \frac{1}{\sqrt{2\pi}|\Sigma|^{k/2}}e^{\frac{-1}{2}(y-f(x|w))^T\Sigma^{-1} (y-f(x|w))}
$$
with mean $f(x|w)$ and variance $\Sigma=2\mathbb{I}_{k\times k}$ where $(y-f(x|w))^T$ is the transpose of the column vector $(y-f(x|w))$, and $|\Epsilon|$ is the determinant of $\Epsilon$.

### Example 1.2: Absolute error loss
Another loss function used in practice is the $L_1$ norm.

$$
\begin{aligned}
\text{Loss}_{\text{AE}}(y, f(x|w)) &\triangleq L_1(y, f(x|w))\\
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

### Example 1.3: Cross entropy loss
When $y$ is on a categorical scale (i.e. dog vs cat vs bird, happy vs sad, a number in the set $\{0, \cdots, 9\}$), one typically uses a network with number of output units matching the cardinality of $\mathcal{Y}$ and the cross entropy loss

$$
\text{Loss}_{\text{CE}}(y, f(x|w)) = -\sum_{i\in \mathcal{Y}} \big\{\delta[y=i]\text{log}(\text{softmax}[f(x|w)])_i\big\}
$$
where $\delta[y=i]$ is the dirac delta, and $(\text{softmax}(f(x|w)))_i$ is the $i$-th component of $\text{softmax}(f(x|w))$. The associated conditional probability mass function is in fact explicit

$$
p(y=i|x,w) = \text{softmax}(f(x|w))^i \quad i\in\mathcal{Y}
$$
When the cardinality of $\mathcal{Y}$ is $2$, the cross-entropy reduces to the binary cross-entropy.

Now that we have a good handle on the conditional distribution induced by our choice of the loss function and the nature of the output layer of the network, one can apply the tools of frequentist statistics such as maximum likelihood, hypothesis testing, and asymptotic theory for analyzing supervised learning methods. For instance, under the assumption that the probability of any pair $(x_i,y_i)$ is independent, we can write the log empirical likelihood of the dataset given our model parameter $w$ as 

$$
\text{log}\prod_{i=1}^n p(y_i|x_i,w) = \sum_{i=1}^n \text{log}p(y_i|x_i,w)
$$

yielding the following log empirical maximum likelihood equation everyone is familiar with from 2nd year undergraduate statistics.

$$
\hat{w}_{\text{mle}} = \underset{w\in W}{\text{argmax}}\big\{\sum_{i=1}^n \text{log}p(y_i|x_i,w)\big\}
$$

Maximizing this likelihood equation is equivalent to minimizing the average negative empirical log likelihood

$$
\underset{w\in W}{\text{argmin}}\big\{-\frac{1}{n}\sum_{i=1}^n \text{log}p(y_i|x_i,w)\big\}
$$

Observe that $\text{logp(y|x,w)}=-\text{Loss}(y,f(x|w))+\text{log(Z)}$. In other words, maximizing the empirical likelihood of the data set is equivalent to minimizing $\frac{1}{n}\sum_{i=1}^n\text{Loss}(y_i, f(x_i|w))$.

One might argue that since the minimization of the empirical loss is equivalent to maximization of the empirical likelihood, we didn't gain much by characterizing the conditional probability density (or mass) associated with a given loss and network. This is not exactly true. First, making the nature of the assumed noise in the model explicit provide us with more information about the nature of our model and ways to change it. For instance, in the probabilistic characterization one could choose a non-diagnoal matrix to represent known correlations in our noise. Even more, one could compute the observed errors and check if they indeed conform with the assumed noise distribution (a statistical technique for quantifying the nature of our model).

Second, one can apply tools of information theory to formally characterize what it means to be surprised when using our model to stand in for the unknown distribution that generate the data $q(y|x)$. Many notion of surprise is available in practice. If we subscriber to the infamous notion of Shannon surprise, then the average surprise when using the model $p(y|x, w_{\text{mle}})$ instead of the true unknown distribution that generates the data $q(y|x)$ is defined by the cross entropy

\begin{equation} \label{eq:cross-entropy}\tag{2}
H(q,p) \triangleq \int_{\mathcal{X}\times\mathcal{Y}} q(y|x)\text{log}p(y|x,w_{\text{mle}})dydx
\end{equation}

The minimum average (Shannon) surprise when observing samples from $q(y|x)$ is $\int_{\mathcal{X}\times\mathcal{Y}}q(y|x)\text{log}q(y|x)dxdy$, the entropy of $q(y|x)$. As a result, the excess average surprise when using our model $p(y|x,w)$ instead of the true distribution $q(y|x)$ is defined as

\begin{equation} \label{eq:average-kl}\tag{3}
\begin{aligned}
D_{\text{KL}}(q,p) &\triangleq \int_{\mathcal{X}\times\mathcal{Y}} q(y|x)\frac{\text{log}q(y|x)}{\text{log}p(y|x,w_{\text{mle}})}dydx\\
    &= H(q,p) - H(q)
\end{aligned}
\end{equation}
where $H(q)$ is the entropy of $q(y|x)$.

This is one important reason why machine learning minimizes the cross entropy $H(q,p)$ (which is equivalent to minimizing the KL divergence from $q$ to $p$) since if our model is any good it should stand in for $q$ when making decisions related to our observables $x$ and $y$.

Last but not least, by understanding the conditional probabilisty distribution induced by a network and loss, one can use laws that are unique to probability theory such as conditional expectation, and bayes' rule to study complex models and build powerful learning machines.

## 3. Regularization as prior distributions
In the early days of machine learning, it was clarified that, even though the learning problem is ill-posed (learning about a function from point observations is always ill-posed), with regularization we can make the problem well-posed.

For neural networks to be characterized as bayesian statistical models, one needs to define a prior probability distribution over the set of parameters $W$. This is typically defined indirectly through regularization. Let's see some examples.

1. <b>L_2 regularization</b>: $L_2$ regularization enters supervised learning by enriching the loss function with a squared $L_2$ norm of the weights. For example 1 above, we can defined the regularized squared error loss

$$
L_{regularized}(y, f(x|w)) = |y-f(x|w)|_2^2 + \lambda |w|_2^2
$$
observe that

$$
\frac{1}{Z}e^{-L_{regularized}(y,f(x|w))^2} = p(y|x,w) \frac{1}{Z_{weights}}e^{-\lambda |w|_2^2}
$$
which we can immediately recognize as the product of the conditional density $p(y|x,w)$ with a $d$-dimensional normal distribution with mean $0$ and variance $\frac{1}{2\lambda}I_{d\times d}$

2. <b>L_1 regularization</b>: $L_1$ regularization is specified by considering the $L_1$ norm of $W$ instead of $L_1$. This can easily been seen to specifiy a laplace prior on $W$ with parameter $\frac{1}{\lambda}$

<h3>Networks of unsupervised learning</h3>
In unsupervised learning, one is given a data set $\{x_1, \cdots, x_n\}$, and we are asked to learn a hidden structure to characterize the variability in the data set such that one can characterize future unseen values. For instance, one might believe that there exists a latent structure to our observations such that, for every $x_i$ there exists a cluster $c$



Despite the above description being correct, it fails to directly acknowledge that neural networks are parameteric statistical models. Even more, many of the deep learning methods and challenges can easily be understood within a bayesian inference framework. We start this post by emphasizing the statistical model nature of neural networks while emphasizing learning within a bayesian inference framework. We finish by recalling a few important examples and develop the probabilistic view more concretly.

<h3>Neural networks are statistical models</h3>
A parameteric statistical model is a map $p:\mathcal{W}\rightarrow\mathcal{P}$ from a topological space of parameters $\mathcal{W}$, typically subset of some euclidean space $\mathbb{R}^d$, to a subset of all the probability densities $\mathcal{P}$ where each point $P\in\mathcal{P}$ is a probability density over the same sample space $\Omega$. 

For a given neural network $f(x|w)$ and 



This construction can seem odd at times. Let's see what happens when we use some of the well-known loss functions.



<!-- 

The above description of a statistical model is very similar to that of a neural network. They are both maps between a parameter space and a subset of some functional space; however, the statistical model prespective has a few important benefits:

3. It makes it 

Latent variable models are powerful statistical models where we are interested in learning about hidden variables that can explain the variability observed in the data. In these models, the sample space is a product space $\Omega_{\text{observed}}\times\Omega_{\text{hidden}}$.

It is reasonable to wonder why the statistical model view is more natural. After all, both neural networks and statisticals models are maps from a space of parameters to some subset of a space of functions. The statistical model view however empahsizes the fact that, our learning  -->

Let's see how this can be applied in a few classical examples.
</hr/>


Given a random sample $D^n\triangleq\{\omega_1,\cdots,\omega_n\}$, a statistical model can be used in a learning machine

$$
D^n\mapsto \hat{p}(\cdot|\theta)
$$
which is a map from the data $D^n$ to a probability distribution $\hat{p}(\cdot|\theta)\in\mathcal{P}$. Many learning machines are possible. The following deserve notable mentions:

1. The maximum likelihood machine 

$$
D^n\mapsto \underset{\theta\in\Theta}{\text{argmax}} \big(p(D^n|\theta)\big)
$$

2. The maximum a posteriori machine 

$$
D^n\mapsto p(\omega|\theta_{\text{posterior}}) = p(\omega|\underset{\theta\in\Theta}{\text{argmax}}\big(p(\theta|D^n)\big))
$$
where $p(\theta|D^n)=\frac{p(D^n|\theta)\varphi(\theta)}{p(D^n)}$ is the posterior distribution of the parameters given the data, $\varphi(\theta)$ is a prior distribution on $\Theta$, and $p(D^n)$ is the evidence under the given statistical model.

3. The predictive distribution machine 

$$
D^n\mapsto p(\omega_{n+1}|D^n)=\int_{\Omega}p(\omega_{n+1}|\theta)p(\theta|D^n)d\omega
$$

The above learning machines can be equivalent; however, in most powerful statistical models they are not. The situations under which they are not equivalent is of great importance in applied statistics and should also be the case in deep learning.



<h4>Example 1: classification of handwritten digits</h4> 
Consider the classical problem of recognizing hand written $0,\cdots,9$ digits from images recorded as $28\times 28$ grayscale pixel data (MNIST-10 dataset).
1. $\mathcal{X}$ are points in the measurement space $[0,255]^{28\times 28}$ pixel data. In this representation, every pixel is a point in the interval $[0,255]$ where $0, 255$ are arbitrarily mapped to black and white.
2. $\mathcal{Y} is the finite set $\{0, \cdots, 9\}$.

To solve this problem, one could use a neural network with a single hidden layer:
1. Layer 1 (input layer) contains $28\times 28$ activation nodes with output values in the interval $[0,255]$.
2. Layer 2 (hidden layer) contains $1000$ activation nodes with real output and a tanh activation function.
3. Layer 3 (output layer) contains $10$ nodes each taking a value in the unit interval.

To complete the description we consider a fully connected network with weights $w_{i,j}^l$ where $i$ is the index of the node in layer $l$, $j$ is the index of the node in layer $l+1$ and $l\in\{1,2\}$. We can compactly write the weights between layers as a matrix $W^l=(w_{i,j})^l$. For any given realization of weights $w\in \mathbb{R}^{256\times 1000}\times\mathbb{R}^{1000\times 10}$, the above network is described as a parameteric map $f(\cdot|w):\mathcal{X}\rightarrow\mathcal{Y}$.

The objective of learning can be described as minimizing the loss function
$$
\begin{aligned}
L(y_i,\hat{y}_i) = (y_i-\hat{y}_i)^2
\end{aligned}
$$
where $y_i$ is the known label for a given sample input image $x_i$, and $\hat{y}_i=\text{argmax}_{0,\cdots, 9} f(x_i|w)$, the index of the output with maximal output with ties broken arbitrarily.


<h4>Example 2: clustering of handwritten digits<h4>
Consider again the MNIST-10 dataset mentioned in example 1 and suppose that the labels went missing for all the inputs $x_i$. A classical clustering task is to try and clusters sample images into 10 clusters (not necessarily corresponding to the $\{0,\cdots,9\}$ digits). To solve this problem, one can consider a toy autoencoder network with a latent layer having 10 activation units each taking values in the unit interval (representing the probability of being a member of any one cluster). The full network specificiation is as follows: 
1. Layer 1 (input layer) and layer 3 (output layer) are identical to layer 1 in the previous example.
2. Layer 2 and Layer 4 contains $1000$ nodes with tanh activation functions.
3. Layer 3 (latent layer) contains $8$ nodes with activation functions $(tanh+1)/2$. This hidden layer structure is thought of as a projection from $\mathcal{X}$ to the 8-simplex where cluster membership can be determined by comparing the activations of the nodes in the latent layer and a virtual node whose activation is $1-\text{sum of activation of the latent layer}$ and choosing the one with maximal values with ties broken arbitrarily.

Similar to example 1, the network is fully connected. We compactly describe the weights between layers as a matrix with the full space of weights being $\mathbb{R}^{256\times 8}\times\mathbb{R}^{8\times 256}$

As a training regime, we consider the construction error loss
$$
L(x_i, \hat{x}_i) = \sum_{i=0}^{255}(x_i^j-\hat{x}_i^j)^2
$$
where $x_i^j, \hat{x}_i^j$ are the j-th components of sample image $x_i$, and the output of the network $f(x_i|w)$ respectively.

</hr>

The computational description of neural networks emphasizes the computational structure of the network rather than its role in the learning machine. It is not clear to me why this computational structure is often emphasized rather than the more intuitive statistical one. One could conjecture that this is a consequence of having the majority of new comers to the field being more familiar with writting code than carring out a deep mathematical analysis. Before we discuss the benefits of the probabilistic view, lets start by clarifying the learning problem by using the language of probability theory.

<h2>Supervised learning and conditional density estimation</h2>
In supervised learning our objective is to minimize 

$$
\int_{\mathcal{x}\times\mathcal{Y}}L(y, f(x|w))q(x,y)dxdy
$$
<h2>The loss function as an objective of learning</h2>

Unde the computational view, the goal of learning is to minimize the loss over the product set $\mathcal{X}\times\mathcal{Y}$. This is not exactly true. To make this point clear, suppose one developed a more powerful digital camera which instead of recording $28\times 28$ pixels, it is capabable of recording $128\times 128$ pixels. This seemingly scaled up problem have a different product set $\mathcal{X}\times\mathcal{Y}$. In learning, we are infact still hopeful that the rule we learned in the unscaled problem is similar to that under the scaled version, otherwise, our learning is not robust enough. We are in fact interested in some computable rule that relate the world state of what is being measured and recorded as points in the measurement set $\mathcal{X}$ and another world state being measured or labelled as points in another measurement set $\mathcal{Y}$.

A more explicit learning objective is to say that, the objective of learning is to better approximate the true rule between the world states being represented by the sets $\mathcal{X},\mathcal{Y}$. Under this statement, it is clear that 1. We do assume that such a rule exist in the world before it is represented or measured.
2. We do assume that such a rule is computable.
3. We do assume that our measuring device did capture enough information about the true relationship.



A better way to specify the goal of learning is to acknowledge the fact that the data are recorded using a measurement system. What we are hoping to learn is not how the pairs $\mathcal{X}$ and $\mathcal{Y}$ are related but rather, how the world state of what is being measured in $\mathcal{X}$ and labelled in $\mathcal{Y}$. One can always change the measuring system and how things are labelled but this doen't change what we are actually measuring or labelling (unless you are learning about a quantum mechanical system).

the random sample (which contains both the training and the validation data set) is composed of a pair $(x_i, y_i)$ where $i\in\{1,\cdots,n\}$ with n being finite. The pair are thought of as realization of 


In this post I argue that the statistical model characterization is informative and in fact required for a thorough understanding of deep learning. In fact, one could claim that without the statistical prespective, the practice of deep learning becomes more of an art than a science. We start by recalling the formal definition of a learning machine, and cast all deep learning problems as probability density estimation problems. In section II, we show how to reconstruct the learning machine view fo each deep learning problem and summarize how the tools of differential and algebraic geometry provides rigorous answers to the following important phenomenas:

 1. Scaling laws empirically observed in large models.
 2. Double descent phenomena observed with very large data sets.
 3. Improved generalization without overfitting despite the extremely large network size relative to the data set.
 4. Data inefficiency of learning algorithms.
 5. The highly singular structure of loss functions.
 6. In what ways does the structure of the network improves learning.
 
 <h2>I: Neural networks as learning machines</h2>
For any random sample $X^n=\{x_1, \cdots, x_n\}$, a learning machine is a map
$$
X^n\rightarrow m(.|w)
$$
from data $X^n$ to a map $m(.|w):\mathcal{X}\rightarrow\mathbb{R}$ which one can use for making decisions (including prediction of future values). The view emphasizes the fact that learning depends on the random sample in a very explicit way.
In machine learning, to build a classifier or a regression model is described as finding the value of the parameter that minimizes a particular loss. For example, in the task of classifying hand written digits with examples from <a href="https://en.wikipedia.org/wiki/MNIST_database">MNIST dataset</a>, the output space is $\{0, \cdots, 9\}$, and the input space $\mathbb{R}^{28\times 28}$

Reinforcement learning is characterized by three characteristics:
- Problems in RL are closed-loop. The agent's action influence future states of the environment. 
- The agent is not given a set of instructions on how to behave, it must figure that on its own from observations.
- The agent is not told what are the consequences of its action, it must learn that on its own.

An RL agent must be able to:
1. Sensation: Sense the environment.
2. Action: Take actions that affect the environment.
3. Goal: The agent must have a goal that is tied to the state of the environment.

Reinformcent learning is different unsupervised learning becase it is all about maximizing some reward signal, not finding "hidden" structure.

An important problem in RL but not in other paradigms of learning is the exploration-exploitation tradeoff. Actions that might lead to maximization of reward signal might not have been tried in the past and discovering them requires trying less optimal actions. They can't be used exclusively without failing at the task. It must balance planning and real-time action selection.

RL integrates optimization and statistics.

RL's objective is to achieve a goal despite uncertainity about the state of the environment.

The concept of an agent is very abstract. It might not be the entire robot/organism. The environment can include internal senstations.

The three elements:
1. Agent.
2. Environment.
3. Policy: a mapping from percieved states of the environment to actions (a set of stimulus-response rules).
4. Reward signal: at every time step, the environment must sent a reward signal (can be 0 or $\infty$). Agent cann't alter the process that does this. Stochastic function of the state of the envrionment and the agent's action.
5. value function: Describes what is good in the long run. is a function on the state of the environment. It represents the total amount of reward the agent can expect starting from the given state. It takes into accounts the results that might follow (it depends on the policy of the agenet).
6. (Optional) model of the environment.

Reinforcement learning is different from evolutionary methods. 

Policy gradient methods: RL methods that directly find the optimal policy by searching in the space of policies while interacting with the environment.

Connection of Reinforcement learning, control and dynamic programming:
- 

Reward Hypothesis:
All goals can be described by the maximization of expected cumulative reward.



As a concrete example, consider the problem of building a learning machine to classify, using previous observations as samples, whether or not a $10\times 10$ grayscale image is for a handwritten number $4$ on an envelop or as a graffiti on a wall. In this case $x$ represents the physical apperance of a handwritten number which we could capture with a camera at anytime, while $y$ is an indicator of the presence of the number $4$. In order to solve the problem, we need to assign numbers and a scale to our measurements of the observable quantities $x, y$. For $y$, a typical convention is to assign the number $1$ if the image contains the number $4$ and $0$ otherwise. Keep in mind that there is nothing special about the numbers $0,1$ other than this notation simplifies our mathematical expressions a little. For $x$, a reasonable choice is for each pixel to vary in the interval $[0,1]$ where $0$ represents the color white, and $1$ the color black. Mathematically, this is a reasonable choice because the intensity of color is linked to the physical notion of electromagnetic frequency which is an infinite. Similar to the case of $y$, the interval $[0,1]$ is arbitrary and if our model is to be successful, our predictions should not be impacted if we change $[0,1]$ to say $[-10,10]$ as the range of values for each pixel. These invariances are important and we will have more to say about them in a future post.

<h2>The learning problem is probabilistic in nature</h2>
There are two primary reasons the learning machine should treat the observed quantities as random. One is the fact that all measurements are corrupted by noise. For our example above, noise enter our measurements of $x$ due to factors such the variability of the manufactoruing process of cameras and/or the lighting conditions at the time of capturing images. For $y$, noise could be a result of having different observers each of whom apply different mental rules for judging when a handwritten digit is $9$ rather than a $4$.

Another reason for treating $x$ and $y$ as random quantities is complexity. In the real world, many quantities interact giving rise to some phenomena. More precisely, there could be another quantity $z$ different from $x$ that is non-observed and non-controlled which affects the range of values of $y$. Observing all quantities that cause variation in $y$ might be impossible or very impractical. 

As a toy example, suppose we are building a learning machine that predicts the distance travelled by a projectile based on its initial velocity and mass. In addition, suppose the data was collected around the world at different altitudes without measuring the air viscosity at the time measurements were collected. A possible immediate consequence is that for the same values of the projectile's initial velocity and mass different displacements of the projectiles will be measured not due to noise but because we did not take into account the viscosity of the air. One should conclude that observing all quantities that influence some quantity of interest is very unlikely to come by without painstaking scientific investigation, and experimental design. As a result, any learning machine operating in the real world should assume that values of $x$ does not uniquely determine values of $y$.

<h3>The supervised learning problem</h3>
In the language of probability theory, one can describe the variations in our observations of $x$ and $y$ by two random variables $X$ and $Y$ with values in the sets $\mathcal{X}:=\{\text{all }10\times 10\text{ grayscale images}\}$ and $\mathcal{Y}:=\{0,1\}$ respectively. To rigorously define $X$ and $Y$ as random variables, one should first define the probability space on which they are to be defined. In the machine learning liteature this is often ignored even though for new students of machine learning, this could cause unnecessary confusion. To see that, suppose a new camera technology was developed such that our images are now $100\times 100$ pixels. The physical objects we are taking pictures of have not changed as a result of changing the camera. In other words, one should not confuse the environment in which the number $4$ physically exists with the images generated by some measuring device.

Defining the probability space on the environment is rather hard and complex. To make progress, we will assume that our mathematical description of the physical environment admits a probability space $(\Omega, \mathcal{B}(\Omega), \mathbb{P})$, where $\Omega$ is the set of all outcomes, $\mathcal{B}(\Omega)$ is the relevant Sigma algebra, and $\mathbb{P}$ is some probability measure. Formally, the random variables are defined as the measurable functions $X:\Omega\rightarrow\mathcal{X}\subset\mathbb{R}^{10\times 10}$, and $Y:\Omega\rightarrow\mathcal{Y}\subset\mathbb{R}$.

Now suppose that the quantities $x$ and $y$ are linked in the real word by some deterministic function $g$ such that 

$$\tag{1}
y=g(x,z)
$$

where $z$ is an unobserved and non-controlled quantity. To account for the measurements noise of $y,x$ and uncertainity about $z$, we modify the model by adding a stochastic term $e$, leadings us to the general statistical model 
$$\tag{2}
Y=f(X,e)
$$
where $f$ is a deterministic measurable map and $e$ is a random variable with values in an unknown set $E$ (also defined on the probability space $\Omega$). It should be clear that solving this problem requires at minimum making some assumption about the nature of $e$. Even more, if the set of all measurable maps $\mathcal{F}:=\mathcal{X}\times E\rightarrow\mathcal{Y}$ is infinite, in general we have an ill-posed problem. This is a result of the fact that our data is always finite and we are trying to select an element from an infinite dimensional space $\mathcal{F}$.

Statistical learning theory and mathematical statistics are concerned with approximating the function $f$ by choosing an element of $\mathcal{F}$ after making some assumptions about the nature of $f$ and $e$. In statistical learning theory, these assumptions are typically reflected in the choice of some model space (a well behaving subset of $\mathcal{F}$) and a loss function which further refines the model space and establishes a notion of distance between its elements.

There is another way of looking at the statistical learning problem above using the language of information geometry. Regardless of the nature of $f$ and $e$, there exist some joint probability measure on the product set $\mathcal{X}\times\mathcal{Y}$ that represents the true probabilistic relationship between $X$ and $Y$. That data generating process (DGP) is thought of as an element in the set $\mathcal{M}$ of all probability measures on $\mathcal{X}\times\mathcal{Y}$. Solving the statistical learning problem amounts to choosing an element from $\mathcal{M}$ that best explains the data collected about $X,Y$. 

It is important to note, this reformulation of the problem, does not necessarily solves the ill-posedness of the general learning problem. To see that, note if the set $\mathcal{X}\times\mathcal{Y}$ is infinite, which is the case in our running example, the set of all probability measures on it is infinite. It is well known that the general approximating of a probability density on an infinite set using finite observations is an ill-posed problem (except for some special cases such as guassian family of probability distributions).

One might wonder, why would we consider the formulation of solving the learning problem as that of a search of a probability measure that best explains the data in the space of all probability measures on some probability space?. One primary reason is due to the impact of geometry on the method used in solving the problem. Another is related to the nature of the statistical learning problem. As we argued earlier in this post, the statistical learning problem is probabilistic in nature and as a result, 

$$
\begin{aligned}
P_{X,Y}(x,y) &= P_X(x)P_{Y|X}(y|x)\\
    &= P_X(x)P_{Y|X}(\{\chi_{f(x,e)=y}=1\}, x)
\end{aligned}
$$

As it stands, we cannot proceed without making assumptions about the noise $e$ and the nature of interactions between $\epsilon$, $z$ and $x$. If we do our best to measure $Y$, one should expect that the measurement noise $e$ is centered around the true value of $y$. This gives rise to the statistical model 
$$
\mathbb{E}_e[Y|X]=g(X,z)
$$
where $f:\mathcal{X}\rightarrow\mathcal{Y}$ is some deterministic measurable map with the term $e$ capturing all the variability in $Y$ due to $z$ and $\epsilon$.
 In its most abstract form $Y=f(X,e)$ is such a statistical model where $\epsilon$ is a stochastic component that captures both measurement noise and the effect of $z$, and $f$ is a measurable map from $\mathcal{X}\times \Epsilon \rightarrow \mathcal{Y}$, where $e \in \Epsilon$. The corresponding statistical model must contain a noise term to describe the measurement noise and uncertainity we  Let $P_{X,Y}(x,y)$ be the joint probability (or density) function of the random variables $X,Y$. We can rewrite this joint probability by conditioning on $x$ whenever $P_X(x)>0$ as follows:
$$
\begin{aligned}
P_{X,Y}(x,y) &= P_X(x)P_{Y|X}(y|x) \\
    &= P_X(x)\chi_{f(x)=y} \\
\end{aligned}
$$
$$P_{X,Y}(x,y)=\int_{Z}dP(x,y,z)=p_X(x)\int_{Z}dP_{Z,Y|X}(z,y|x)=P_X(x)\int_{Z}P_{Y|X,Z}(y|x,z)dP_{Z|X}(z|x)=P_X(x)\int_{Z}\chi_{\{y=g(x,z)\}}dP_{Z|X}(z|x)
$$, where $\chi_{\{y=g(x,z)\}}$ is an indicator random variable for the event $y=g(x,z)$  joint probability represents the joint probability mass (or density) of observing the values $x,z,y$. We can write this joint probability as $P_{X,Z,Y}(x,z,y)=P_{X}(x)P_{Y,Z|X}(y,z|x)=P_{X}(x)\int_{\mathcal{Z}}P_{Y,Z|X}(y,z|x)dz$
Without making additional assumptions, we can immediately write the joint probability (or probability density function) of the random variables $X,Y$ as $P_{X,Y}(x,y)=P_X(x)P_{Y|X}(y|x)=P_X(x)P_{Y|X}(y=g(x,\cdot)|x)$ whenever $P_X(x)>0$. 

In its more abstract form, the statistical model relating $X$ to $Y$ is the model $Y = f(X)$ where $f$ is some measurable map from $\mathcal{X}$ to $\mathcal{Y}$. In regression, we assume that measurement noise and variations in $y$ due to $z$ can all be captured by an error term $\epsilon$ such that the statistical model $Y=f(X)+\epsilon$ holds where $f:\mathcal{X}\rightarrow\mathcal{Y}$ is a measurable map a parameteric map with the parameters linearily related to $y$. In nonlinear regression, we assume the same general form $Y=f(X)+\epsilon$ but $f$ can be any parameteric map (i.e. the parameters determining $f$ can be both linear and nonlinear). The objective of any statistical learning machine is to learn something useful about the function $f$ using previous observations $\{(x^i,y^i): i\in\{1,\cdots,l\}\subset\mathbb{N}\}$ One way to phrase the statistical learning problem is that of Vapnik, where we are interesting in searching for or approximating the map $f$ is the space of all maps $\mathcal{F}$ from $\mathcal{X}$ to $\mathcal{Y}$. In our running example the set of all solutions $\mathcal{F}$ is infinite. To see that, 

This fact can is rigorously treated by introducing a probability space $(\Omega, \mathscr{B}(\Omega), \mathbb{P})$, where $\Omega$ is the sample space such that for any $\omega\in\Omega$, $X(\omega)$ is a 10x10 grayscale image, and $Y(\omega)$ is an indicator for the number $0$. In this case, the joint map $(X,Y):\Omega\rightarrow \mathcal{X}\times \mathcal{Y}$ induces a probability space on the cartesian product $\mathcal{X}\times\mathcal{Y}$ which we hope that it preserves some learnable features of the true object in the real world.

Feedforward artificial neural networks (ANNs) are described in introductory materials {% cite bishop:neural-networks-for-pattern-recognition:1995 goodfellow:deep-learning:2016 -l 117 -l 164  %} as function approximators of some unknown function $f(x)$ defined on some set $\mathcal{X}$ with values in another set $\mathcal{Y}$. Another view that aids the understanding of the challenges faced when using feedforward ANNs is that, for any fixed architecture, the set of all ANN with that architecture form a statistical manifold. To see that, let us view the learning problem as that of a joint probability estimation.




This descriptions mask a rather important view of the learning problem. The learning problem is all about the relationship between two random variables. In probabilistic terms, it is amount approximating the joint probability distribution between two random

Even though this description is correct, it hides the rather important view that feedforward ANNs are statistical models. One could argue that when feedforward ANNs are used for solving learning problems in the real world, the view that they are statistical models is more important for designing and training them because these problems are better explained by understanding the inherit challenges in building any powerful learning machine.

In this post, we adapt the standard view that a statistical model is a pair $(\Theta, \mathfrak{p})$ where $\Theta$, the set of parameters, is a subset of a finite dimensional manifold, and $\mathfrak{p}$ is a map on $\Theta$ with values in the space of all probability measures of interest.

There are several ways one can formulate the learning problem. In its most abstract form, the learning problem can be formulated as that of learning about the relationship between two random variables on some probability space of interest. In the case of supervised learning, we have samples drawn from the probability distributions of both random variables. Typicaly, we are interested in using one random variable to provide information about the other random variable, as a result, one random variable is termed input (machine learning), explanatory/predictor (statistics), independent variable (applied mathematics), while the other is termed output (machine learning), response (statistidcs), dependent variable (applied mathematics).

The characterization of the learning problem as that of learning about the relationship between two random variables could be surprising to some, especially because, in some texts, the input quantity is not considered random (i.e. sampled uniformally with infinite percision) while the output quantity depends on the input in a deterministic manner, and all the variability of the output given the input is explained using a noise term which is random. This indeed could be the case in a simulation or an artificial experiments; however, in the case that the input is a measurement of some observable phenomena, measurement noise makes that quantity random. 

As is the case in probability theory, the joint probability between the input and output describes how the two variables are related. 

It is important to note, that the above designation doesn't necessary imply that the input causes the output, in fact, it is possible that variations in the output causes variation in the input which is the case when the input random variable is a measured degree of sickness while the output is an indictator for the disease itself. 


The interdependence between the input and output variables is fully characterized by their joint probabilities. To see that, let $\mathcal{X}$ be the set of all values that the input variable can take
is a statistical model where $f:\mathcal{X}\times\mathcal{Z}\rightarrow\mathcal{Y}$ is some measurable map. As is the case of statistical learning, we can only collect measurements from $X$ and $Y$ while $Z$ is a hidden random factor.
The input/output terminology reflects that the input variable is an input to a neural network model while the output variable is the output of the same model. This terminology could be misleading since it implies that there could be a cause/effect relationship between the input and output which is not always true. The generality of the 

 enough contex regarding the nature of the learning problem the fact that, Feedforward artifical neural networks  building a learning machine, we are not interested in all the maps $\mathcal{X}\rightarrow\mathcal{Y}$, rather, we are interested in making sense of the relationship between elements of $\mathcal{X}$ and that of $\mathcal{Y}$ in the hope that one can use different values of $\mathcal{X}$ to predict outcomes in $\mathcal{Y}$ (supervised learning) or know something about the data generating process of $\mathcal{X}$ which one can use to to relate elements of $\mathcal{X}$ to future sets such as $\mathcal{Y}$.  optimal decisions about choosing values from $\mathcal{Y}$ given values in $\mathcal{X}$ after collecting some samples from both $\mathcal{X, Y}$ (supervised learning) or just $\mathcal{X}$ (unsupervised learning). In other words, the learning settings is always that of making decisions based on the statistical properties of our samples. As a result, the language of probability theory, statistical inference, and information geometry is more appropriate for studying the design and training of feedforward ANNs.

In this post, I attempt to recast all the elements of a supervised learning problems, including classification and regression in a uniform language and describe the design and the training process of feedforward ANNs in the language of information geometry with the hope that the abstract settings will provide a strong link between the theory of feedforward ANNs and that of statistical manifolds such as restricted boltzmann machines, exponential and mixture families. 

Consider the problem of classifying 10x10 grayscale (8 bit) images by whether or not the image contains a dog. In this case, the set $\mathcal{X}$ is the power set of all grayscale images, and the set $\mathcal{Y}$ consists of two elements that can be arbitrarily labeled as $1$ for when the image contains a dog, and $0$ otherwise. Because of measurement noise, the varying ways one can take pictures, and the degree of similarity between the different features of dogs and other animals such as wolves and foxes, our decisions of whether or not a particular image $x\in\mathcal{X}$ contains a dog are probabilistic in nature.

Before we can introduce the random variables of interest, we need to define the probability space on which the random variables are defined. In this case, the sample space $\Omega$ is the cartesian product $\mathcal{X}\times \mathcal{Y}$ which is very large but finite. The probability space is $(\Omega, \mathscr{P}(\Omega), \mathbb{P})$ where $\mathscr{P}(\Omega)$ is the power set of $\Omega$, and $\mathbb{P}$ is an arbitrary probability measure on the measure space $(\Omega, \mathscr{P}(\Omega))$.

Let $X:\Omega\rightarrow\mathbb{R}$ be the random variable 

If the noise is high occluding our ability in making decisions, our ability in deciding whether or not a particular image contains a dog  where the labels are arbitrarily assigned, say $1$ when the image contains a dog We are clearly not interested in trivial maps that  that assign values in $\cal{Y}$ based on the value of any particular pixel of any image even though this is clearly a legitimate map between $\cal{X}$ and $\cal{Y}$. Even more, in all practical applications, images are generated using a measurement process which is no matter what the measuring device is, contains noise. In other words, deciding whether or not an image is that of a dog must take into account the probabilistic nature of the problem. In other words, the probabilistic nature of classification and that of regression is not a mere convenience for applying tools from probability theory but rather, is the correct language for studying problems concerned with making optimal inference over measurements that are made of some phenomena in the real world.

This probabilistic view does not enter the problem only because of noise.  


As we will see, a probabilistic view  clear if we we are interested in all functions between $\cal{X}$ and $\cal{Y}$ or any particular subset. As we will see later, a probabilitic view of feedforward artificial neural networks (ANN for short) is more useful and infact more accurate.  feedforward It is not clear from this description what is the nature of the sets $\cal{X}$, $\cal{Y}$ and what type of functions are of interest. In other words, are we interested in all  an input space $\cal{X}$ and output space $\cal{Y}$. However, this description does not say what kind of functions we are interested in. Feed forward artificial neural networks are used to solving particular type of problems that are statistical in nature. This characterization is indeed true, and useful for building intuition; however, it can be misleading. This description is not sufficient for conveying the true nature of the functions we are trying to approximate. To help make this argument, let us consider the case of an image classifier that classifier 10x10 grayscale images into whether or not the image is that of a dog. Suppose a pixel is measured from 0 (black) to 255 (white). The set of all 10x10 grayscale images is a set with 256^(10x10) elements. A large but a finite set. The output is a 

where each element is a tuple in 10    enough for truely characterizing what neural networks   in introductory materials, the focus is generally is on their layered structured, and on how to use gradient based optimization to solve problems. This publication has a rather different goal. Rather than focusing on the layered structure, we will recast the elements typically encountered in the language of Information Geometry the full rigor of probability theory. My objective is to organize the mathematical concepts in my mind and cast the problems encountered in applied applications in 

$$
e = mc^2. \tag{1}
$$
{% endkatexmm %}

Cool!