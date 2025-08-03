---
title: Thermodynamic integration and Evidence
subtitle: Here is some extra detail about the post.
layout: default
date: 2025-05-24
keywords: thermodynamic integration, prior, posterior, generalized likelihood, free energy, bayesian factors, WBIC
published: false
---

{% katexmm %}
Statistical models are fundamental mathematical structures in applied statistical work. In this paper, we adopt the view that parametric statistical models are maps from a set of parameters $W$ to a set of probability distributions $\mathcal{P}$ defined over the same probability space $\Omega$. In applications, $W$ is topological subspace of some euclidean space $\mathbb{R}^d, d\in\mathbb{N}$ with the standard euclidean topology, and $\Omega$ is some probability space associated with some set of outcomes $\mathcal{X}$ equipped with some topology. 

At this level of generality, any statistical model is not very useful for applications without additional regularity assumptions.

Singular learning theory aims to clarify the mathematical foundations of  statistical inference in the case that a parametric statistical model is non-identifiable or the Fisher information metric is degenerate, the so called singular statistical models. The following are some of the key highlights of singular learning theory as developed by Sumio Watanbe:
$$
\begin{aligned}
p(w|X^n) &= \frac{p(X^n|w)\varphi(w)}{p(X^n)}
\end{aligned}
$$

The normalizing constant $p(X^n)$ is an important quantity for statistical inference and neural networks (the normalizing constant is a corner stone of building autoencoder networks and generative models in general, we will elaborate on this connection in a future post). When the dimension of the parameter space $W$ is large, computing the normalizing constant is a hard problem in general except for some very special models such as exponential families with conjugate priors.

This difficult is an immediate consequence of:
1. The geometry of the likelihood function $p(X^n|w)$ and consequently the negative log likelihood loss $-\text{log}p(X^n|w)$ is heavily influenced by algebraic singularities in the parameter space $W$ and can't be approximated by any quadratic forms. In particular, we cannot use laplace's approximation. This geometry has been extensively studied by S. Watanabe (TODO: add citations).
2. $p(X^n)$ requires computing $\int_W{p(X^n|w)\varphi(w)dw}$, a high dimensional integral over the entire parameter space $W$. The larger the dimension $d$ of $W$, the more difficult the problem becomes, an immediate consequence of the curse of dimensionality.

As a motivating example, let us consider a simple bayesian model where we can compute the normalizing constant analytically.
<h4>Example 1: Normal likelihood with normal prior</h4>
Suppose n random samples were drawn from a one dimensional normal distribution with true parameters $\mu_0\in\mathbb{R}, \sigma^2_0\in(0,\infty)$. Also, suppose one used a statistical model with a likelihood function $p(X^n|\mu)$ and a prior $\varphi(\mu)$ defined as follows,

$$
\begin{aligned}
p(X^n|\mu) &= \prod_{i=1}^n p(X_i|w)\\
&= \bigg(\frac{1}{\sqrt{2\pi}\sigma}\bigg)^n \text{exp}\bigg(\frac{-1}{2\sigma^2}{\sum_{i=1}^n(X_i-\mu)^2}\bigg)
\end{aligned}
$$

and

$$
\varphi(\mu) = \frac{1}{\sqrt{2\pi}\beta}\text{exp}\bigg(\frac{-1}{2\beta^2}(w-\alpha)^2\bigg)
$$
where $\mu\in\mathbb{R}$ is random while $\alpha\in\mathbb{R},\beta^2,\sigma^2\in(0,\infty)$ are known and fixed. Let $\overline{X}$ be the mean of the data. By Baye's rule, we can write the posterior density as

$$
\begin{aligned}
p(\mu|X^n) &= \frac{\varphi(\mu)p(X^n|\mu)}{p(X^n)}\\
    &= \frac{1}{p(X^n)} \frac{1}{\sqrt{2\pi}\beta}\bigg(\frac{1}{\sqrt{2\pi}\sigma}\bigg)^n
    e^{\frac{-1}{2}\bigg(
        \frac{1}{\beta^2}(\mu-\alpha)^2+\frac{1}{\sigma^2}\sum_{i=1}^n(X_i-\mu)^2
        \bigg)}
\end{aligned}
$$

Using some algebra, we could simplify the expression in the exponent further,

$$
\begin{aligned}
\frac{1}{\beta^2}(\mu-\alpha)^2+\frac{1}{\sigma^2}\sum_{i=1}^n(X_i-\mu)^2 
    &=\frac{1}{\beta^2}(\mu-\alpha)^2+\frac{1}{\sigma^2}\sum_{i=1}^n(X_i-\bar{x}+\bar{x}-\mu)^2\\
    &=\frac{1}{\beta^2}(\mu^2-2\mu\alpha+\alpha^2)+
      \frac{1}{\sigma^2}\sum_{i=1}^n(X_i-\bar{x})^2-
      \frac{1}{\sigma^2}2(\mu-\bar{x})\sum_{i=1}^n(X_i-\bar{X})+
      \frac{1}{\sigma^2}n(\mu-\bar{x})^2
\end{aligned}
$$

To simplify notation, let $a=\frac{1}{\beta^2}, b=\frac{n}{\sigma^2}$. It follows that,

$$
\begin{aligned}
\frac{1}{\beta^2}(\mu-\alpha)^2+\frac{n}{\sigma^2}(\mu-\bar{x})^2
    &=a(\mu-\alpha)^2+b(\mu-\bar{x})^2\\
    &=a\mu^2-2a\alpha\mu+a\alpha^2+b\mu^2-2b\bar{x}\mu+b\bar{x}^2\\
    &=\mu^2(a+b)-2\mu(a\alpha+b\bar{x})+a\alpha^2+b\bar{x}^2\\
    &=(a+b)\bigg(\mu^2-2\mu\frac{a\alpha+b\bar{x}}{a+b}+\frac{a\alpha^2+b\bar{x}^2}{a+b}\bigg)\\
    &=(a+b)\bigg(
        \mu^2-2\mu\frac{a\alpha+b\bar{x}}{a+b}+
        (\frac{a\alpha+b\bar{x}}{a+b})^2-
        (\frac{a\alpha+b\bar{x}}{a+b})^2+
        \frac{a\alpha^2+b\bar{x}^2}{a+b}
        \bigg)\\
    &=(a+b)\bigg(\mu-\frac{a\alpha+b\bar{x}}{a+b}\bigg)^2+
      \bigg(a\alpha^2+b\bar{x}^2-\frac{(a\alpha+b\bar{x})^2}{a+b}\bigg)
\end{aligned}
$$

Note that, considering the above expression as a function of $\mu$, we could immediately write the posterior distribution of $\mu$, it is a normal distribution with mean $\frac{a}{a+b}\alpha+\frac{b}{a+b}\bar{x}$ and variance $\frac{1}{a+b}$ (this is the conjugacy between the prior and the likelihood that is well know in statistics). After we arrange terms and solve for $p(X^n)$, we have

$$
\begin{aligned}
% \frac{1}{Z_n}\sqrt{\frac{1}{2\pi\beta^2}}\sqrt[n]{\frac{1}{2\pi\sigma^2}}
%     e^{
%     \frac{-1}{2\sigma^2}\sum_{i=1}^n(X_i-\bar{x})^2+
%     \frac{-1}{2}\big(a\alpha^2+b\bar{x}^2-\frac{(a\alpha+b\bar{x})^2}{a+b}\big)+
%     (\frac{-(a+b)}{2})\big(\mu-\frac{a\alpha+b\bar{x}}{a+b}\big)^2
%     } &=
%     \sqrt{\frac{a+b}{2\pi}}e^{
%         \frac{-(a+b)}{2}(\mu-\frac{a\alpha+b\beta}{a+b})^2
%     }\implies\\
% \frac{1}{Z_n}\sqrt{\frac{1}{\beta^2}}\sqrt[n]{\frac{1}{2\pi\sigma^2}}
%     e^{
%     \frac{-1}{2\sigma^2}\sum_{i=1}^n(X_i-\bar{x})^2+
%     \frac{-1}{2}\big(a\alpha^2+b\bar{x}^2-\frac{(a\alpha+b\bar{x})^2}{a+b}\big)
%     } &=
%     \sqrt{a+b}\implies\\
    p(X^n) &= 
        \bigg((a+b)\beta^2(2\pi)^n\sigma^{2n}\bigg)^{-\frac{1}{2}}
        \text{exp}\bigg(
            \frac{-1}{2\sigma^2}\sum_{i=1}^n(X_i-\bar{x})^2+
            \frac{-1}{2}\big(a\alpha^2+b\bar{x}^2-\frac{(a\alpha+b\bar{x})^2}{a+b}\big)
        \bigg)
\end{aligned}
$$
<hr>
Thermodynamic integration, a method developed in statistical physics to compute the difference in free energy of two systems, is deeply connected to the computation of normalizing constants and in fact provides theoretical insights by casting the problem as calculating the "length" of a curve in the statistical model space. In this post, we review this important connection in detail and highlight the important role of thermodynamic integration in singular models as developed by Watanabe. We start by recalling a few important quantities from statistical physics and their connection to bayesian statistica and later review the Widely Applicable Information Criteria introduced by Watanabe for model selection in singular statistical models.

<h2>Thermodynamic integration</h2>
Denote the potential energy of a system $U(w)$ defined over the configuration space $W$ such that the gibbs density 
$$
p(w) \triangleq \frac{1}{Z} e^{-U(w)}
$$
is a well defined probability density (i.e non-negative and integrates to 1). The partition function $Z$ is such that 
$$
Z = \int_W e^{-U(w)}dw
$$. 

If we are to observe the configuration of the system with a function $g(w)$, we denote the average value of this observable by
$$
\begin{aligned}
\mathbb{E}_{p(w)}[g(w)] &= \int_W g(w) p(w)dw \\
    &= \int_W g(w) \frac{1}{Z}e^{-U(w)}dw
\end{aligned}
$$

The free energy $F$ of the system is defined by

$$
F \triangleq -\text{log}Z = -\text{log}\int_W e^{-U(w)}\varphi(w)
$$

Suppose we are given two systems whose potential energies are $U_1(w)$, and $U_2(w)$. The difference in free energy between the the two system is

$$
\begin{aligned}
F_2 - F_1 &= -\text{log}Z_2 + \text{log}Z_1\\
    &= -\text{log} \frac{Z_2}{Z_1}
\end{aligned}
$$

Evaluating this quantity requires the computation of two high dimensional integrals. In addition, because we are dealing with small fractions, our computation of $F_2-F_1$ often suffers from numerical errors.

The key idea of thermodynamic integration is to consider a system whose potential energy is on a path between the two systems. More precisely

$$
U_{\beta}(w) = U_1(w) + \beta(U_2(w)-U_1(w))
$$
where $\beta \in [0,1]$. Note that $U_{\beta=0} = U_1$, and $U_{\beta=1} = U_2$.

The gibbs density and the free energy of the ensemble system are
$$
p_\beta(w) = \frac{1}{Z(\beta)}e^{-U_{\beta}(w)}
$$

$$
F(\beta) = -\text{log}Z(\beta) = -\text{log}\int_W e^{-U_{\beta}(w)}dw
$$
respectively. Observe that
$$
\begin{aligned}
\frac{d}{d\beta}F(\beta) &= \frac{d}{d\beta}\bigg[-\text{log}{Z(\beta)}\bigg]\\
    &= \frac{-1}{Z(\beta)} \frac{d}{d\beta} \int_W e^{-U_\beta(w)}dw\\
    &= \int_W \frac{-1}{Z(\beta)} \frac{d}{d\beta}\bigg\{e^{-U_1(w)+\beta(U_2(w)-U_1(w))}dw\bigg\}\\
    &= \int_W -\big(U_2(w)-U_1(w)\big) \frac{1}{Z(\beta)}e^{-U_\beta(w)}dw\\
    &= \mathbb{E}_{p_\beta(w)}\big[-\big(U_2(w)-U_1(w)\big)\big]
\end{aligned}
$$

So far, it might seem as if we have not gained any computational advantage; however, by the mean value theorem, we have

$$
\begin{aligned}
F(1) - F(0) &= \int_0^1 \frac{d}{d\beta}F(\beta)d\beta\\
    &= \int_0^1 \mathbb{E}_{p_\beta(w)}\big[-\big(U_2(w)-U_1(w)\big)\big] d\beta
\end{aligned}
$$
in other words, the difference in free energy is reduced to the 1 dimensional integral over $\beta\in[0,1]$ of the negative difference in potential energies (w.r.t the gibbs density of the ensemble).

<h2>Thermodynamic integration and bayesian inference</h2>
At this point, one should ask, how is this related to bayesian inference. To see the link, note that for any probability density $p(w)$ over sample space $W$, we can write it in a gibbs density representation over its support as follows
$$
p(w) = e^{-\text{log}\frac{1}{p(w)}}
$$
where $\text{log}\frac{1}{p(w)}$ can serve as a "potential energy" over the parameter space $W$.

Similarily, suppose we have a random sample $X^n$ from some unknown probability distribution. Under a statistical model with likelihood $p(X^n|w)$ and prior $\varphi(w)$, the posterior distribution $p(w|X^n)$ can be written in gibbs density representation as

$$
p(w|X^n) = \frac{1}{p(X^n)}e^{-\text{log}\frac{1}{\varphi(w)p(X^n|w)}}
$$
where the free energy is
$$
\begin{aligned}
F_n &= -\text{log}p(X^n)\\
    &= \int_W \varphi(w)p(X^n|w)dw
\end{aligned}
$$

Thermodynamic integration enters the problem by considering an ensemble "potential energy"
$$
-\text{log}p_{\beta}(w) = -\text{log}\varphi(w)+\beta\big[-\text{log}p(w|X^n)+\text{log}\varphi(w)\big]
$$
which is a mixture of the potential energies of our prior $\varphi(w)$ ($\beta=0$) and the posterior $p(w|X^n)$ ($\beta=1$). Note that, the Gibbs distribution associated with this ensembled potential energy is

$$
\begin{aligned}
p_{\beta}(w) &= \frac{1}{Z(\beta)}e^{-\text{log}\frac{1}{p_{\beta}(w)}}
    &= \frac{1}{Z(\beta)}\varphi(w)p(w|X^n)^{\beta}
\end{aligned}
$$
which of thought of as a generalized posterior or tempered posterior distribution.

{% endkatexmm %}