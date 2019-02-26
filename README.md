# Wasserstein Examples via Projected Sinkhorn Iterates

*A repository that implements the projected sinkhorn algorithm, and applies it towards generating Wasserstein adversarial examples. Created by [Eric Wong](https://riceric22.github.io), and joint work with Frank R. Schmidt and [Zico Kolter](http://zicokolter.com). See our paper on arXiv [here][paper].*

[paper]: http://arxiv.org/abs/1902.07906

[lambertw]: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.lambertw.html
[sinkhorn]: https://arxiv.org/abs/1306.0895

## News
+ 02/21/2019 - Initial release (v0.1) with preprint. 

## What is in this repository? 
+ An implementation of the Projected Sinkhorn Iteration for images described in our [paper][paper].
+ A PyTorch port of the `lambertw` function [(scipy documentation)][lambertw]. 
+ Model weights for the models trained/evaluated in the paper. 
+ Training and evaluation code for adversarial training over the Wasserstein ball

## Installation & Usage
You can install this these functions with 
`pip install projected_sinkhorn`. The package contains the following functions: 
+ `projected_sinkhorn(X, Y, C, epsilon, lam, verbose=False, plan=False, objective='2norm', maxiters=50, return_objective=False)` computes the projection of `Y` onto the Wasserstein ball around `X`. 
+ `conjugate_sinkhorn(X, Y, C, epsilon, lam, verbose=False, plan=False, objective='2norm', maxiters=50, return_objective=False)` computes the support function (conjugate) of the Wasserstein ball. 
+ `wasserstein_cost(X, p=2, kernel_size=5)` creates a cost matrix for the p-Wasserstein distance for a given kernel size. 
+ `lambertw(z0, tol=1e-5)` computes the lambertw function of `z0` on the zero branch. The code is a direct port of the zero branch of the [scipy version][lambertw]. 

## Why do we care about Wasserstein adversarial examples? 
While much work in adversarial examples research has focused on norm-bounded perturbations, these types 
of perturbations largely ignore structure that we typically believe to exist in the data. For example, 
in images we can consider transformations such as translations, rotations, or distortions to be 
small, adversarial changes, yet these types of transformations can be extremely large when measured 
with respect to some p-norm. This work represents a step towards describing convex perturbation regions: 
convex sets of allowable perturbations beyond the norm-ball, which can capture structure or invariants in 
the application domain. 

To this end, we propose adversarial examples which are close in Wasserstein distance. For images, this has 
an interpretation of moving pixel mass: two images that are close in Wasserstein distance require moving only 
a small amount of pixel mass a small distance to transform one image to the other. Examples of image transformations 
that are small in Wasserstein distance include rotations and translations. In practice, we find that adversarial examples 
generated within this ball have perturbations that reflect the actual content and structure of the image itself. For example, in the following figure we can see a Wasserstein perturbation on the top row, which doesn't attack the empty space around the six,  vs an l-infinity perturbation on the bottom row, which attacks all pixels indiscriminately. 

<img src="https://github.com/locuslab/projected_sinkhorn/blob/master/images/perturbation.png" width="800">

We derived a fast, modified [sinkhorn iteration][sinkhorn] that solves the projection problem onto the Wasserstein ball, and restrict our transport plans to local regions to make this tractable for image datasets. The resulting algorithm is fast enough to be run as a subroutine within a PGD adversary, and furthermore within an adversarial training loop. For CIFAR10 classifiers, we find that an adversarial radius of 0.1 is enough to fool the classifier 97% of the time (equivalent to allowing the adversary to move 10\% of the mass one pixel), when restricted to local 5 by 5 transport plans. The main experimental results in the paper can be summarized in the following table. 

|          | CIFAR10 Acc | CIFAR10 Adv Acc (eps=0.1) | MNIST Acc | MNIST Adv Acc (eps=1.0) |
| --------:| ----------:|----------:| ---------:| ------------:|
| Standard     |       95% |       3% |     99% |          4% |
| l-inf robust |       66% |      61% |     98% |         48% |
| Adv training |       81% |      76% |     97% |         86% |
| Binarization |         - |        - |     99% |         14% |
