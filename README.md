# quantile-net

A package for probabilistic regression with neural networks. The novelty comes from the loss function, which utilizes the method for weighting multi-task loss functions proposed in https://arxiv.org/pdf/1705.07115.pdf. The mutli-task loss function allows for easy prediction of a range of quantiles, from which a cumulative density function can be drawn. 

While the package allows you to specify a neural network architecture easily with configs, the loss function could be used in conjunction with any architecture.
