# Neural-Network-Heston

In this project, we train a neural network on Python using monte carlo simulations to optimise the parameters of the Heston model. We then use these to check how accurately the algorithm predicts stock prices using this as the underlying model. The Heston model is solved by following the Quadratic-Exponential (QE) algorithm as illustrated by Leif Andersen in his 2008 paper: Efficient Simulation of the Heston Stochastic Volatility Model. Initial market conditions are taken to be the ones 30 days in the past, and today's market prices are the ones on which parameters are optimised. Further details are found in the comments of the the Python script, heston_model_neural.py.

The repository also includes a plot of the predicted prices and the real prices from a past run, labelled Heston_neural.png, as well as the associated loss curve of the neural network training, labelled loss_curve.png.
