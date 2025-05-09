import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2
import yfinance as yf
import pandas as pd


startTime = datetime.now()

# Define how long ago (in days) you would like to predict from
period_days = 30

# Fetch the 3-month Treasury Bill rate (^IRX)
t_bill = yf.Ticker("^IRX")
t_bill_rate = t_bill.history(period = "1d")['Close'].iloc[-1] / 100  # Convert to decimal
r_real = t_bill_rate
print(f"3-Month Treasury Bill Rate (Risk-Free Rate): {t_bill_rate}")

# Use real-world market option prices from Yahoo Finance 
ticker = "SPY"  # Using SPY as a proxy for European options
spy = yf.Ticker(ticker)
expirations = spy.options 


# Fetch the SPY price from period_days days ago to use as initial value for stock price, S0
S0_real = spy.history(period = f"{period_days}d")['Close'].iloc[-1]
print(f"Using real initial stock price: {S0_real}")

# Option chain for a near-term expiration (essentially today/tomorrow)
opt_data = spy.option_chain(expirations[1])  
calls = opt_data.calls[['strike', 'lastPrice', 'impliedVolatility']]

# Remove NANs and invalid data
calls = calls.dropna()
calls = calls[calls['impliedVolatility'] > 0]

# Fetch real strikes for visualization and volatility 
strikes = calls['strike'].values
implied_vol = calls['impliedVolatility'].values

# Use today's real market prices as input for neural network calibration
market_prices = np.concatenate(np.expand_dims(calls['lastPrice'].values, axis=1))

# Time to maturity and timestep size
expiry_date = datetime.strptime(expirations[1], "%Y-%m-%d")
today = datetime.today()
#T_real = (expiry_date - today).days / 365
T_real = period_days / 365
timestep = T_real / 100

# Strike price, Implied Volatility, Risk-free rate
X = calls[['strike', 'impliedVolatility']]
X['risk_free_rate'] = r_real


# Function to train neural network on training data
def neural_model_train(X_train, Y_train):
    

    # Defining neural network architecture, 2 layers of 64 neurons with rectifier activation function, output gives the 5 varied Heston parameters
    neural_model = Sequential([Dense(64, activation = 'relu', input_shape = (1,)), Dense(64, activation = 'relu'), Dense(4, activation = 'linear')])

    # Compiling the model with Adam optimizer and a standard MSE loss function, fixed learning rate 
    neural_model.compile(optimizer = adam_v2.Adam(learning_rate = 0.001), loss = 'mse')

    # Training the model and storing training history
    history = neural_model.fit(X_train, Y_train, epochs = 100, batch_size = 32, verbose = 1)

    # Plot and save loss curve figure
    plt.figure(figsize = (10, 5))
    plt.plot(history.history['loss'], label = 'Training Loss', color = 'blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Neural Network Training Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png") 
    plt.show()

    return neural_model


# This algorithm follows the Quadratic-Exponential (QE) implementation by Leif Andersen (Efficient Simulation of the Heston Stochastic Volatility Model)
def heston_model_montecarlo(T, dt, monte_runs, S0, mu, K, kappa, theta, epsilon, rho, v0):

    # Defining helpful parameters
    psi_c  = 1.5  # Arbitrary value 1 < psi_c < 2 used for switching rule. According to Andersen, this makes little difference
    gamma1 = 1.0  # Constants gamma1 and gamma2 give the proposed discretization scheme (gamma1 = 1, gamma2 = 0 is Euler for example)
    gamma2 = 0.0

    # The following are timestep dependent parameters that are cached before the loop as they are independent of the dynamical variables
    k1 = np.exp(- kappa * dt)
    k2 = epsilon**2 * k1 / kappa * (1 - k1) 
    k3 = theta * epsilon**2 / (2 * kappa) * (1 - k1)**2
    K0 = -rho * kappa * theta * dt / epsilon
    K1 = gamma1 * dt * (kappa * rho / epsilon - 0.5) - rho / epsilon
    K2 = gamma2 * dt * (kappa * rho / epsilon - 0.5) + rho / epsilon
    K3 = gamma1 * dt * (1 - rho**2) 
    K4 = gamma2 * dt * (1 - rho**2)
    
    # Number of steps and initialising arrays to hold data
    n_steps = int(T / dt)
    lnS     = np.zeros((monte_runs, n_steps))
    v       = np.zeros((monte_runs, n_steps))

    # Initial conditions for stock price (S) and volatility (v)
    lnS[:, 0] = np.log(S0)
    v[:, 0]   = v0
    
    # Update loop for ln(S) and v
    for t in range(1, n_steps):

        # Parameters m, s^2 and psi as defined by Andersen
        m   = theta + (v[:, t - 1] - theta) * k1 
        s2  = v[:, t - 1] * k2 + k3 
        psi = np.maximum(s2 / m**2, 1e-5)
        
        # Random uniform and normal variables to be used later
        Uv = np.random.uniform(0, 1, monte_runs)
        Zx = np.random.normal(0, 1, monte_runs)
        
        # Switching rule and updating v (must be done before updating ln(S))
        for i in range(monte_runs): # Running through all Montecarlo runs
         
            # Case when psi < psi_c, the critical value defined above
            if psi[i] < psi_c:
    
                # Calculating parameters a, b as defined by Andersen and the standard normal variable Z_v 
                b2  = 2 / psi[i] - 1 + np.sqrt(2 / psi[i]) * np.sqrt(np.maximum(2 / psi[i] - 1,0))
                a   = m[i] / (1 + b2)
                Zv  = scipy.stats.norm.ppf(Uv[i])

                # Updating v
                v[i, t] = np.maximum(a * (np.sqrt(b2) + Zv)**2, 0)
               
            # Case when psi > psi_c
            else:
                
                # Calculating parameters p and beta as defined by Andersen
                p    = (psi[i] - 1) / (psi[i] + 1) 
                beta = 2 / (m[i] * (psi[i] + 1))

                # Piecewise function PSI inverse as defined by Andersen
                if Uv[i] < p:
                    v[i,t] = 0

                else:
                    v[i, t] = np.maximum(np.log((1 - p) / np.maximum(1 - Uv[i], 1e-5)) / beta, 0)

               
        # Updating ln(S)
        lnS[:, t] = lnS[:, t - 1] + K0 + mu * dt +  K1 * v[:, t - 1] + K2 * v[:, t] + np.sqrt(K3 * v[:, t - 1] + K4 * v[:, t]) * Zx

    # Exponentiating to get stock price (S) as a functon of time
    S = np.exp(lnS)
    
    # Payoff for a European call option
    payoff = np.maximum(S[:, n_steps - 1] - K, 0)

    # Estimate option price under risk-neutral measure with risk-free rate mu using Monte Carlo
    price = np.exp(- mu * T) * np.mean(payoff)

    return price

# Function to generate training data using Monte Carlo simulations of the Heston model
def training_data_generator(n_samples, S0, strikes,  mu, implied_vol ):

    # Parameters to sample within a reasonable estimation range
    kappa   = np.random.uniform(0.5 , 2,    n_samples)
    theta   = np.random.uniform(0.01, 0.1,  n_samples)
    epsilon = np.random.uniform(0.1 , 1,    n_samples)
    rho     = np.random.uniform(-0.5, 0.5,  n_samples)

    # Time to maturity, timestep size and Monte Carlo runs
    T  = T_real
    dt = timestep
    monte_runs = 50

    # Generate option prices using Heston model for above parameter set
    option_prices = np.array([heston_model_montecarlo(T, dt, monte_runs, S0, mu, K = strike,  kappa = k, theta = t, epsilon = e, rho = r, v0 = v) for strike, k, t, e, r, v in zip(strikes, kappa, theta, epsilon, rho, implied_vol)])

    # Set option prices as features and Heston parameters as labels
    Xfeatures = np.expand_dims(option_prices, axis = 1) 
    Yparams = np.stack([kappa, theta, epsilon, rho], axis = 1)

    return Xfeatures, Yparams


# Generate sample training data and train neural network
n_samples = len(X)
X_train, Y_train = training_data_generator(n_samples, S0_real, strikes, r_real, implied_vol)
neural_model = neural_model_train(X_train, Y_train)

# Compute risk-neutral drift
mu_real = t_bill_rate

# Predict Heston parameters using trained model, we can then use these predicted parameters along with the market price options (to be set as S0) to predict the future trend of S(t)
predicted_parameters = neural_model.predict(market_prices)

# Use real S0 and real mu for predictions
predicted_prices = np.array([
    heston_model_montecarlo(T = T_real, dt = timestep, monte_runs = 50, S0 = S0_real, mu = r_real, K = strike,
                            kappa = p[0], theta = p[1], epsilon = p[2], rho = p[3], v0 = v)
    for strike, p, v in zip(strikes, predicted_parameters, implied_vol)
])

# Check runtime for future optimization
print("Runtime", datetime.now() - startTime) 

# Plot and save Model-Predicted vs. Real Market Prices
plt.figure(figsize = (10, 5))
plt.plot(strikes, market_prices, 'o-', label = "Real Market Prices", color='blue')
plt.plot(strikes, predicted_prices, 's-', label = "Heston Model Predictions", color = 'red')
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.legend()
plt.title("Model-Predicted vs. Real Market Option Prices")
plt.savefig("Heston_neural.png")
plt.show()

