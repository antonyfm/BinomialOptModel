import numpy as np
import math

def binomial_model(N, T, S0, K, rf, SD):
    """
    N = number of binomial iterations
    T = total time period
    S0 = initial stock price
    K = strike price    
    rf = risk free interest rate per annum
    SD = Standard deviation
    
    Calculated values
    deltaT = Total time / No of steps
    u = e^(SD * sqrt(deltaT))
    d = 1/u
    p = (e^(rf X delta t) - d) / (u - d)
    q = 1 - p
    """
    
    deltaT = T / N
    u = math.exp(SD * math.sqrt(deltaT))
    d = 1 / u
    p = (math.exp(rf * deltaT) - d) / (u - d)
    q = 1 - p

    # make stock price tree
    stock = np.zeros([N + 1, N + 1])
    for i in range((N + 1)):
        for j in range(i + 1):
            stock[j, i] = S0 * (u ** (i - j)) * (d ** j)
    print("Forward binomial pricing tree:")
    print (stock)

    # Generate option prices recursively
    value = math.exp(-1 * rf * deltaT)
    print(value)
    option = np.zeros([N + 1, N + 1])
    option[:, N] = np.maximum(np.zeros(N + 1), (stock[:, N] - K))
    for i in range(N - 1,-1,-1):
        for j in range(0, i + 1):
            option[j, i] = (
                math.exp(-1 * rf * deltaT) * (p * option[j, i + 1] + q * option[j + 1, i + 1])
            )
    print("Reverse binomial pricing tree:")
    print(option)
    return 0

if __name__ == "__main__":
    print("Calculating example option price:")
    op_price = binomial_model(3, 0.25, 2, 2.1, 0.08, 0.63)
