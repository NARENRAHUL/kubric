import requests
import pandas
import scipy
import numpy
import numpy as np 
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"

import numpy as np 
import matplotlib.pyplot as plt 

def coefficients(x, y): 
	n = np.size(x) 
	m_x, m_y = np.mean(x), np.mean(y) 
	p= np.sum(y*x) - n*m_y*m_x 
	q = np.sum(x*x) - n*m_x*m_x 
	b_1 = p / q 
	b_0 = m_y - b_1*m_x 
	return(b_0, b_1) 


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    #print(response.text)
    data = response.text
    #print(type(data))
    split = data.split("price")
    lisA = split[0].split(",")
    del lisA[0]
    lisP = split[1].split(",")
    del lisP[0]
    for i in range(0,len(lisA)):
    	lisA[i]=float(lisA[i].strip())
    for i in range(0,len(lisP)):
    	lisP[i]=float(lisP[i].strip())
    #print(lisP)
    x = numpy.array(lisA)
    y = numpy.array(lisP)
    coe = coefficients(x, y) 
    area = (area * coe[1]) + coe[0]
    return area


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
