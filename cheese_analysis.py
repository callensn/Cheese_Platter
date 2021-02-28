"""
Analysis and answering of the questions for 
the cheeseplate csv data given by Alaska Airlines
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import linalg as la
from scipy import stats
import math
from statistics import mean
import time
from scipy.spatial import KDTree

df = pd.read_csv('cheesplate.csv')

#plt.hist(df['Cheese Platters Sold'])
#plt.show()
#print(x)
def scatter_plots():
    x_departure = df['Passengers Boarded']
    y_sold = df['Cheese Platters Sold']
    x,y,z,q, v = stats.linregress(x_departure, y_sold)
    print(x,y,z,q,v)
    plt.scatter(x_departure, y_sold, alpha=.5, edgecolor="none")
    plt.show()

#scatter_plots()


def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R = la.qr(A,mode="economic")
    return la.solve_triangular(R,Q.T@b)

#Starting to create the matrices for the least squares solution without the stock outs factored in
A = df[df['Stock Out Occurred'] == 0]
B = df[df['Stock Out Occurred'] == 1]

b = np.array(A['Cheese Platters Sold']).T

depart = np.array(A['Dptr Hour']).T
length = np.array(A['Length of Flight (Hrs)']).T
boarded = np.array(A['Passengers Boarded']).T
DtD = np.array(A['Passengers Booked 2 DtD']).T
full_array = np.column_stack((depart,length,boarded,DtD))

x_hat = least_squares(full_array,b)

day = np.array(A['Day of Week']).T
day_array = np.column_stack((day,b))
monday_array = A[A['Day of Week'] == 'Monday']

#Now create a matrix with the stock outs factored in
b_hat = df['Cheese Platters Sold']
depart = np.array(df['Dptr Hour']).T
stock_out = np.array(df['Stock Out Occurred'])
length = np.array(df['Length of Flight (Hrs)']).T
boarded = np.array(df['Passengers Boarded']).T
DtD = np.array(df['Passengers Booked 2 DtD']).T
A_hat = np.column_stack((depart,length,boarded,DtD,stock_out))
x_estimate = least_squares(A_hat,b_hat)
print(A_hat[5]@x_estimate, b_hat[5])
count = 0
for i in range(len(b_hat)):
    if abs(b_hat[i] - (A_hat[i]@x_estimate)) <= 1:
        count += 1
#print(count/len(b_hat))

def plot_days():
    n,m = np.shape(day_array)
    stock_monday = B[B['Day of Week'] == 'Monday']
    mon = sum(stock_monday['Stock Out Occurred'])
    stock_tuesday = B[B['Day of Week'] == 'Tuesday']
    tue = sum(stock_tuesday['Stock Out Occurred'])
    stock_wednesday = B[B['Day of Week'] == 'Wednesday']
    wed = sum(stock_wednesday['Stock Out Occurred'])
    stock_thursday = B[B['Day of Week'] == 'Thursday']
    thu = sum(stock_thursday['Stock Out Occurred'])
    stock_friday = B[B['Day of Week'] == 'Friday']
    fri = sum(stock_friday['Stock Out Occurred'])
    stock_saturday = B[B['Day of Week'] == 'Saturday']
    sat = sum(stock_saturday['Stock Out Occurred'])
    stock_sunday = B[B['Day of Week'] == 'Sunday']
    sun = sum(stock_sunday['Stock Out Occurred'])
    #print(wed/len(df[df['Day of Week'] == 'Wednesday']), thu/len(df[df['Day of Week'] == 'Thursday']), sat/len(df[df['Day of Week'] == 'Saturday']), sun/len(df[df['Day of Week'] == 'Sunday']))
    #print(mon/len(df[df['Day of Week'] == 'Monday']), tue/len(df[df['Day of Week'] == 'Tuesday']), fri/len(df[df['Day of Week'] == 'Friday']))

    
    sold_monday = sum(df[df['Day of Week'] == 'Monday']['Cheese Platters Sold'])/len(df[df['Day of Week'] == 'Monday'])
    sold_tuesday = sum(df[df['Day of Week'] == 'Tuesday']['Cheese Platters Sold'])/len(df[df['Day of Week'] == 'Tuesday'])
    sold_wednesday = sum(df[df['Day of Week'] == 'Wednesday']['Cheese Platters Sold'])/len(df[df['Day of Week'] == 'Wednesday'])
    sold_thursday = sum(df[df['Day of Week'] == 'Thursday']['Cheese Platters Sold'])/len(df[df['Day of Week'] == 'Thursday'])
    sold_friday = sum(df[df['Day of Week'] == 'Friday']['Cheese Platters Sold'])/len(df[df['Day of Week'] == 'Friday'])
    sold_saturday = sum(df[df['Day of Week'] == 'Saturday']['Cheese Platters Sold'])/len(df[df['Day of Week'] == 'Saturday'])
    sold_sunday = sum(df[df['Day of Week'] == 'Sunday']['Cheese Platters Sold'])/len(df[df['Day of Week'] == 'Sunday'])
    print(sold_monday, sold_tuesday, sold_wednesday, sold_thursday, sold_friday, sold_saturday, sold_sunday)

    
    
    labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    values = [mon, tue, wed, thu, fri, sat, sun]
    
    positions = np.arange(len(labels))
    plt.bar(positions, values, align='center')
    plt.xticks(positions,labels)
    plt.tight_layout()
    plt.show()
    
#plot_days()

def histogram():
    plt.hist(b,bins=16)
    plt.title("Distribution of Cheese Platters Sold")
    plt.xlabel("# of Platters Sold")
    plt.ylabel("Frequency of Occurrence")
    plt.tight_layout()
    plt.show()
#histogram()

def predict(christmas, label, target):
    #christmas = KDTree(data_set)
    distance, indices = christmas.query(target, 7)
    the_mode, the_index = stats.mode(label[indices])
    return int(the_mode[0])

depart1 = list(A['Dptr Hour'])

length1 = list(A['Length of Flight (Hrs)'])

boarded1 = list(A['Passengers Boarded'])
DtD1 = list(A['Passengers Booked 2 DtD'])
zipped = list(zip(depart1,length1,boarded1,DtD1))
b1 = np.array([A['Cheese Platters Sold'].astype(np.float)]).T
#targ = [17, 3, 165, 165]
#print(predict(zipped,b1,targ))


half1 = pd.read_csv('half1.csv')
half2 = pd.read_csv('half2.csv')

A_not = half1[half1['Stock Out Occurred'] == 0]

depart_half_1 = list(A_not['Dptr Hour'])
length_half_1 = list(A_not['Length of Flight (Hrs)'])
boarded_half_1 = list(A_not['Passengers Boarded'])
DtD_half_1 = list(A_not['Passengers Booked 2 DtD'])
zipped_half_1 = list(zip(depart_half_1, length_half_1, boarded_half_1, DtD_half_1))
b_half_1 = np.array([A_not['Cheese Platters Sold'].astype(np.float)]).T

B_not = half2[half2['Stock Out Occurred'] == 0]
depart_half_2 = list(B_not['Dptr Hour'])
length_half_2 = list(B_not['Length of Flight (Hrs)'])
boarded_half_2 = list(B_not['Passengers Boarded'])
DtD_half_2 = list(B_not['Passengers Booked 2 DtD'])
zipped_half_2 = list(zip(depart_half_2, length_half_2, boarded_half_2, DtD_half_2))
b_half_2 = list(B_not['Cheese Platters Sold'])

christmas_tree = KDTree(zipped_half_1)
targey = [14,1,139,126]
#print(predict(christmas_tree, b_half_1, targey))
new_predict = []
for i in zipped_half_2:
    new_predict.append(predict(christmas_tree, b_half_1, i))
counter = 0
for j in range(len(new_predict)):
    if new_predict[j] == b_half_2[j]:
        counter += 1
        
percent = counter/len(new_predict)
print(percent)

stocked_out = []
B_stock_out = half2[half2['Stock Out Occurred'] == 1]
board_out = list(B_stock_out['Passengers Boarded'])
depart_out = list(B_stock_out['Dptr Hour'])
length_out = list(B_stock_out['Length of Flight (Hrs)'])
DtD_out = list(B_stock_out['Passengers Booked 2 DtD'])
b_out = list(B_stock_out['Cheese Platters Sold'])
zipped_out = list(zip(depart_out, length_out, board_out, DtD_out))

for k in zipped_out:
    stocked_out.append(predict(christmas_tree, b_half_1, k))
stock_count = 0
for l in range(len(stocked_out)):
    if stocked_out[l] >= b_out[l]:
        stock_count += 1

percent_out = stock_count/len(stocked_out)
print(percent_out)


