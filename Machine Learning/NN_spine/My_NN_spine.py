import numpy as np

# Load data set and code labels as 0 = ’NO’, 1 = ’DH’, 2 = ’SL’
labels = [b'NO', b'DH', b'SL']
data = np.loadtxt('column_3C.dat', converters={6: lambda s: labels.index(s)} )

# Separate features from labels
x = data[:,0:6]
y = data[:,6]

# Divide into training and test set
training_indices = list(range(0,20)) + list(range(40,188)) + list(range(230,310))
test_indices = list(range(20,40)) + list(range(188,230))

trainx = x[training_indices,:]
trainy = y[training_indices]
testx = x[test_indices,:]
testy = y[test_indices]

def get_dist(x,y):
    dist = np.sum(np.square(x-y))
    return dist

def get_NN(x):
    return np.argmin([get_dist(trainx[i,],x) for i in range(len(trainy))])

def get_NN_ind(x):
    return  trainy[get_NN(x)]

def NN_L2(trainx, trainy, testx):
    predictions = [get_NN_ind(testx[i,]) for i in range(len(testy))]
    return np.array(predictions)


testy_L2 = NN_L2(trainx, trainy, testx)

assert( type( testy_L2).__name__ == 'ndarray' )
assert( len(testy_L2) == 62 ) 
assert( np.all( testy_L2[50:60] == [ 0.,  0.,  0.,  0.,  2.,  0.,  2.,  0.,  0.,  0.] )  )
assert( np.all( testy_L2[0:10] == [ 0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.] ) )


def get_dist1(x,y):
    dist = np.sum(np.abs((x-y)))
    return dist

def get_NN1(x):
    return np.argmin([get_dist1(trainx[i,],x) for i in range(len(trainy))])

def get_NN_ind1(x):
    return  trainy[get_NN1(x)]

def NN_L1(trainx, trainy, testx):
    predictions = [get_NN_ind1(testx[i,]) for i in range(len(testy))]
    return np.array(predictions)

testy_L1 = NN_L1(trainx, trainy, testx)

assert( type( testy_L1).__name__ == 'ndarray' )
assert( len(testy_L1) == 62 ) 
assert( not all(testy_L1 == testy_L2) )
assert( all(testy_L1[50:60]== [ 0.,  2.,  1.,  0.,  2.,  0.,  0.,  0.,  0.,  0.]) )
assert( all( testy_L1[0:10] == [ 0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.]) )

def error_rate(testy, testy_fit):
    return float(sum(testy!=testy_fit))/len(testy) 

print("Error rate of NN_L1: ", error_rate(testy,testy_L1) )
print("Error rate of NN_L2: ", error_rate(testy,testy_L2) )


def confusion(testy,testy_fit):
    conf=np.zeros((3, 3))
    for i in range(len(testy)):
        if testy_fit[i]==testy[i]:
            conf[int(testy[i]),int(testy[i])]+=1.
        else:
            conf[int(testy[i]),int(testy_fit[i])]+=1.
    return conf
            

L1_neo = confusion(testy, testy_L1) 
assert( type(L1_neo).__name__ == 'ndarray' )
assert( L1_neo.shape == (3,3) )
assert( np.all(L1_neo == [[ 16.,  2.,  2.],[ 10.,  10.,  0.],[ 0.,  0.,  22.]]) )
L2_neo = confusion(testy, testy_L2)  
assert( np.all(L2_neo == [[ 17.,  1.,  2.],[ 10.,  10.,  0.],[ 0.,  0.,  22.]]) )

result = 0
for i in range(len(testy_L1)):
    if testy_L1[i]!=testy_L2[i]:
        result += 1
result