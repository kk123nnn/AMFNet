import arff
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



dimensions = 6
data = []
for i in range(1, dimensions + 1):
    filename = 'RacketSportsDimension' + str(i) + '_TRAIN.arff'
    file = open(filename, "r")

    # load each dimension
    dataset = arff.load(file)
    dataset = np.array(dataset['data'])
    
    # obtain all row and column except the last column
    data.append(dataset[ : , 0 : -1])
data = np.array(data)

# adjust the dimension sample_size * seq_size * variable_num
data = np.transpose(data, (1, 2, 0))
print(data.shape)

# save as .npy file
np.save('X_train.npy', data)

# only obtain the last column
label = np.array(dataset[ : , -1])

# transform and fit
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(label)

print(integer_encoded.shape)
np.save('y_train.npy', integer_encoded)