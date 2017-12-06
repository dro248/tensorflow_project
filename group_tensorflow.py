from create_sentiment_featuresets import create_feature_sets_and_labels
import tflearn
import numpy as np 

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=10, batch_size=100, show_metric=True)

# calculate predictions
predictions = model.predict(test_x)

# round predictions
rounded = np.around(predictions)
numcorrect = 0

for x, y in zip(rounded, test_y):
	if((x == y).all()):
		numcorrect = numcorrect + 1

print("Test set percentage correct = ", float(numcorrect) / len(test_x))
