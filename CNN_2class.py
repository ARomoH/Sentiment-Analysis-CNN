import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Merge, Dense, Activation
from keras.layers.pooling import GlobalMaxPooling1D
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.layers import advanced_activations
from keras import initializers
from keras import losses


import time

        

################## PARAMETERS ##################

size_filters = (2, 3, 4, 5)
size_batch = 500
num_filters = 300
n_iter = 150 
dropout = 0.5
alpha = 3.0

################################################
cont_num = 0
cont_words = 0
cont_tweet = 0

n_tweets_train = 3496
n_tweets_test = 1543
tam_fijo = 10
embedding_vecor_length = 300

# Read file with train embeddings
taggs_train = []
X_train = np.zeros((n_tweets_train, tam_fijo, embedding_vecor_length))

with open('data/Train1_x.txt','r') as f_x, open('data/Train1_y.txt','r') as f_y:
    for line, line_y in zip(f_x, f_y):
        if line_y.strip() == 'P':
            taggs_train.append(1)
            elem = line.split()
            if len(elem) > embedding_vecor_length*tam_fijo:
                elem = line.split()[0:(embedding_vecor_length*tam_fijo)]
            for num in elem:
                if cont_num == embedding_vecor_length:
                    cont_words += 1
                    cont_num = 0
                # print(str(cont_tweet) + ' ' + str(cont_words) + ' ' + str(cont_num))
                X_train[cont_tweet][cont_words][cont_num] = num
                cont_num += 1
            cont_words = 0
            cont_num = 0
            cont_tweet += 1
        elif line_y.strip() == 'N':
            taggs_train.append(0)
            elem = line.split()
            if len(elem) > embedding_vecor_length*tam_fijo:
                elem = line.split()[0:(embedding_vecor_length*tam_fijo)]
            for num in elem:
                if cont_num == embedding_vecor_length:
                    cont_words += 1
                    cont_num = 0
                # print(str(cont_tweet) + ' ' + str(cont_words) + ' ' + str(cont_num))
                X_train[cont_tweet][cont_words][cont_num] = num
                cont_num += 1
            cont_words = 0
            cont_num = 0
            cont_tweet += 1

y_train = np.asarray(taggs_train)


# Read file with test embeddings
cont_tweet = 0
cont_num = 0
cont_words = 0
cont_tweet = 0
taggs_test = []
X_test = np.zeros((n_tweets_test, tam_fijo, embedding_vecor_length))


with open('data/Test1_x.txt','r') as f_x, open('data/Test1_y.txt','r') as f_y:
    for line, line_y in zip(f_x, f_y):
        if line_y.strip() == 'P':
            taggs_test.append(1)
            elem = line.split()
            if len(elem) > embedding_vecor_length*tam_fijo:
                elem = line.split()[0:(embedding_vecor_length*tam_fijo)]
            for num in elem:
                if cont_num == embedding_vecor_length:
                    cont_words += 1
                    cont_num = 0
                X_test[cont_tweet][cont_words][cont_num]= num
                cont_num += 1
            cont_words = 0
            cont_num = 0
            cont_tweet += 1
        elif line_y.strip() == 'N':
            taggs_test.append(0)
            elem = line.split()
            if len(elem) > embedding_vecor_length*tam_fijo:
                elem = line.split()[0:(embedding_vecor_length*tam_fijo)]
            for num in elem:
                if cont_num == embedding_vecor_length:
                    cont_words += 1
                    cont_num = 0
                # print(str(cont_tweet) + ' ' + str(cont_words) + ' ' + str(cont_num))
                X_test[cont_tweet][cont_words][cont_num]= num
                cont_num += 1
            cont_words = 0
            cont_num = 0
            cont_tweet += 1



y_test = np.asarray(taggs_test)

y_train = to_categorical(y_train, num_classes=None)
y_test = to_categorical(y_test, num_classes=None)

# Convolutional model
submodels = []
for kw in size_filters:
	submodel = Sequential()
	submodel.add(Conv1D(num_filters, kw, padding='valid', kernel_initializer=initializers.RandomNormal(np.sqrt(2/kw)) ,input_shape=(tam_fijo, embedding_vecor_length)))
	submodel.add(advanced_activations.PReLU(initializers.Constant(value=0.25)))
	submodel.add(GlobalMaxPooling1D())
	submodels.append(submodel)

model = Sequential()
model.add(Merge(submodels, mode="concat"))	
model.add(Dropout(dropout))
model.add(Dense(2,activation='softmax'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs22', write_graph=True)
adadelta = optimizers.Adadelta(lr=alpha)
model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy','mse'])

model.fit([X_train]*len(size_filters), y_train, epochs=n_iter, callbacks=[tensorBoardCallback], batch_size=size_batch, validation_data=([X_test]*len(size_filters), y_test))

# Evaluation on the test set
scores = model.evaluate([X_test]*len(size_filters), y_test, verbose=0)
print("---- TEST EVALUATION ----")
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Mean Square Error: %.2f" % (scores[2]))



# serialize model to JSON
model_json = model.to_json()
with open("./models/arquitect1_3class.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./models/arquitect1_3class.h5")
print("Saved model to disk")

print("---- MODEL SUMMARIZE ----")
print(model.summary())


### Evaluate model later ###
## load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
#
## evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
