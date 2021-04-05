
## Usual Imports
import numpy as np
# import matplotlib.pyplot as plt
# import re
import json
import datetime
import string
import gc

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.sequence import pad_sequences




def load_data(filename, load_func=json.load):

    """
    Load the data from json file, 
    Returns a pandas dataframe containing the data.
    """

    with open(filename) as f:
        d = np.asarray(load_func(f))

    # shuffle the array to randomize the data sets used.
    # Needed in this case as the file has true cases all before
    # the false ones. Seed the random for consistent results
    np.random.seed(42)
    np.random.shuffle(d)

    print(f"Loaded {len(d)} data records.")

    return d



def tokenize_sentences(x_train, x_val, x_test, max_len=100):

    # Defaults of the tokenizer:
    # set to lower case and split on spaces.
    # No restriction on length of vocabulary
    #
    # Overriding the default filters with string.punctuation
    # as the default does not strip apostrophies which is
    # expected in the GloVe tokenizer.

    t = keras.preprocessing.text.Tokenizer(filters=string.punctuation)

    # create vocabulary from all the words in x_train
    t.fit_on_texts(x_train)

    return( pad_sequences(t.texts_to_sequences(x_train), max_len, padding='post', truncating = 'post'),
            pad_sequences(t.texts_to_sequences(x_val), max_len, padding='post', truncating = 'post'),
            pad_sequences(t.texts_to_sequences(x_test), max_len, padding='post', truncating = 'post'),
            t )


def embed_matrix(t, embed_dim, embed_loc = "/mnt/export/NLPData", embed_file = "glove.6B.50d.txt"):

    """
    Creates a vocabulary list and corresponding embedding matrix such that 
    embedding_matrix[i] is the representation of word at vocab_list[i]
    as required by the embedding layer which takes in the IDs (value of i) for
    each word in the sentence.

    t is the tokenizer that has already been fit to the texts in the training data
    """

    vocab_list = list(t.word_index.keys())
    vocab_size = len(vocab_list) + 1  # to allow for the Zero unknown token.
    embedding_matrix = np.zeros( (vocab_size, embed_dim) )

    with open(embed_loc + "/" + embed_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]

            if word in vocab_list:
                embedding_matrix[vocab_list.index(word)] = np.asarray(values[1:], "float32")   
            
    return(vocab_list, embedding_matrix, vocab_size)      





def create_model(embedding_matrix,
                 max_len,
                 num_filters,
                 kernel_sizes,
                 dense_layer_dims,
                 dropout_rate,
                 train_embeds,
                 opt = 'adam'
                 ):

    """
    Create the CNN keras model based on passed parameters
    Returns the keras model
    """

    # This is a 2-class problem, so only need a single binary class
    num_classes = 1
    embed_dim = embedding_matrix.shape[1]  # dimensions used in the embedding.
    vocab_size = embedding_matrix.shape[0]   # number of words (including UNK) in the vocab

    # set up input layer (receives word IDs) and embedding that turns that into GloVe embeddings
    word_ids = keras.layers.Input(shape=(max_len,))
    h=keras.layers.Embedding(vocab_size,
                             embed_dim,
                             weights=[embedding_matrix],
                             trainable = train_embeds)(word_ids)

    # Add convolutional layers and pooling layers based on number of filters and kernel size(s)
    conv_layers_for_all_kernel_sizes = []
    for kernel_size, filters in zip(kernel_sizes, num_filters):
        print(f"Adding Convolution: Kernel Size: {kernel_size}, Filter Count: {filters}")
        # note that all convolution layers take the same input "h" the output from the embedding layer
        conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(h)
        conv_layer = keras.layers.GlobalMaxPooling1D()(conv_layer)
        conv_layers_for_all_kernel_sizes.append(conv_layer)

    # Concat the feature maps from each different size.
    h = keras.layers.concatenate(conv_layers_for_all_kernel_sizes, axis=1)

    # Dropout can help with overfitting
    h = keras.layers.Dropout(rate=dropout_rate)(h)

    # Add the fully connected feed forward layers for categorization
    # Add a fully connected layer for each dense layer dimension in dense_layer_dims.
    for dim in dense_layer_dims:
        h = keras.layers.Dense(dim, activation='relu')(h)

    # Add the output layer for classifier - in this case, there is only one output
    prediction = keras.layers.Dense(num_classes, activation='sigmoid')(h)

    # Create and compile the model
    model = keras.Model(inputs=word_ids, outputs=prediction)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',               # as we only have a single output class
                  # metrics=['binary_accuracy'])                    # What metric to output as we train.
                  metrics=['accuracy'])                    # What metric to output as we train.
    model.reset_states()


    return(model)






def train_model(model,
                x_train_ids,
                y_train,
                x_val_ids,
                y_val,
                logdir = "tb_dir/",
                epochs = 30,
                batch_size = 50,
                callbacks = []):

    """
    Train the passed model using the training and validation sets.
    If a logfile is specified, append the history to the file
    in pipe delimited format
    """

    model.fit(x_train_ids, y_train,
              epochs=epochs,
              batch_size = batch_size,
              validation_data = (x_val_ids, y_val),
              callbacks = callbacks   )

    hist = model.history.history
    
    return(hist)








# Defaults match the best values from optimization 1
def run_model(embedding_matrix,
              x_train_ids,
              y_train,
              x_val_ids,
              y_val,
              max_len = 100,        # set at top of notebook
              epochs = 5,
              batch_size = 50,
              embed_dim = 50,
              num_filters = [8,16,32],
              kernel_sizes = [2, 3, 4],
              dense_layer_dims = [8],
              dropout_rate = 0.2,
              train_embeds = True,  # Whether we allow the embeddings to be changed
              opt = 'adam', 
              logfile = None,
              logdir = "tb_dir/",
              logtag = "tblogs",
              model_dir = "../models",
              ):


    
    tag  = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tblog_dir = f"{logtag}-{kernel_sizes}/".replace("[","").replace("]","").replace(", ","") + \
            datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    callbacks = [ keras.callbacks.TensorBoard(log_dir=tblog_dir, histogram_freq=1), 
                  keras.callbacks.ModelCheckpoint(filepath=model_dir + "/" + tag,
                                             monitor='val_loss',
                                             mode='min',
                                             save_best_only=True) ]


    model = create_model(embedding_matrix = embedding_matrix,
                            max_len = max_len,
                            num_filters = num_filters,
                            kernel_sizes = kernel_sizes,
                            dense_layer_dims = dense_layer_dims,
                            dropout_rate = dropout_rate,
                            train_embeds = train_embeds,
                            opt = opt)

    
    hist = train_model(model,
                       x_train_ids = x_train_ids,
                       y_train = y_train,
                       x_val_ids = x_val_ids,
                       y_val = y_val,
                       logdir = logdir,
                       epochs = epochs,
                       batch_size = batch_size,
                       callbacks = callbacks)

    if logfile is not None:


        with open(logfile, 'a') as f:
            f.write(f"{tag}|{max_len}|{epochs}|{batch_size}|{embed_dim}|{num_filters}|{kernel_sizes}|{dense_layer_dims}|{dropout_rate}|")
            for metric in list(hist.keys()):
                # print(metric)
                f.write(f"{metric}|")
                for i in range(0,epochs):
                    f.write(f"{hist[metric][i]}|")
            f.write(f"END\n")        
            f.close()

    # Destroy the model to free up GPU memory for the next run

    del model
    del hist
    gc.collect()
    
    return(None)

