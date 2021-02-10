from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import tensorflow_datasets as tfds

import tensorflow_hub as h

# Split the training set into 60%(train[:60%]) and 40%(train[60%:]), which will leave
# us with 15,000 example for training, 10,000 for validation and 25,000 for testing

train_data, validation_data, test_data = tfds.load(name='imdb_reviews',
                                                    split=('train[:60%]', 'train[60%:]', 'test'),
                                                    as_supervised=True)


train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)));


'''
The neural network is created by stacking layers—this requires three main architectural decisions:

How to represent the text.
How many layers to use in the model.
How many hidden units to use for each layer.
'''

'''
One way to represent the text is to convert sentences into embeddings vectors. We can use a pre-trained text embedding as the first layer, which will have three advantages:

we don't have to worry about text preprocessing,
we can benefit from transfer learning,
the embedding has a fixed size, so it's simpler to process.
'''
#====================================================================================
#Pre-trained Model #1 : https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1
# Accuracy 0.853
# Loss: 0.320
#====================================================================================

embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"

hub_layer = hub.KerasLayer(embedding, input_shape=[],
                            dtype=tf.string, trainable=True)


hub_layer(train_examples_batch[:3])

#Let's now build the full model:
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1)) # We have one hidden unit in the last layer as we are only interested in classifying a single class; the "Movie Rating"
#Summary of the model in table form
model.summary()


# Compile the model
model.compile(optimizer='adam',
              loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics=['accuracy']) 


# Train the model
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs = 5,
                    validation_data = validation_data.batch(512),
                    verbose = 1)


# Evaluate Model
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

