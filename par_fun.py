import pandas as pd
import numpy as np
from random import seed
from random import random

df = pd.DataFrame(np.random.randint(0,100000,100000), columns=list('x'))
df['y'] = df['x'].apply(lambda x: 1 if x%2 == 0 else 0)

from keras.layers import Dense, Input, ReLU, AlphaDropout, Activation, LSTM
from keras.models import Model
import keras
from keras.preprocessing.sequence import pad_sequences

X = []
# for row in df['x'].values:
#     x = pad_sequences(df['x'].apply(lambda x : str(x)),maxlen=6, padding='post', value=0)
#     X.append(x)
# X = np.array(X)

inp = Input(shape=(1,))
mod = LSTM(1024)(inp)
#mod = Activation('selu')(mod)
mod = LSTM(256)(mod)
mod = Activation('selu')(mod)
out = Dense(1, activation = 'sigmoid')(mod)

model = Model(inputs=inp, outputs=out)

history = model.compile(optimizer=keras.optimizers.SGD(lr=0.0001),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'],
                  )

model.fit(x = df['x'], y = df['y'], validation_split= 0.2, epochs=100)