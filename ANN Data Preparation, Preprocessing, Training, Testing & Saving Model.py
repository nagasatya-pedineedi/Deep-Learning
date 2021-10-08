#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


# #### Train Data

# In[2]:


train_labels = []
train_samples = []


# Example data
# - An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial
# - The trial had 2100 participants. Half we under 65 years and half were 65 years or older
# - 95% participants 65 or older experienced side effects
# - 95% participants under 65 experienced no side effects

# In[3]:


for i in range(50):
    # 5% of the younder individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    # 5% of the older individuals who experienced side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # 95% of the younger individuals who experienced side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    # 95% of the younger individuals who experienced side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)


# Converting lists into numpy array to use data for training neural networks

# In[4]:


train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)


# In[5]:


scaler = MinMaxScaler()
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
scaled_train_samples


# In[6]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from keras.optimizer_v1 import Adam
from tensorflow.keras.metrics import categorical_crossentropy


# In[7]:


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# # tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[8]:


model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])


# In[9]:


model.summary()


# In[10]:


opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[11]:


model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, shuffle=True, verbose=2)


# Adding a validation split parameter 

# In[12]:


model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)


# #### Test Data

# In[13]:


test_labels = []
test_samples = []


# In[14]:


for i in range(10):
    # 5% of the younder individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    # 5% of the older individuals who experienced side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # 95% of the younger individuals who experienced side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    # 95% of the younger individuals who experienced side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)


# In[15]:


test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)


# In[16]:


scaled_test_samples = scaler.transform(test_samples.reshape(-1,1))


# #### Predict

# In[17]:


predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)


# In[18]:


for i in predictions:
    print(i)


# In[19]:


rounded_predictions = np.argmax(predictions, axis = 1)


# In[20]:


for i in rounded_predictions:
    print(i)


# Confusion Matrix plotting

# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


# In[22]:


cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
cm


# In[44]:


def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion Matrix',
                         cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
#     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalised confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0], cm.shape[1])):
        plt.text(j, i, cm[i,j],
                 horizontalalognment='centre',
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[45]:


cm_plot_labels = ['no_side_effects', 'had side effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix') 


# ### Saving the model

# In[27]:


import os.path
if os.path.isfile('models/medical_trail_model.h5') is False:
    model.save('models/medical_trail_model.h5')


# This save function saves:
# - The architecture of the model, allowing to re-create the model
# - The weights of the model
# - The training configuration (loss, optimizer)
# - The state of the optimizer, allowing to resume training exactly where you left off

# ### Loading the model

# In[28]:


from tensorflow.keras.models import load_model
new_model = load_model('models/medical_trail_model.h5')


# In[29]:


new_model.summary()


# In[30]:


new_model.get_weights()


# In[31]:


new_model.optimizer


# ### model.to_json()

# If you only need to save the architecture of a model and not it's wight or it's configuration, you can use the following function to save the architecture only

# In[32]:


# Save as json
json_string = model.to_json()

# save as YAML
# yaml_string = model.to_yaml()


# In[33]:


json_string


# In[34]:


# model reconstruction from JSON:
from tensorflow.keras.models import model_from_json
model_architecture = model_from_json(json_string)

# model reconstruction from JSON
# from tensorflow.keras.models import model_from_yaml
# model = model_from_yaml(yaml_string)


# In[36]:


model_architecture.summary()


# ### model.save_wights()
# If only you need to save the wights of the model, you can use the following function

# In[40]:


if os.path.isfile('models/my_model_weights.h5') is False:
    model.save_weights('models/my_model_weights.h5')


# In[41]:


model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])


# In[42]:


model2.load_weights('models/my_model_weights.h5')


# In[43]:


model2.get_weights()


# In[ ]:




