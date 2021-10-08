#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
            )

img = load_img("D:\Deep Learning\me.jpg")
x = img_to_array(img)  # This is a numpy array with the shape (3, 150, 150)
x = x.reshape((1, ) + x.shape)  # This is a numpy array with shape of (1, 3, 150, 150)

# flow() coomand below generates batches of randomly transformed images 
# and saves the resluts to the preview directory
i=0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='me', save_format='jpg'):
    i += 1
    if i > 20:
        break #otherwise the generator would loop infinitely


# In[ ]:




