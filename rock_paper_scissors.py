"""
Tittle:rock_paper_scissors

This is the website of dateset:
https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip

Learn something from:
https://blog.csdn.net/weixin_36167031/article/details/114420199
https://blog.csdn.net/wanghuiqiang1/article/details/113323586

"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

rock_dir = os.path.join('D:\\ML\\rps\\rps\\rock')
paper_dir = os.path.join('D:\\ML\\rps\\rps\\paper')
scissors_dir = os.path.join('D:\\ML\\rps\\rps\\scissors')

# print the number of training image
print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

# print names of training image
rock_files = os.listdir(rock_dir)
print(rock_files[:10])
paper_files = os.listdir(paper_dir)
print(paper_files[:10])
scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

TRAINING_DIR = "D:\\ML\\rps\\rps"
training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

VALIDATION_DIR = "D:\\ML\\rps-test-set\\rps-test-set"
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                       target_size=(150, 150),
                                                       class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(150, 150),
                                                              class_mode='categorical')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(160, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
# GD，SGD，Momentum，RMSProp，Adam
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_generator,
                              epochs=15,
                              validation_data=validation_generator,
                              verbose=1)
model.save("D:\\ML\\rps.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)
with summary_writer.as_default():
    for n in enumerate(loss):
        tf.summary.scalar('loss', loss[n], step=n)
        tf.summary.scalar('acc', acc[n], step=n)
    tf.summary.image("trainpicture", train_generator[1], step=0)
# tensorboard --logdir logs

print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# use the model for a picture
new_model = tf.keras.models.load_model('D:\\ML\\rps.h5')
path = 'D:\\ML\\rps-try-set\\1.jpg'
img = image.load_img(path, target_size=(150, 150))  # list
x = image.img_to_array(img)  # array
x = np.expand_dims(x, axis=0)  # tensor
images = np.vstack([x])
classes = new_model.predict(images, batch_size=10)
print(classes)

# use the model for pictures
try_dir = os.path.join('D:\\python project\\tensorflow rock_paper_scissors\\rps-try-set')
try_files = os.listdir(try_dir)
uploaded = try_files.upload()
for fn in uploaded.key():
    path = fn
    img = img.load_img(path, target_size=(150, 150)) #列表
    x = image.img_to_array(img) #数组
    x = np.expand_dims(x, axis=0) #向量
    images = np.vstack([x]) #3个通道连起来，形成一个长向量
    classes = new_model.predict(images, batch_size=10)
    print(fn)
    print(classes)

# use the file of pictures
def is_image_file(filename):
    """Is it a picture?"""
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg',
                                                              '.PNG', '.JPG', '.JPEG'])


dataset_dir = 'D:\\python project\\tensorflow rock_paper_scissors\\rps-try-set'
image_filenames = [os.path.join(dataset_dir, x)
                   for x in os.listdir(dataset_dir) if is_image_file(x)]

for image_filename in image_filenames:
    print(image_filename)
