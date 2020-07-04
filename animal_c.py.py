import os
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import  Flatten
import numpy as np
from PIL import Image
from skimage import transform
from keras.applications import ResNet50V2


base_dir='drive/derin/all_animals'      #elimizdeki resimleri numpy dizisine çevireceğiz
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'drive/derin/dene/goat.1816.jpeg')

model = Sequential()
conv_base=ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling=max,
    classes=7,
)
model.add(conv_base)
model.add(Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096,activation="relu"))
model.add(layers.Dense(4096,activation="relu"))
model.add(layers.Dense(7,activation="softmax"))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss="categorical_crossentropy",
              metrics=["categorical_accuracy"])

datagen=ImageDataGenerator(rescale=1./255)

train_datagen=ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',)
validation_datagen=ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',)



train_generator=train_datagen.flow_from_directory(
        train_dir,
        color_mode="rgb",
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')
validation_generator=validation_datagen.flow_from_directory(
        validation_dir,
        color_mode="rgb",
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')

test_datagen = ImageDataGenerator(
   rescale=1./255
)


history=model.fit_generator(train_generator,
                  steps_per_epoch=100,
                  epochs=50,
                  validation_data=validation_generator,
                  validation_steps=50)

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('drive/derin/dene/goat.1816.jpeg')
model.predict(image)


acc=history["acc"]
val_acc=history["val_acc"]
loss=history["loss"]
val_loss=history["val_loss"]

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,"bo",label="Eğitim başarımı")
plt.plot(epochs,val_acc,"b",label="Doğrulama başarımı")
plt.title("Eğitim ve Doğrulama başarımı")
plt.legend()

plt.figure()

plt.plot(epochs,loss,"bo",label="Eğitim kaybı")
plt.plot(epochs,val_loss,"b",label="Doğrulama kaybı")
plt.title("Eğitim ve Doğrulama kaybı")
plt.legend()

plt.legend()
plt.show()