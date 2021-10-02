'''This Case For Implement to searching for the lightweight Solution model for AOI classification Task'''

import tensorflow as tf
import numpy as np
from .dataset import Dataset
from .utils import with_probability, random_shift, random_rotate
from numpy import load


## Optional for RandAug -- Searching Model 
from imgaug import augmenters as iaa
import imgaug as ia
tf.random.set_seed(54)
ia.seed(42)
from sklearn.model_selection import train_test_split
#AUTO = tf.data.AUTOTUNE
AUTO= tf.data.experimental.AUTOTUNE

class AOI(Dataset):
    def __init__(self, img_height, img_width, validation_split=0.15, seed=0, binary=False, ):#split for val and test data
        self.binary = binary
        self.img_height=img_height
        self.img_width= img_width 

        x_train=load('train_x_int32_mNAS.npy')
        y_train=load('train_y_int32_mNAS.npy')
        x_test=load('valx_int32_mNAS.npy')
        y_test=load('valy_int32_mNAS.npy')
        

        # Preprocessing
        # def preprocess(x, y):
        #     x = x.astype('float32') / 255
        #     x = (x - np.array((0.4914, 0.4822, 0.4465))) / np.array((0.2023, 0.1994, 0.2010))
        #     if binary:
        #         y = (y < 5).astype(np.uint8)
        #     return x, y

        # x_train, y_train = preprocess(x_train, y_train)
        ##Processing and Normalize Data
        '''AattentION we might need some standardization'''

        x_train=x_train.astype(np.float32)/255
        x_val=x_test.astype(np.float32)/255
        y_val=y_test.astype(np.uint8)
        y_train=y_train.astype(np.uint8)


        # x_test, y_test, x_val, y_val = \
        #     self._train_test_split(x_val, y_val, split_size=validation_split, random_state=seed, stratify=y_train)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=validation_split, random_state=42)
        
        self.x_train=x_train
        self.y_train=y_train
        self.train = (x_train, y_train)
        self.val = (x_val, y_val)
        #self.test = preprocess(x_test, y_test)
        # x_test=np.zeros((100, self.img_height,self.img_width,3), dtype=np.float)
        # y_test=np.zeros((100,   ), dtype=np.uint8)
        self.test = (x_test, y_test)

    @staticmethod
    def _augment(x, y):
        x = tf.image.random_flip_left_right(x)

        # x = tfa.image.random_hsv_in_yiq(
        #     x,
        #     max_delta_hue=0.5,
        #     lower_saturation=0.1,
        #     upper_saturation=0.9,
        #     lower_value=0.3,
        #     upper_value=0.8)

        #x = with_probability(0.6, lambda: random_rotate(x, 0.3), lambda: x)
        x = random_shift(x, 4, 4)
        # x = tf.clip_by_value(x, 0.0, 1.0)
        return x, y


   
 

    def train_dataset(self):
     
        # train_data = tf.data.Dataset.from_tensor_slices(self.train)
        # train_data = train_data.map(AOI._augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        BATCH_SIZE=45
        rand_aug = iaa.RandAugment(n=2, m=7)
        def Randaug(images):
                #input 
                #Define randome arug

                images= tf.cast(images, dtype=np.uint8)# tf.unit8
                #images=images.astype(np.uint8)
                return rand_aug(images=images.numpy) 


        train_ds_rand = (
            tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
            # .shuffle(BATCH_SIZE * 100)
            # .batch(BATCH_SIZE)
            .map(
                lambda x, y: (tf.image.resize(x, (self.img_height, self.img_width)), y),
                num_parallel_calls=AUTO,
            )
            .map(AOI._augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # .map(
            #     lambda x, y: (tf.py_function(Randaug, [x], [tf.float32])[0], y),
            #     num_parallel_calls=AUTO,
            # )
            # .map(
            #     lambda x, y: (x/255, y),
            #     num_parallel_calls=AUTO,
                    

            .prefetch(AUTO)
        )

        
        return train_ds_rand

    def validation_dataset(self):
        # return tf.data.Dataset.from_tensor_slices(self.val)

        val_ds = (
            tf.data.Dataset.from_tensor_slices(self.val)
            #.batch(BATCH_SIZE)
            .map(
                lambda x, y: (tf.image.resize(x, (self.img_height, self.img_width)), y),
                num_parallel_calls=AUTO,
            )
            .prefetch(AUTO)
            )
        return val_ds

    def test_dataset(self):
       
        test_ds = (tf.data.Dataset.from_tensor_slices(self.test)
            # .batch(BATCH_SIZE)
            .map(
                lambda x, y: (tf.image.resize(x, (self.img_height, self.img_width)), y),
                num_parallel_calls=AUTO,
            )
            .prefetch(AUTO)
            )
        #return tf.data.Dataset.from_tensor_slices(self.test)
        return test_ds

    @property
    def num_classes(self):
        return 6 if not self.binary else 2

    @property
    def input_shape(self):
        return (self.img_height, self.img_width, 3)
