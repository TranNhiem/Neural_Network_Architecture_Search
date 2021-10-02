'''To runing this model 
1. Coppy thecallback List
2. Coppy the weight training 256, 256 model 
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from dataset import AOI
from callback_list import *
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def B0_model(input_shape, num_class):
    
    '''1.Get the Baseline MODEL'''
    base_line=tf.keras.applications.EfficientNetB0(include_top=False,weights=None,
    input_shape=input_shape,
    )

    # Enable to train the whole 
    base_line.trainable = True

    last_layer= base_line.layers[-1].output    
    x = tf.keras.layers.GlobalAveragePooling2D()(last_layer)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs =  tf.keras.layers.Dense(num_class, activation='softmax')(x)


    model = tf.keras.Model(base_line.input, outputs)
    #Set all layers,except the last one to not trainable
    for layer in model.layers: 
        layer.trainable=True
    
    return model
        

def compile_train_model(model, compile_kwargs={},): 

    #compile_args.update(compile_kwargs)
    loss=compile_kwargs["loss"]
    optimizer=compile_kwargs["optimizer"]
    metrics=compile_kwargs["metrics"]
    initial_lr =compile_kwargs["initial_lr"]
    momentum=compile_kwargs["momentum"]

    if optimizer=="Adam":
        opt=tf.keras.optimizers.Adam(learning_rate=initial_lr)
    elif optimizer=="RMSprop":
        opt=tf.keras.optimizers.RMSprop(learning_rate=initial_lr, momentum=momentum)
    elif optimizer=="SGD":
        opt=tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=momentum)
    else: 
        print('your custom Optimizer')

    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    print("model_custom_quantize_configure Finished compiling")
    return model


def main(): 
    img_width=120
    img_height= 120

    dataset= AOI(img_height, img_width, validation_split=0.2)

    # if dataset.input_shape == (256, 256, 3): 
    #     model= B0_model(input_shape=dataset.input_shape,num_class=dataset.num_classes)


    #     compile_kwargs = {
    #         "optimizer": "RMSprop",  # RMSprop, SGD, Adam
    #         "loss": "categorical_crossentropy",
    #         "metrics": ["accuracy"],
    #         "initial_lr": 1e-2,
    #         "momentum": 0.9}

    #     model=compile_train_model(model,compile_kwargs= compile_kwargs)
    #     model.load_weights("B0_Mixup_update.h5")
    #     model.save_weights("B0_AOI_weight_reference.h5")
    #     model.summary()

    # else: 
        
    model= B0_model(input_shape=dataset.input_shape,num_class= dataset.num_classes)

    compile_kwargs = {
        "optimizer": "RMSprop",  # RMSprop, SGD, Adam
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "initial_lr": 1e-2,
        "momentum": 0.9}

    model=compile_train_model(model,compile_kwargs= compile_kwargs)

    batch_size = 45
    train_data = dataset.train_dataset() \
        .shuffle(8 * batch_size) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    valid_data = dataset.validation_dataset() \
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


    add_kwargs={
        "EPOCHS": 200, 
        "monitor": "val_loss", 
        "patience_stop": 40,# epochs
        "reducelr_patience": 10, #epochs
        "min_lr": 1e-7, 
        "checkpoint_period": 3, 
        "checkpoint_name": "B0_AOI_weight_reference.h5",# original "B0_Mixup_update.h5", 99.2 test--prune_check_v 99.6/ 99.7
        #result 28-3 almost the same 28-2
        "log_path": "./BO_AOI", 
        "callback_list": [0, 5,6],#[checkpoint, earlystop,reducelr, lr_cosine_annealing ,schedule_lr, tensorboard, lr_cosine_annealing2]
    }
    '''Note model during experiment

    ''''
    #callbacks_list, name_list=callback_func(add_kwargs, pruning=False,)
    #print(name_list)
    def lr_schedule(epoch):
        if 0 <= epoch < 35:
            return 0.1
        if 35 <= epoch < 65:
            return 0.01
        return 0.001
    
    model.fit(train_data, validation_data=valid_data, callbacks=[LearningRateScheduler(lr_schedule)], epochs=200)
    model.save_weights('B0_AOI_weight_reference.h5')
    #model.summary()
       
if __name__ == '__main__':
    main()