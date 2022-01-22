import argparse
import os
import shutil
from tabnanny import verbose
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np

STAGE = "creating base model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    #params = read_yaml(params_path)
    # get data 
    (x_train_full,y_train_full) ,(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    x_train_full = x_train_full/255.0
    x_test = x_test/255.0
    x_valid,x_train,= x_train_full[:5000],x_train_full[5000:]
    y_valid,y_train,= y_train_full[:5000],y_train_full[5000:]

    #seed
    seed = 2022
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## define layers
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28,28]),
        tf.keras.layers.Dense(300,name="h1_layers"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(100,name="h2_layers"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(10,activation="softmax",name = "outputlayer")
    ]
    ## compile 
    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]
    model = tf.keras.models.Sequential(LAYERS)

    model.compile(loss=LOSS,optimizer = OPTIMIZER, metrics = METRICS)
    model.summary()

    history = model.fit(x_train,y_train,
    epochs=10,
    validation_data =(x_valid,y_valid),
    verbose=2)

    model_dir_path = os.path.join("artifacts","models")
    create_directories([model_dir_path])
    model_file_path = os.path.join(model_dir_path,"base_model.h5")
    model.save(model_file_path)
    logging.info(f"base model save dir {model_file_path}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    #args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e