# -*- coding: utf-8 -*-
# @Time    : 15:16 2022/11/2 
# @Author  : Haohao Song
# @Email   : songhaohao2021@stu.xmu.edu.cn
# @File    : self_contained_dnn.py
import itertools
import logging

import numpy as np
import tensorflow as tf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

writer = tf.summary.create_file_writer('./files/logs/')


def get_datasets():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255., x_test / 255.
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32),
    x_train, x_test = x_train[..., None], x_test[..., None],

    identity_matrix = np.eye(N=10)
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32),
    y_train, y_test = identity_matrix[y_train], identity_matrix[y_test],

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10).batch(10)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)

    return train_ds, test_ds


def load_train_dataset(train_index):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[train_index]
    y_train = y_train[train_index]

    x_train, x_test = x_train / 255., x_test / 255.
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32),
    x_train, x_test = x_train[..., None], x_test[..., None],

    identity_matrix = np.eye(N=10)
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32),
    y_train, y_test = identity_matrix[y_train], identity_matrix[y_test],

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10).batch(10)

    return train_ds

def load_test_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255., x_test / 255.
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32),
    x_train, x_test = x_train[..., None], x_test[..., None],

    identity_matrix = np.eye(N=10)
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32),
    y_train, y_test = identity_matrix[y_train], identity_matrix[y_test],

    # train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10).batch(10)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)

    return  test_ds

class MyModel(tf.keras.Model):
    def __init__(self, name='mymodel'):
        super(MyModel, self).__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(32, 2, activation='relu', name='conv1')
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.d1 = tf.keras.layers.Dense(10, activation='relu', name='d1')
        self.d2 = tf.keras.layers.Dense(10, name='d2')

    def call(self, inputs, trianing=None, mask=None):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        return x

def get_model(train_ds):
    mymodel = MyModel(name='mymodel')
    input_shape = train_ds.element_spec[0].shape
    mymodel.build(input_shape=input_shape)
    mymodel.call(inputs=tf.keras.Input(input_shape[1:]))
    print(mymodel.summary())
    return mymodel

def get_loss():
    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='cce_loss')
    return cce_loss


def get_optimizer():
    initial_learning_rate = 0.1
    lr_scheduling = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_rate=0.96,
            decay_steps=100,
            staircase=False
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduling)

    return optimizer

def get_metric_loss():
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    return train_loss

def get_metric_acc():
    train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    return train_acc

def get_train_step():
    def train_step(x, y, model, cce_loss, optimizer, train_loss, train_acc):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = cce_loss(y_true=y, y_pred=pred)
        gradients = tape.gradient(target=loss, sources=model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_acc.update_state(y_true=y, y_pred=pred)

    return train_step


def build_train_elements(train_ds):
    mymodel = MyModel(name='mymodel')
    input_shape = train_ds.element_spec[0].shape
    mymodel.build(input_shape=input_shape)
    mymodel.call(inputs=tf.keras.Input(input_shape[1:]))
    print(mymodel.summary())

    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='cce_loss')
    initial_learning_rate = 0.1
    lr_scheduling = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_rate=0.96,
            decay_steps=100,
            staircase=False
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduling)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')

    return mymodel, cce_loss, optimizer, train_loss, train_acc


def train_step(x, y, model, cce_loss, optimizer, train_loss, train_acc):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = cce_loss(y_true=y, y_pred=pred)
    gradients = tape.gradient(target=loss, sources=model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(values=loss)
    train_acc.update_state(y_true=y, y_pred=pred)


def train(train_ds, model, cce_loss, optimizer, train_loss, train_acc):
    train_checkpoint = tf.train.Checkpoint(mymodel=model)
    iteration_idx = 0
    for x, y in itertools.cycle(train_ds):
        train_step(x, y, model, cce_loss, optimizer, train_loss, train_acc)

        iteration_idx += 1

        with writer.as_default():
            tf.summary.scalar('train/loss', train_loss.result(), step=iteration_idx)
            tf.summary.scalar('train/acc', train_acc.result(), step=iteration_idx)

        if iteration_idx % 10 == 0:
            logging.info('iteration:{}, train_loss:{:.2f}, trian_acc:{:.2f}'.format(iteration_idx, train_loss.result(), train_acc.result()))
        if iteration_idx % 10 == 0:
            train_checkpoint.save(file_prefix='./files/ckpt/')
        if iteration_idx > 100:
            break


def build_test_elements(test_ds):
    test_model = MyModel(name='mymodel')

    # warning::执行前必须先实例化
    input_shape = test_ds.element_spec[0].shape
    test_model.build(input_shape=input_shape)
    test_model.call(inputs=tf.keras.Input(input_shape[1:]))

    test_checkpoint = tf.train.Checkpoint(mymodel=test_model)
    status = test_checkpoint.restore(save_path=tf.train.latest_checkpoint(checkpoint_dir='./files/ckpt/'))
    status.assert_consumed()

    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='cce_loss')

    test_loss = tf.metrics.Mean(name='test_loss')
    test_acc = tf.metrics.CategoricalAccuracy(name='test_acc')

    test_loss.reset_states()
    test_acc.reset_states()

    return test_model, cce_loss, test_loss, test_acc

def get_test_step():
    return test_step

def test_step(x, y, model, cce_loss, test_loss, test_acc, iteration_idx):
    pred = model(inputs=x, training=False)
    loss = cce_loss(y_true=y, y_pred=pred)

    test_loss.update_state(values=loss)
    test_acc.update_state(y_true=y, y_pred=pred)

    # with writer.as_default():
    #     tf.summary.scalar('test/loss', test_loss.result(), step=iteration_idx)
    #     tf.summary.scalar('test/acc', test_acc.result(), step=iteration_idx)
    #     writer.flush()


def test(test_ds, model, cce_loss, test_loss, test_acc):
    for iteration_idx, (x, y) in enumerate(test_ds):
        test_step(x, y, model, cce_loss, test_loss, test_acc, tf.constant(value=iteration_idx, dtype=tf.int64))
    logging.info('test_acc:{:.2f}'.format(test_acc.result()))


def test_train():
    train_ds, _ = get_datasets()
    mymodel, cce_loss, optimizer, train_loss, train_acc = build_train_elements(train_ds)
    train(train_ds, mymodel, cce_loss, optimizer, train_loss, train_acc)


def test_test():
    _, test_ds = get_datasets()
    mymodel, cce_loss, test_loss, test_acc = build_test_elements(test_ds)
    test(test_ds, mymodel, cce_loss, test_loss, test_acc)


def test_func():
    _, test_ds = get_datasets()

    test_model = MyModel(name='mymodel')

    input_shape = test_ds.element_spec[0].shape
    test_model.build(input_shape=input_shape)
    # test_model.call(inputs=tf.keras.Input(input_shape[1:]))

    var_tensors = test_model.trainable_variables
    print(len(var_tensors))
    for va in var_tensors:
        print(type(va))
        type_str=str(type(va))
        print(type_str,'tensorflow' in type_str)

        for att in dir(va):
            print(att)

        # 获取值的方法
        val=va.read_value()
        print(type(val))

        # 赋值的方法
        va.assign(val)

        break


if __name__ == '__main__':
    test_train()
    # test_test()
    # test_func()
