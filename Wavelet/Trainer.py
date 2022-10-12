import math
import math

import numpy as np
import tensorflow as tf
from fastprogress import master_bar, progress_bar
from sklearn.metrics import classification_report


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    # checkpoint settings
    ckpt_path = '/content/drive/MyDrive/EMOTION/Wavelet/checkpoint_VIT_Without/1/'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, model_config, train_dataset, train_dataset_len, test_dataset, test_dataset_len, config):
        self.train_dataset = train_dataset
        self.train_dataset_len = train_dataset_len
        self.test_dataset = test_dataset
        self.test_dataset_len = None
        self.test_dist_dataset = None
        if self.test_dataset:
            self.test_dataset = test_dataset
            self.test_dataset_len = test_dataset_len
        self.config = config
        self.tokens = 0
        self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        if len(tf.config.list_physical_devices('GPU')) > 1:
            self.strategy = tf.distribute.MirroredStrategy()

        with self.strategy.scope():
            self.model = model(**model_config)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                     reduction=tf.keras.losses.Reduction.NONE)
            # self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            # if self.test_dataset:
            #    self.test_dist_dataset = self.strategy.experimental_distribute_dataset(self.test_dataset)
        self.restore = False
        # self.right = 0

    def save_checkpoints(self, epoch):
        if self.config.ckpt_path is not None:
            self.model.save_weights(self.config.ckpt_path + str(epoch))

    def restore_checkpoints(self):
        if self.restore:
            self.model.load_weights('/content/drive/MyDrive/EMOTION/Wavelet/checkpoint_VIT_Without/1/1')

    def accuracy_fn(self, logits, y_target):
        y_pred = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
        # y_pred = y_pred.eval(session=tf.compat.v1.Session())
        # print(y_pred)
        summ = tf.math.reduce_sum(y_pred)
        # print('summ : ',summ)
        pred = 0
        if summ > 16:
            pred = 1
        return pred

    def train(self):

        train_loss_metric = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        test_loss_metric = tf.keras.metrics.Mean('testing_loss', dtype=tf.float32)

        train_accuracy = tf.keras.metrics.Accuracy('training_accuracy', dtype=tf.float32)
        test_accuracy = tf.keras.metrics.Accuracy('testing_accuracy', dtype=tf.float32)

        @tf.function
        def train_step(x, y):

            def step_fn(X, Y):
                # X, Y = inputs

                with tf.GradientTape() as tape:
                    # training=True is only needed if there are layers with different
                    # behavior during training versus inference (e.g. Dropout).
                    logits = self.model(X, training=True)
                    num_labels = tf.shape(logits)[-1]
                    label_mask = tf.math.logical_not(Y < 0)
                    label_mask = tf.reshape(label_mask, (-1,))
                    logits = tf.reshape(logits, (-1, num_labels))
                    logits_masked = tf.boolean_mask(logits, label_mask)
                    label_ids = tf.reshape(Y, (-1,))
                    label_ids_masked = tf.boolean_mask(label_ids, label_mask)
                    cross_entropy = self.cce(label_ids_masked, logits_masked)
                    loss = tf.reduce_sum(cross_entropy) * (1.0 / self.config.batch_size)
                    y_pred = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
                    train_accuracy.update_state(tf.squeeze(Y), y_pred)

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
                return cross_entropy

            per_example_losses = self.strategy.run(step_fn, args=(x, y,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        @tf.function
        def test_step(x, y):

            def step_fn(X, Y):
                # X, Y = inputs
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                logits = self.model(X, training=False)
                pred = self.accuracy_fn(logits, Y)
                num_labels = tf.shape(logits)[-1]
                label_mask = tf.math.logical_not(Y < 0)
                label_mask = tf.reshape(label_mask, (-1,))
                logits = tf.reshape(logits, (-1, num_labels))
                logits_masked = tf.boolean_mask(logits, label_mask)
                label_ids = tf.reshape(Y, (-1,))
                label_ids_masked = tf.boolean_mask(label_ids, label_mask)
                cross_entropy = self.cce(label_ids_masked, logits_masked)
                loss = tf.reduce_sum(cross_entropy) * (1.0 / self.config.batch_size)
                y_pred = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
                test_accuracy.update_state(tf.squeeze(Y), y_pred)

                return cross_entropy, pred

            per_example_losses, pred = self.strategy.run(step_fn, args=(x, y,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss, pred

        train_pb_max_len = math.ceil(float(self.train_dataset_len) / float(32))
        test_pb_max_len = math.ceil(float(self.test_dataset_len) / float(32)) if self.test_dataset else None

        epoch_bar = master_bar(range(self.config.max_epochs))
        with self.strategy.scope():
            countt = 0
            for epoch in epoch_bar:
                for data, label in progress_bar(self.train_dataset, total=train_pb_max_len, parent=epoch_bar):
                    x = (data.numpy()).reshape(32, 1, 48, 48)
                    y = label.numpy().reshape(-1)
                    loss = train_step(x, y)
                    self.tokens += tf.reduce_sum(tf.cast(y >= 0, tf.int32)).numpy()
                    train_loss_metric(loss)
                    epoch_bar.child.comment = f'training loss : {train_loss_metric.result()}'
                    if countt == 0:
                        countt += 1
                        self.restore_checkpoints()
                print(
                    f"epoch {epoch + 1}: train loss {train_loss_metric.result():.5f}. train accuracy {train_accuracy.result():.5f}")
                train_loss_metric.reset_states()
                train_accuracy.reset_states()

                if self.test_dataset:
                    total_right = 0
                    countt = 0
                    predss = []
                    correctss = []
                    for data, label in progress_bar(self.test_dataset, total=test_pb_max_len, parent=epoch_bar):
                        x = (data.numpy()).reshape(32, 1, 48, 48)
                        y = label.numpy().reshape(-1)
                        loss, pred = test_step(x, y)
                        if pred == y[0]:
                            total_right += 1
                        correctss.append(y[0])
                        predss.append(pred)
                        test_loss_metric(loss)
                        epoch_bar.child.comment = f'testing loss : {test_loss_metric.result()}'
                        countt += 1

                    print(
                        f"epoch {epoch + 1}: test loss {test_loss_metric.result():.5f}. test accuracy {test_accuracy.result():.5f}. test right accuracy {(total_right / countt):.5f}")
                    print(classification_report(np.array(correctss), np.array(predss)))
                    test_loss_metric.reset_states()
                    test_accuracy.reset_states()

                print(
                    '------------------------------------------------------------------------------------------------------------------------------------')

                self.save_checkpoints(epoch)
