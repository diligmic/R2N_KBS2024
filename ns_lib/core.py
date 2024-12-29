from keras import Model
import tensorflow as tf

class NSModel(Model):



    def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose='auto',
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):

        for i in range(epochs):
            for batch, (X,y) in enumerate(x):
                ms = self.train_step((X,y))
                print("Epoch %d" % i, "Step %d / %d" % (batch+1, len(x)), [(n,m) for n,m in ms.items()])
            if i % validation_freq == 0:
                self.evaluate(validation_data)


    def train_step(self, data):
        X,y = data
        with tf.GradientTape() as tape:
            y_pred = self(X)
            loss_v = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        train_vars = self.trainable_variables
        gradients = tape.gradient(loss_v, train_vars)
        self.optimizer.apply_gradients(zip(gradients, train_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):

        pass



    def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False,
               return_dict=False,
               **kwargs):
        pass
