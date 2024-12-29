import os.path
import sys

from collections.abc import Iterable
import tensorflow as tf
import argparse
import numpy as np
import sys
from typing import Dict
from tensorflow_ranking.python.utils import sort_by_scores, ragged_to_dense

#######################################
# Utils.

# Recursive flattening to a 1D Iterable. TODO(check if this works).
# Example usages:
# Flatten([[1,2,3], 2, ['a','b']], tuple) -> (1,2,3,2,'a','b')
# Flatten([[1,2,3], 2, ['a','b']], list) -> [1,2,3,2,'a','b']
# Flatten(([1,2,3], 2, ('a','b')), list) -> [1,2,3,2,'a','b']
# Flatten(([1,2,3], 2, ('a','b')), np.array) -> np.array(1,2,3,2,'a','b')
def Flatten(lst: Iterable, flattening_function=tuple) -> Iterable:
    return flattening_function(Flatten(i, flattening_function)
                               if (not isinstance(elem, str) and
                                   isinstance(i, Iterable))
                               else i for i in lst)

# Can we just use the one above? Or this one? TODO cleanup.
def to_flat(nestedList):
    ''' Converts a nested list to a flat list '''
    flatList = []
    # Iterate over all the elements in given list
    for elem in nestedList:
        # Check if type of element is list
        if not isinstance(elem, str) and isinstance(elem, list):
            # Extend the flat list by adding contents of this element (list)
            flatList.extend(to_flat(elem))
        else:
            # Append the element to the list
            flatList.append(elem)
    return flatList

# Skips lines starting with comment_start, otherwise all lines are kept.
def read_file_as_lines(file, allow_empty=False, comment_start='#'):
    try:
        with open(file, 'r') as f:
            if comment_start:
                return [line.rstrip() for line in f.readlines()
                        if line[:len(comment_start)] != comment_start]
            else:
                return [line.rstrip() for line in f.readlines()]
    except IOError as e:
        if allow_empty:
            return []
        raise IOError("Couldn't open file (%s)" % file)

# Used by operation, TODO move to logic/commons.py or remove as there are already methods to do this.
def parse_atom(atom):
    spls = atom.split("(")
    atom_str = spls[0].strip()
    constant_str = spls[1].split(")")[0].split(",")

    return [atom_str] + [c.strip() for c in constant_str]


################################
# Callbacks.
class ActivateFlagAt(tf.keras.callbacks.Callback):
    """Activate a boolean tf.Variable at the beginning of a specific epoch"""

    def __init__(self, flag: tf.Variable, at_epoch : int):
        super().__init__()
        self.flag  = flag
        self.at_epoch = at_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch  == self.at_epoch - 1:
            print("Activating flag %s at epoch %d" % (self.flag.name, epoch + 1))
            self.flag.assign(True)

class PrintEachEpochCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, fun):
        super().__init__()
        self.model  = model
        self.fun = fun

    def on_epoch_end(self, epoch, logs=None):
        print(self.fun(self.model))


# Checkpointer that allows both in-memory and filename checkpointing.
# This extends the functionalities of tf.keras.callbacks.ModelCheckpoint,
# which can save only to file and it is much slower for small
# non-persistent tests.
class MMapModelCheckpoint(tf.keras.callbacks.Callback):
  """Save models to Memory or files as a Keras callback."""

  def __init__(self, model: tf.keras.Model,
               monitor: str='val_loss',
               maximize: bool=True,
               verbose: bool=True,
               filepath: str=None,
               frequency: int = 1):

    self._model = model
    self.best_val = -sys.float_info.max if maximize else sys.float_info.max
    self.monitor = monitor
    self._weights_saved: bool = False
    self._best_weights = None
    self.best_epoch = None
    self.maximize = maximize
    self.verbose = verbose
    self.frequency = frequency
    # Basepath where checkpoints are saved.
    self._filepath: str = filepath
    self._last_checkpoint_filename: str = None

  def on_epoch_end(self, epoch, logs):
    if (epoch+1) % self.frequency != 0:
        return

    assert self.monitor  in logs, (
        'Unknown metric %s at epoch %d. Use the MMapModelCheckpoint.frequency if you are not validating at each step' % (self.monitor, epoch))
    val = logs[self.monitor]
    if (self.maximize and val >= self.best_val) or (
        not self.maximize and val <= self.best_val):
      self.best_val = val
      self.best_epoch = epoch
      if self.verbose:
        print('Checkpointing %s: new best val (%.3f)' % (self.monitor, val), flush=True)
      if self._filepath is not None:
          filename = '%s__epoch%d.ckpt' % (self._filepath, epoch)
          self._model.save_weights(filename)
          if self.verbose:
              print('Weights stored to %s' % filename, flush=True)
          self._last_checkpoint_filename = filename
      else:
          self._best_weights = self._model.get_weights()
      self._weights_saved = True


  def restore_weights(self):
    if not self._weights_saved:
        print('Can not restore the weights as they have not been saved yet')
        return

    assert self._model is not None
    if self.verbose:
        print('Restoring weights from epoch', self.best_epoch)

    if self._last_checkpoint_filename is not None:
        print('Restoring from file %s' % self._last_checkpoint_filename)
        self._model.load_weights(self._last_checkpoint_filename)
    else:
        # In memory restoring.
        assert self._best_weights is not None
        self._model.set_weights(self._best_weights)

#############################################
# Runtime utils.
class NSParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        self.add_argument("-k", "--kge", type=str,
                            default='complex', help="The KGE embedder.")
        self.add_argument("-a", "--kge_atom_embedding_size", type=int,
                            default=10, help="Atom embedding size.")
        self.add_argument("-rd", "--reasoning_depth", type=int,
                            default=1, help="Reasoning depth.")
        self.add_argument("-e", "--epochs", type=int,
                            default=200, help="Epoch number for training.")
        self.add_argument("-s", "--seed", default=0, type=int,
                            help="Seed for random generators.")
        self.add_argument("-lr", "--learning_rate", default=0.01, type=float,
                            help="Learning rate.")

class Logger():


    def __init__(self, file):
        self.file = file
        if os.path.exists(self.file):
            self.df = pd.read_csv(self.file)
        else:
            self.df = None



    def log(self, args:dict):
        if self.df is None:
            self.df = pd.DataFrame(columns=[k for k in args.keys()])
        self.df = self.df.append(args, ignore_index = True)
        self.df.to_csv(self.file, index=False)


    def exists(self, args:dict):
        if self.df is None:
            return False
        ddf = self.df[list(args.keys())]
        s = pd.Series(args)
        r = (ddf == s)
        res = bool(r.all(axis=1).any())
        return res


class FileLogger():

    def __init__(self, folder):
        self.folder = folder


    def _read_last_line(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            return lines[-1]

    def log(self, args:dict, filename):
        """Append`the results as last line of a filename"""
        header_filename = os.path.join(self.folder, "header.txt")
        if not os.path.exists(header_filename):
            header = [str(a) for a in list(args.keys())]
            with open(header_filename, "w") as f:
                f.write(",".join(header))
        with open(filename, "a") as f:
            f.write("\n")
            f.write(",".join(['%s:%s' % (str(k), str(v)) for k,v in list(args.items())]))

    def exists(self, run_signature: str):
        for filename in os.listdir(self.folder):
            if not filename.startswith('log'):
                continue
            path = os.path.join(self.folder, filename)
            last_line = self._read_last_line(path)
            if run_signature in last_line:
                return True
        return False

    def write_to_csv(self, to_write):
        lines = []
        for filename in os.listdir(self.folder):
            if filename.startswith("log"):
                last_line = self._read_last_line(os.path.join(self.folder,filename))
                lines.append(last_line)
            if filename.startswith("header"):
                header = self._read_last_line(os.path.join(self.folder,filename))
        with open(os.path.join(self.folder,to_write), "w") as f:
            f.write(header + "\n")
            for line in lines:
                f.write(line+"\n")


######################################################
# Loss utils. TODO these are training specific and should be moved.
class BinaryCrossEntropyRagged(tf.keras.losses.Loss):
    def __init__(self, balance_negatives=False, from_logits=False):
        super().__init__()
        self.balance_negatives = balance_negatives
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if (isinstance(y_true, tf.RaggedTensor) or
            isinstance(y_pred, tf.RaggedTensor)):
            y_true = y_true.to_tensor()
            y_pred = y_pred.to_tensor()
        loss = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)

        if self.balance_negatives:
            num_positives = tf.reduce_sum(tf.where(y_true == 1, 1.0, 0.0),
                                          axis=-1, keepdims=True)
            num_negatives = tf.reduce_sum(tf.where(y_true == 0, 1.0, 0.0),
                                          axis=-1, keepdims=True)
            loss_positive = tf.math.divide_no_nan(
                tf.where(y_true == 1, loss, 0.0),
                tf.expand_dims(num_positives, axis=-1))
            loss_negative = tf.math.divide_no_nan(
                tf.where(y_true == 0, loss, 0.0),
                tf.expand_dims(num_negatives, axis=-1))
            loss = loss_positive + loss_negative
        return loss

class PairwiseCrossEntropyRagged(tf.keras.losses.Loss):
    def __init__(self, balance_negatives=False, from_logits=False):
        super().__init__()
        self.balance_negatives = balance_negatives
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if (isinstance(y_true, tf.RaggedTensor) or
            isinstance(y_pred, tf.RaggedTensor)):
            y_true = y_true.to_tensor()
            y_pred = y_pred.to_tensor()
        pos_loss = -tf.reduce_sum(tf.where(y_true == 1, tf.math.log(1e-7 + tf.nn.sigmoid(y_pred)), 0.0),
                                  axis=-1, keepdims=True)
        neg_loss = -tf.reduce_sum(tf.where(y_true == 0, tf.math.log(1e-7 + tf.nn.sigmoid(-y_pred)), 0.0),
                                  axis=-1, keepdims=True)
        if self.balance_negatives:
            num_positives = tf.reduce_sum(tf.where(y_true == 1, 1.0, 0.0),
                                          axis=-1, keepdims=True)
            num_negatives = tf.reduce_sum(tf.where(y_true == 0, 1.0, 0.0),
                                          axis=-1, keepdims=True)

        loss = tf.squeeze(pos_loss + neg_loss)
        return loss


class CategoricalCrossEntropyRagged(tf.keras.losses.Loss):
    def __init__(self, from_logits=False):
        super().__init__()
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if (isinstance(y_true, tf.RaggedTensor) or
            isinstance(y_pred, tf.RaggedTensor)):
            y_true = y_true.to_tensor()
            y_pred = y_pred.to_tensor()
        return tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)

######################################
# KGE utils and metrics, this is application dependent, TODO move to kge.py?
def KgeLossFactory(name: str) -> tf.keras.losses.Loss:
    if name == 'hinge':
        return HingeLossRagged(gamma=1.0)
    elif name == 'l2':
        return L2LossRagged()
    elif name == 'categorical_crossentropy':
        return CategoricalCrossEntropyRagged()
    elif name == 'binary_crossentropy':
        return BinaryCrossEntropyRagged()
    elif name == 'balanced_binary_crossentropy':
        return BinaryCrossEntropyRagged(balance_negatives=True)
    elif name == 'balanced_pairwise_crossentropy':
        return PairwiseCrossEntropyRagged(balance_negatives=True)
    else:
        assert False, 'Unknown loss %s'% name

class MRRMetric(tf.keras.metrics.Metric):
  """Implements mean reciprocal rank (MRR). It uses the same implementation
     of tensorflow_ranking MRRMetric but with an online check for ragged
     tensors."""
  def __init__(self, name='mrr', dtype=None, **kwargs):
      super().__init__(name, dtype, **kwargs)
      self.mrr = self.add_weight("total", initializer="zeros")
      self._count = self.add_weight("count", initializer="zeros")
      self.reset_state()

  def reset_state(self):
      self.mrr.assign(0.)
      self._count.assign(0.)

  def result(self):
      return tf.math.divide_no_nan(self.mrr, self._count)

  def update_state(self, y_true, y_pred, sample_weight=None):
    mrrs = self._compute(y_true, y_pred)
    self.mrr.assign_add(tf.reduce_sum(mrrs))
    self._count.assign_add(tf.reduce_sum(tf.ones_like(mrrs)))

  def _compute(self, labels, predictions):
    if any(isinstance(tensor, tf.RaggedTensor)
           for tensor in [labels, predictions]):
      labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)

    topn = tf.shape(predictions)[1]
    sorted_labels, = sort_by_scores(predictions, [labels], topn=topn, mask=None)
    sorted_list_size = tf.shape(input=sorted_labels)[1]
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    reciprocal_rank = 1.0 / tf.cast(
        tf.range(1, sorted_list_size + 1), dtype=tf.float32)
    # MRR has a shape of [batch_size, 1].
    mrr = tf.reduce_max(
        input_tensor=relevance * reciprocal_rank, axis=1, keepdims=True)
    return mrr

# Wrapper around tf.keras.metrics.AUC adding the convertion to ragged tensors.
class AUCPRMetric(tf.keras.metrics.AUC):
  """Implements mean reciprocal rank (MRR). It uses the same implementation
     of tensorflow_ranking MRRMetric but with an online check for ragged
     tensors."""
  def __init__(self, name='auc-pr', dtype=None, **kwargs):
      super().__init__(curve='PR', name=name, dtype=dtype)

  def _compute(self, labels, predictions):
    if any(isinstance(tensor, tf.RaggedTensor)
           for tensor in [labels, predictions]):
      labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)
    return super()._compute(labels, predictions)

class HitsMetric(tf.keras.metrics.Metric):
  """Implements the HITS@N metric. It uses the same implementation
     of tensorflow_ranking MRRMetric but with an online check for ragged
     tensors."""
  def __init__(self, n, name='hits', dtype=None, **kwargs):
      super().__init__('%s@%d' % (name, n), dtype, **kwargs)
      self._n = n
      self.hits = self.add_weight("total", initializer="zeros")
      self._count = self.add_weight("count", initializer="zeros")
      self.reset_state()

  def reset_state(self):
      self.hits.assign(0.)
      self._count.assign(0.)

  def result(self):
      return tf.math.divide_no_nan(self.hits, self._count)

  def update_state(self, y_true, y_pred, sample_weight=None):
    hits = self._compute(y_true, y_pred)
    self.hits.assign_add(tf.reduce_sum(hits))
    self._count.assign_add(tf.reduce_sum(tf.ones_like(hits)))

  def _compute(self, labels, predictions):
    if any(isinstance(tensor, tf.RaggedTensor)
           for tensor in [labels, predictions]):
      labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)

    topn = tf.shape(predictions)[1]
    sorted_labels, = sort_by_scores(predictions, [labels], topn=topn, mask=None)
    sorted_list_size = tf.shape(input=sorted_labels)[1]
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    top_relevance = relevance[:, :self._n]
    hits = tf.reduce_sum(top_relevance, axis=1, keepdims=True)
    return hits

  def get_config(self):
      base_config = super().get_config()
      config = { 'n': self._n, }
      return {**base_config, **config}


def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
