#from memory_profiler import profile
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
#import tensorflow_ranking as tfr
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

########to limit the numbers of CORE
# num_threads = 5
# os.environ["OMP_NUM_THREADS"] = "5"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "5"
# os.environ["TF_NUM_INTEROP_THREADS"] = "5"
#
# tf.config.threading.set_inter_op_parallelism_threads(
#     num_threads
# )
# tf.config.threading.set_intra_op_parallelism_threads(
#     num_threads
# )
# tf.config.set_soft_device_placement(True)


import argparse
import ns_lib as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle
from typing import List, Tuple

from dataset import KGCDataHandler, build_domains
from model import CollectiveModel
from keras.callbacks import CSVLogger
from ns_lib.logic.commons import Atom, Domain, FOL, Rule, RuleLoader
from ns_lib.grounding.grounder_factory import BuildGrounder
from ns_lib.utils import MMapModelCheckpoint, KgeLossFactory

explain_enabled: bool = False

def get_arg(args, name: str, default=None, assert_defined=False):
    value = getattr(args, name) if hasattr(args, name) else default
    if assert_defined:
        assert value is not None, 'Arg %s is not defined: %s' % (name, str(args))
    return value

#@profile
def main(base_path, output_filename, kge_output_filename, log_filename, args):

    # print("Num GPUs Available: ", len(gpus))
    csv_logger = CSVLogger(log_filename, append=True, separator=';')
    print('ARGS', args)

    seed = get_arg(args, 'seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Enable this to find to debug NANs.
    #tf.debugging.experimental.enable_dump_debug_info(
    #    "/tmp/tfdbg2_logdir",
    #    tensor_debug_mode="FULL_HEALTH",
    #    circular_buffer_size=-1)

    # Params
    ragged = get_arg(args, 'ragged', None, True)
    valid_frequency = get_arg(args, 'valid_frequency', None, True)
    train_file = get_arg(args, 'train_file', 'train.txt', False)
    valid_file = get_arg(args, 'valid_file', 'valid.txt', False)
    test_file = get_arg(args, 'test_file', 'test.txt', False)
    domain_file = get_arg(args, 'domain_file', None, False)

    # Data Loading
    data_handler = KGCDataHandler(
        dataset_name=args.dataset_name,
        base_path=base_path,
        format=get_arg(args, 'format', None, True),
        domain_file=domain_file,
        train_file=train_file,
        valid_file=valid_file,
        test_file=test_file,
        fact_file=None)

    corrupt_mode = get_arg(args, 'corrupt_mode', 'TAIL')
    dataset_train = data_handler.get_dataset(
        split="train",
        number_negatives=args.num_negatives)
    dataset_valid = data_handler.get_dataset(
        split="valid",
        number_negatives=args.valid_negatives, corrupt_mode=corrupt_mode)
    dataset_test = data_handler.get_dataset(split="test", corrupt_mode=corrupt_mode)
    if explain_enabled:
        dataset_test_positive_only = data_handler.get_dataset(
            split="test", number_negatives=0, corrupt_mode=corrupt_mode)

    fol = data_handler.fol
    domain2adaptive_constants: Dict[str, List[str]] = None
    dot_product = get_arg(args, 'engine_dot_product', False)
    num_adaptive_constants = get_arg(args, 'engine_num_adaptive_constants', 0)
    if num_adaptive_constants > 0:
        domain2adaptive_constants = {
            d.name : ['__adaptive_%s_%d' % (d.name, i)
                      for i in range(num_adaptive_constants)]
            for d in fol.domains
        }

    ### defining rules and grounding engine
    rules = []
    engine = None

    enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
    if enable_rules:
        rules = RuleLoader.load(
            join(base_path, args.dataset_name, args.rule_file), args.num_rules)
        facts = list(data_handler.train_known_facts_set)
        engine = BuildGrounder(args, rules, facts, fol,
                               domain2adaptive_constants)
    print('Rules', rules, flush=True)

    print('Build serializer', flush=True)
    serializer = ns.serializer.LogicSerializerFast(
        predicates=fol.predicates, domains=fol.domains,
        constant2domain_name=fol.constant2domain_name,
        domain2adaptive_constants=domain2adaptive_constants)

    print('Build model', flush=True)
    # The model can be built here or passed from the outside in case of
    # usage of a pre-trained one.
    model = CollectiveModel(
        fol, rules,
        kge=args.kge,
        kge_regularization=args.kge_regularization,
        model_name=get_arg(args, 'model_name', 'dcr'),
        constant_embedding_size=args.constant_embedding_size,
        predicate_embedding_size=args.predicate_embedding_size,
        kge_atom_embedding_size=args.kge_atom_embedding_size,
        kge_dropout_rate=args.kge_dropout_rate,
        reasoner_single_model=get_arg(args, 'reasoner_single_model', False),
        reasoner_atom_embedding_size=args.reasoner_atom_embedding_size,
        reasoner_formula_hidden_embedding_size=args.reasoner_formula_hidden_embedding_size,
        reasoner_regularization=args.reasoner_regularization_factor,
        reasoner_dropout_rate=args.reasoner_dropout_rate,
        reasoner_depth=args.reasoner_depth,
        aggregation_type=args.aggregation_type,
        signed=args.signed,
        resnet=get_arg(args, 'resnet', False),
        embedding_resnet=get_arg(args, 'embedding_resnet', False),
        temperature=args.temperature,
        filter_num_heads=args.filter_num_heads,
        filter_activity_regularization=args.filter_activity_regularization,
        num_adaptive_constants=num_adaptive_constants,
        dot_product=dot_product,
        cdcr_use_positional_embeddings=get_arg(
            args, 'cdcr_use_positional_embeddings', True),
        cdcr_num_formulas=get_arg(args, 'cdcr_num_formulas', 3),
        r2n_prediction_type=get_arg(args, 'r2n_prediction_type', 'full'),
    )

    # Preparing data as generators for model fit
    print('Build Train generators', flush=True)
    data_gen_train = ns.dataset.DataGenerator(
        dataset_train, fol, serializer, engine,
        batch_size=args.batch_size, ragged=ragged)

    print('Build Valid generators', flush=True)
    data_gen_valid = ns.dataset.DataGenerator(
        dataset_valid, fol, serializer, engine,
        batch_size=args.val_batch_size, ragged=ragged)

    print('Build Test generators', flush=True)
    data_gen_test = ns.dataset.DataGenerator(
        dataset_test, fol, serializer, engine,
        batch_size=args.test_batch_size, ragged=ragged)

    #print('BATCH_TRAIN', next(iter(data_gen_train))[0], flush=True)
    #print('BATCH_TEST', next(iter(data_gen_test))[0], flush=True)

    assert get_arg(args, 'checkpoint_load', None) is None or (
        get_arg(args, 'kge_checkpoint_load', None) is None)
    if get_arg(args, 'checkpoint_load', None) is not None:
        checkpoint_load = get_arg(args, 'checkpoint_load', None)
        print('Loading weights from ', checkpoint_load, flush=True)
        _ = model(next(iter(data_gen_train))[0])  # force building the model.
        model.load_weights(checkpoint_load)
        model.summary()

    if get_arg(args, 'kge_checkpoint_load', None) is not None:
        kge_checkpoint_load = get_arg(args, 'kge_checkpoint_load', None)
        print('Loading weights from ', kge_checkpoint_load[0],
              'Trainable', kge_checkpoint_load[1], flush=True)
        _ = model(next(iter(data_gen_train))[0])  # force building the model.
        model.kge_model.load_weights(kge_checkpoint_load[0])
        model.kge_model.trainable = kge_checkpoint_load[1]
        model.kge_model.summary()
        model.summary()

    # Model Compilation must follow loading weights, as sub-model freezing is
    # not possible after compile.
    print('Compile model', flush=True)
    #Loss
    loss_name = get_arg(args, 'loss', 'binary_crossentropy')
    loss = KgeLossFactory(loss_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    metrics = [ns.utils.MRRMetric(),
               ns.utils.HitsMetric(1),
               ns.utils.HitsMetric(3),
               ns.utils.HitsMetric(10),
               ns.utils.AUCPRMetric()]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  run_eagerly=True)

    callbacks = []
    callbacks.append(csv_logger)
    best_model_callback = MMapModelCheckpoint(
        model, 'val_output_2_mrr',
        frequency=valid_frequency,
        # if path is not None, checkpoint to file.
        filepath=get_arg(args, 'ckpt_filepath', None))
    callbacks.append(best_model_callback)
    kge_filepath = get_arg(args, 'ckpt_filepath', None)
    if kge_filepath is not None:
        kge_filepath =  '%s_kge_model' % kge_filepath
    kge_best_model_callback = MMapModelCheckpoint(
        model.kge_model, 'val_output_1_mrr',
        frequency=valid_frequency,
        # if path is not None, checkpoint to file.
        filepath=kge_filepath)
    callbacks.append(kge_best_model_callback)

    if args.epochs > 0:
        print('Start Train', flush=True)
        model.fit(data_gen_train,
                  epochs=args.epochs,
                  callbacks=callbacks,
                  validation_data=data_gen_valid,
                  validation_freq=valid_frequency)
        best_model_callback.restore_weights()

    if output_filename is not None:
        print('Saving model weights to', output_filename)
        model.save_weights(output_filename, overwrite=True)

    print("\nEvaluation", flush=True)
    train_results = {}  #model.evaluate(data_gen_train, return_dict=True)
    valid_results = {}  #model.evaluate(data_gen_valid, return_dict=True)
    test_results  = model.evaluate(data_gen_test, return_dict=True)
    print('Results',
          'Train', train_results,
          'Val', valid_results,
          'Test', test_results,
          flush=True)

    if explain_enabled and enable_rules and (
            args.model_name == 'dcr' or args.model_name == 'cdcr'):
        model.explain_mode(True)
        print('\nExplain Train', flush=True)
        print(model.predict(data_gen_train)[-1])

        print('\nExplain Test', flush=True)
        data_gen_test_explain = ns.dataset.DataGenerator(
            dataset_test, fol, serializer, engine, batch_size=-1, ragged=ragged)
        print(model.predict(data_gen_test_explain)[-1])

        data_gen_test_positive_only = ns.dataset.DataGenerator(
            dataset_test_positive_only, fol, serializer, engine,
            batch_size=args.test_batch_size, ragged=ragged)
        for r in model.reasoning[-1].rule_embedders.values():
            r._verbose=True
        print(model.predict(data_gen_test_positive_only)[-1])

    return train_results, valid_results, test_results, model
