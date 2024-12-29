import copy
import datetime
import gc
import os
from itertools import product
from train import main
import shutil as sh
from ns_lib.utils import FileLogger, NSParser
import tensorflow as tf


if __name__ == '__main__':

    base_path :str = "data"
    parallel :bool = False

    epochs: int = 100
    assert epochs > 0
    skip_already_run: bool = False

    output_folder :str = 'results_wn18rr_001'
    log_folder :str = output_folder
    src_folder :str = os.path.join(output_folder, "src")
    ckpt_folder :str = os.path.join(output_folder, "ckpt")
    dataset_name :str = os.path.join('wn18rr')
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    if not os.path.exists(log_folder): os.mkdir(log_folder)
    if not os.path.exists(src_folder): os.mkdir(src_folder)
    if ckpt_folder and not os.path.exists(ckpt_folder): os.mkdir(ckpt_folder)
    for filename in os.listdir("."):
        filepath = os.path.join(".", filename)
        if os.path.isfile(filepath):
            sh.copy2(filepath, src_folder)

    SEED = [0]
    E = [200]
    DROPOUT = [0.0]

    NEG_PER_SIDE = [16]
    R = [0.0]
    RR = [0.0]
    LR = [0.0005]
    NUM_RULES = [10]
    DEPTH = [1]
    VALID_SIZE = [None]
    KGE = ['rotate']  # ["distmult", "transe", "complex", "rotate"]
    MODELS = ['dcr']  # , 'rnm', 'gsbr', 'sbr']
    BACKWARD_CHAINING_COMPLETE_PROOFS = [True]
    BACKWARD_CHAINING_DEPTH = [1]
    GROUNDER = ['KnownBodyGrounder']
    all_args = []

    for model_name, seed, dropout, r, neg, e, lr, nr, dp, v, kge, rr, bwcp, bwd, gr in product(
            MODELS,
            SEED, DROPOUT, R,
            NEG_PER_SIDE, E, LR,
            NUM_RULES, DEPTH,
            VALID_SIZE, KGE, RR,
            BACKWARD_CHAINING_COMPLETE_PROOFS, BACKWARD_CHAINING_DEPTH, GROUNDER):

        if (gr != 'ApproximateBackwardChainingGrounder' and
            (BACKWARD_CHAINING_COMPLETE_PROOFS.index(bwcp) != 0 or
             BACKWARD_CHAINING_DEPTH.index(bwd) != 0)):
            continue

        if (nr == 0 and
            (DEPTH.index(dp) != 0 or
             GROUNDER.index(gr) != 0 or
             BACKWARD_CHAINING_COMPLETE_PROOFS.index(bwcp) != 0 or
             BACKWARD_CHAINING_DEPTH.index(bwd) != 0)):
            continue

        run_vars = (model_name, seed, dropout, r, neg, e, lr, nr, dp, v, kge, rr, bwcp, bwd, gr)

        # Base parameters
        parser = NSParser()
        args = parser.parse_args()
        args.run_signature = '_'.join(f'{v}' for v in run_vars)
        # Checkpoint to file.
        args.ckpt_filepath = (os.path.join(ckpt_folder, args.run_signature)
                              if ckpt_folder else None)
        args.kge_checkpoint_load = None  #((os.path.join(
        #    ckpt_folder,
        #    'r2n_0_0.0_0.0_16_%d_0.0005_0_1_None_%s_0.0_True_1_RelationalEntityGrounder_kge_model__epoch99.ckpt'% (e, kge))
        #                             if ckpt_folder else None), False)
        # Basic training params."
        args.epochs = epochs
        args.learning_rate = lr
        args.ragged = True
        args.seed = seed
        args.debug = False
        args.model = None
        args.model_name = model_name
        args.batch_size = 512
        args.val_batch_size = -1
        args.test_batch_size = 10
        args.valid_size = v
        args.valid_negatives = 200
        args.valid_frequency = 5
        args.test_negatives = -1
        args.loss = "binary_crossentropy"
        args.num_negatives = neg
        args.corrupt_mode = 'HEAD_AND_TAIL'

        # Dataset params.
        args.dataset_name = dataset_name
        args.format = "functional"
        args.train_file = "train.txt"
        args.valid_file = "test.txt"
        args.test_file = "test.txt"
        args.rule_file = "rules_flat_dcr.txt"  # rules

        # Grounding params.
        args.num_rules = nr
        args.grounding_type = gr
        args.engine_num_negatives = 0
        args.engine_num_adaptive_constants = 0
        args.relation_entity_grounder_max_elements = 20
        args.backward_chaining_prune_incomplete = bwcp
        args.backward_chaining_depth = bwd
        args.backward_chaining_max_groundings_per_rule = 10000

        # Stuctural params.
        args.adaptation_layer = "identity"  # "dense", "sigmoid","identity"
        args.output_layer = "dense" # "wmc" or "kge" or "positive_dense" or "max
        args.stop_gradient_on_kge_embeddings = False
        args.rule_weight = "number" # "embedding"
        args.semiring = "product"

        # KGE params.
        args.kge = kge
        args.kge_atom_embedding_size = e
        args.kge_dropout_rate = dropout
        args.kge_regularization = r

        # Constant params.
        args.constant_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == 'complex' or args.kge == 'rotate'
            else args.kge_atom_embedding_size)
        args.predicate_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == 'complex'
            else args.kge_atom_embedding_size)

        # Reasoner params.
        args.resnet = True
        args.embedding_resnet = False
        args.reasoner_depth = dp if nr > 0 else 0
        args.enabled_reasoner_depth = args.reasoner_depth
        args.reasoner_regularization_factor = rr
        args.reasoner_formula_hidden_embedding_size = args.kge_atom_embedding_size
        args.reasoner_dropout_rate = dropout
        args.reasoner_atom_embedding_size = args.kge_atom_embedding_size

        # Model-specific params
        args.signed = True
        args.temperature = 0.0
        args.aggregation_type = "max"
        args.filter_num_heads = 3
        args.filter_activity_regularization = 0.0
        args.cdcr_use_positional_embeddings = False
        args.cdcr_num_formulas = 3

        all_args.append(args)


    def main_wrapper(args):
        logger = FileLogger(log_folder)
        if skip_already_run and logger.exists(args.run_signature):
            print("Skipping", args)
            return

        time = str(datetime.datetime.now()).replace(":","-")
        log_filename_tmp = os.path.join(log_folder, '_tmp_log%s.csv' % time)
        log_filename = os.path.join(
            log_folder, 'log%s_%s.csv' % (args.run_signature, time))

        train_results, valid_results, test_results, _ = main(
            base_path, None, None, log_filename_tmp, args)

        # Split the args used for training from the logged data.
        if hasattr(args, 'model'):
            delattr(args, 'model')
        logged_data = copy.deepcopy(args)
        # Add some extra info to log.
        logged_data.valid_results = valid_results
        logged_data.test_results = test_results
        logged_data.log_filename = log_filename
        # Log the data to its final location.
        logger.log(logged_data.__dict__, log_filename_tmp)
        if os.path.exists(log_filename):
            os.remove(log_filename)
        os.rename(log_filename_tmp, log_filename)
        print('Run completed', args.run_signature, flush=True)

    for args in all_args:
        main_wrapper(args)
        print('Running garbage collector', flush=True)
        tf.keras.backend.clear_session()
        gc.collect()
