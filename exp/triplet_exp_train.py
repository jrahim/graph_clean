import socket, shutil, os, pickle, json, glob
from torch.utils.data import ConcatDataset
from core_dl.train_params import TrainParameters
from exp.triplet_exp_box import DenseCorrTrainBox
from data.oned_sfm.local_feat_dataset_fast import OneDSFMDataset
from data.capture.capture_dataset_fast import CaptureDataset
import torchvision.transforms as transforms
# from exp.make_yfcc_dataset import clean_cache, make_dataset
# from exp.make_dataset_old import clean_cache, make_dataset
import utils.make_yfcc_dataset_new as yfcc
server_name = socket.gethostname()

# [1]
""" Train Parameters ---------------------------------------------------------------------------------------------------
"""
# toggle `DEBUG` to disable logger (won't dump to disk)
DEBUG = True

# set train parameters
train_params = TrainParameters()
train_params.MAX_EPOCHS = 40
train_params.START_LR = 1.0e-4
train_params.DEV_IDS = [0, 1]
train_params.LOADER_BATCH_SIZE = 1
train_params.LOADER_NUM_THREADS = 0
train_params.VALID_STEPS = 700
train_params.MAX_VALID_BATCHES_NUM = 20
train_params.CHECKPOINT_STEPS = 6000
train_params.VERBOSE_MODE = True

# specific unique description for current training experiments
train_params.NAME_TAG = 'dense_correspondence_triplet_mixed'
train_params.DESCRIPTION = 'Training to find dense correspondence with triplet loss'

# [2]
""" Configure dataset and log directory, depend on server ---------------------------------------------------------------
"""
if server_name == 'cs-gruvi-24-cmpt-sfu-ca':

    # load checkpoint if needed
    checkpoint_dict = {'ckpt': None,
                       'vlad': '/mnt/Tango/pg/pg_akt_old/cache/netvlad_vgg16.tar'}

    # set log dir
    log_dir = './log/'


# [3]
""" define dataset ------------------------------------------------------------------------------------------------------
"""
# clean_cache()
train_set, valid_set = yfcc.make_dataset('/mnt/Tango/pg/pg_akt_rot_avg/train_config/yfcc_2_80nodes.json')

# [4]
""" Train --------------------------------------------------------------------------------------------------------------
"""

train_box = DenseCorrTrainBox(train_params=train_params,
                                   log_dir=log_dir,
                                   ckpt_path_dict=checkpoint_dict)
if not DEBUG:
    shutil.copy(os.path.realpath(__file__), train_box.model_def_dir)          # save the train interface to model def dir
train_box.train_loop(train_set, valid_data=valid_set)