import sys
device_server = True
if device_server:
    config_path = {
        # 'robosense_caffe': '/media/DataCenter/deeplearningoflidar/Yan/robosense_caffe/robosense_caffe_227_sti/python',
        'STIpytorch': '/media/DataCenter/deeplearningoflidar/overfitover/rs_project/perception/STIpytorch/',
        'preprocess_lib': '/media/DataCenter/deeplearningoflidar/overfitover/rs_project/perception/STIClibs-ice-dev/robosense_caffe_227_sti_convexoffset/build',
        # 'robosense_caffe': '/media/DataCenter/deeplearningoflidar/Yan/robosense_caffe/robosense_caffe_227_kitti/python',
    }
else:
    config_path = {
        'robosense_caffe': '/home/yan/Documents/Software/robosense_caffe_localtest/caffe_local/python',
        'STIpytorch': '/home/yan/Documents/DL_project/STIPerception/STIpytorch',
        'preprocess_lib': '/home/yan/Documents/Software/STIClibs-master/preprocessor/build',
    }
for k, v in config_path.items():
    if v not in sys.path:
        sys.path.insert(0, v)

import os
import datetime

__all__ = ['TemplateOptions']

class TemplateOptions(object):
    def __init__(self):
        '''init class: define template class to set experimental parameters
        '''

        self.device_server = device_server  # if True, run code on server
        self.using_ice_caffe = False

        self.debug = False  # whether to debug code

        self.dataset = "RS32"  # options: RS32 | MEMS

        self.use_gpu_id = [7, ]  # list of GPUs to be used

        self.experiment_id = "227_convex_label_carpedoffset_delete"  # tag of each experiment

        self.net_type = "FCNDSD"  # options: FCN, FCNDense, FCNSparse, FCNThin

        self.net_input_channels = 10  # number of input channels of network
        self.net_output_channels = 14  # number of output channels of network
        self.label_channels = 13
        self.feature_generate_channels = 10

        # -------------------------- training options ---------------------------------------
        self.epochs = 150

        self.lr_step_scheduler = [0, 100, 130]
        self.lr_scheduler = [1e-4, 1e-5, 1e-6]
        self.weight_decay = 1e-5

        self.batchsize = [12, 12, 12]  # [train, val, test]

        self.num_workers = 8

        self.trans_aug = [0.8, -5, 5,  # trans_tx_chance,trans_tx_min,trans_tx_max
                          0.8, -5, 5,  # trans_ty_chance,trans_ty_min,trans_ty_max
                          0.8, -0.3, 0.3,  # trans_tz_chance,trans_tz_min,trans_tz_max
                          0.8, -0.02, 0.02,  # trans_rx_chance,trans_rx_min,trans_rx_max
                          0.8, -0.02, 0.02,  # trans_ry_chance,trans_ry_min,trans_ry_max
                          0.8, -1.57, 1.57]  # trans_rz_chance,trans_rz_min,trans_rz_max

        # [vertical_x_flip_augment, horizon_y_flip_augment]
        self.flip_aug = [0.3, 0.3]

        self.expand_aug = 0.

        # --------------------------- dataset -------------------------------------------
        self.training_set_len = -1  # number of training samples
        self.val_set_len = -1  # number of validate samples
        self.testing_set_len = -1  # number of testing samples
        self.test_only = False

        # ---------------------------- checkpoint ----------------------------------------------
        self.weight_file = None#'../../pretrained/apollo_init.pth' #None  # path to weight file for intializing model

        # options for resume checkpoints
        self.resume = None#"../MutiClass_Semantic/RS32_ExfuseMut_139_exfuseMul_1018072714/check_point/checkpoint.pth"

        self.resume_epoch = 0

        self.resume_opt = None

        self.manualSeed = 1  # manually set RNG seed

        # initialize all parameters
        self.prepare()

    def prepare(self):
        # ----------------------------------- initialize path ---------------------------------------------
        assert isinstance(
            self.use_gpu_id, list), "invalid type of use_gpu_id, expect [list], while the real type is: %s" % type(self.use_gpu_id)

        # set number of gpu
        self.n_gpu = len(self.use_gpu_id)

        if self.device_server:
            # set path
            # mems lidar
            if self.dataset == "RS32":
                # rslidar-32
                # self.ptx_train_source = '/media/DataCenter/deeplearningoflidar/stiRS32Dataset/caffe_train_label_data/label/trainsets.txt'
                # self.ptx_eval_source = '/media/DataCenter/deeplearningoflidar/stiRS32Dataset/caffe_train_label_data/label/valsets.txt'
                # self.ptx_test_source = '/media/DataCenter/deeplearningoflidar/stiRS32Dataset/caffe_train_label_data/label/testsets.txt'
                self.ptx_train_source = '../../../train_val_test_sets/trainsets.txt'
                self.ptx_eval_source = '../../../train_val_test_sets/valsets.txt'
                self.ptx_test_source = '../../../train_val_test_sets/testsets.txt'
                self.ptx_root_folder = '/media/DataCenter/datacenter/label/32/'

            elif self.dataset == "Kitti":
                self.ptx_train_source = '/media/DataCenter/deeplearningoflidar/stiKittiDataSet/Kitti/object/sti_trainval_sets/trainsets.txt'
                self.ptx_eval_source = '/media/DataCenter/deeplearningoflidar/stiKittiDataSet/Kitti/object/sti_trainval_sets/valsets.txt'
                self.ptx_test_source = '/media/DataCenter/deeplearningoflidar/stiKittiDataSet/Kitti/object/sti_trainval_sets/testsets.txt'
                self.ptx_root_folder = '/media/DataCenter/deeplearningoflidar/stiKittiDataSet/'

            elif self.dataset == "MEMS":
                # mems
                self.ptx_train_source = '/media/DataCenter/deeplearningoflidar/stiRS32Dataset/caffe_train_label_data/mems/trainsets.txt'
                self.ptx_eval_source = '/media/DataCenter/deeplearningoflidar/stiRS32Dataset/caffe_train_label_data/mems/valsets.txt'
                self.ptx_test_source = None
                self.ptx_root_folder = '/media/DataCenter/deeplearningoflidar/stiRS32Dataset/mems/'

            else:
                assert False, "invalid dataset: %s" % self.dataset

        else:

            # only support one gpu on local PC
            assert self.n_gpu == 1, "local PC only support 1 GPU, while n_gpu is %d" % self.n_gpu
            # set path
            # self.root = '/home/yan/Documents/DL_project/STIPerception/STIpytorch/'
            if self.dataset == "RS32":
                self.ptx_train_source = '../../../train_val_test_sets/trainsets.txt'
                self.ptx_eval_source = '../../../train_val_test_sets/valsets.txt'
                self.ptx_test_source = '../../../train_val_test_sets/testsets.txt'
                self.ptx_root_folder = '/media/DataCenter/datacenter/label/32'
            else:
                assert False, "invalid dataset: %s" % self.dataset

        # set gpu
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        # change this parameter to use specific devices
        os.environ['CUDA_VISIBLE_DEVICES'] = str(
            self.use_gpu_id).strip('[').strip(']')  # "0, "

        # set save path : name=dataset+net_type+(experiment_id[optional])+timestamp
        timestamp = datetime.datetime.now().strftime('%m%d%H%M%S')
        if self.experiment_id is not None:
            self.save_path = "%s_%s_%s_%s" % (
                self.dataset, self.net_type, self.experiment_id, timestamp)
        else:
            self.save_path = "%s_%s_%s" % (
                self.dataset, self.net_type, timestamp)
        # create folder
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)


if __name__ == '__main__':
    # import TemplateOptions as Options
    settings = TemplateOptions()
    print(settings.debug)
