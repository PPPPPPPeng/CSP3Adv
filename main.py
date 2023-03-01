import argparse
from PNet_unet import train
from test import test
from test_lpips import lpips_cal
import multiprocessing
parser = argparse.ArgumentParser()
# Input Options

# chose the mode: 'train' or 'test'
parser.add_argument('--mode', type=str, default='test', help='model mode: build')


########################### train settings ########################
parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
#  the path you save the insight model
# parser.add_argument('--insightface_model_path', type=str, default='/media/yl/东方芝士/pretrained/Insightface_ms1m_100_334k/best-m-334000', help='model path')
parser.add_argument('--insightface_model_path', type=str, default='/media/yl/东方芝士/pretrained/Insightface_ms1m_100_1006k/best-m-1006000', help='model path')
parser.add_argument('--val_data', type=str, default='', help='val data, a dict with key as data name, value as data path')
parser.add_argument('--train_mode', type=int, default=0, help='whether set train phase to True when getting embds. zero means False, one means True')

parser.add_argument('--facenet_model_path', type=str, default='/media/yl/东方芝士/pretrained/facenet/20180402-114759/model-20180402-114759')
parser.add_argument('--sphereface_model_path', '-m',type=str, default="/media/yl/东方芝士/pretrained/sphereface-torch/sphere20a_20171020.pth")

parser.add_argument('--mobilefacenet_model_path', type=str, default='/media/yl/东方芝士/pretrained/MobileFaceNet-pretrained_model/MobileFaceNet_TF.ckpt')
parser.add_argument('--train_data', type=str, default='/data/jiaming/datasets/faces/faces_emore/100000.pkl')
parser.add_argument('--train_model_input', type=str, default='')
parser.add_argument('--train_model_output', type=str, default='/data/jiaming/code/InsightFace-tensorflow/model/mm/test_zx/model_apf.ckpt')

parser.add_argument('--train_epoch', type=int, default=60)
parser.add_argument('--train_batchsize', type=int, default=100)

########################### test settings ########################
parser.add_argument('--test_insightface_model_path', type=str, default='', help='model path')
parser.add_argument('--test_mobilefacenet_model_path', type=str, default='')
parser.add_argument('--test_data', type=str, default='')
parser.add_argument('--test_model', type=str, default='')



args = parser.parse_args()
if __name__ == "__main__":

    if args.mode=='train':
        print("train start.")
        train(args)
    if args.mode=='test':
        print("test start.")
        test(args)
    if args.mode=='lpips':
        print("test Lpips start.")
        lpips_cal(args)
