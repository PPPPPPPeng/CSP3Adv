# CSP3Adv
code for Collaborative Face Privacy Protection Method Based On Adversarial Examples In Social Platforms  

# datasets:  
lfw:http://vis-www.cs.umass.edu/lfw/lfw.tgz  
calfw:https://pan.baidu.com/s/1ge6wFuV#list/path=%2F  
cplfw:https://pan.baidu.com/s/1i6iHztN#list/path=%2F  
agedb:https://www.dropbox.com/s/mkjsyqytd5lcai9/AgeDB.zip?dl=0  

# Running Environment:

    Python 3.7.7
    pillow, scipy, numpy ...
    tensorflow 1.15.0
    mxnet 1.3.1 (only needed when reading mxrec file)

# Pretrain:
The official InsightFace (ArcFace) project share their training data and testing data in the DataZoo(https://github.com/deepinsight/insightface/wiki/Dataset-Zoo).   
This data is in mxrec format, you can transform it to tfrecord format with ./data/generateTFRecord.py by the following script:
python generateTFRecord.py 
--mode=mxrec
--image_size=112
--read_dir=$DIRECTORY_TO_THE_TRAINING_DATA$
--save_path=$DIRECTORY_TO_SAVE_TFRECORD_FILE$/xxx.tfrecord

# Model Prepare:
We use MobileFaceNet(https://github.com/sirius-ai/MobileFaceNet_TF) as probe model and ArcFace(https://github.com/luckycallor/InsightFace-tensorflow) as server model. You can download pre-trained model weights of MobileFaceNet(https://github.com/sirius-ai/MobileFaceNet_TF/tree/master/arch/pretrained_model/) and ArcFace (password:51hc)(https://pan.baidu.com/s/1p-O-uJGANawMDB41XWd2UQ) to your model directory.
