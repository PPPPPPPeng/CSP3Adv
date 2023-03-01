import tensorflow as tf
import torch

def convert(bin_path, ckptpath):
    with  tf.Session() as sess:
        for var_name, value in torch.load(bin_path, map_location='cpu').items():
            print(var_name)  # 输出权重文件中的变量名
            tf.Variable(initial_value=value, name=var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, ckpt_path)

bin_path = '/content//attn/DAMSMencoders/bird/t200.pth'
ckpt_path = '/content/attn/models/new_model.ckpt'
convert(bin_path, ckpt_path)