import tensorflow as tf
import argparse

# Pass the filename as an argument
parser = argparse.ArgumentParser()
parser.add_argument("--frozen_model_filename", default="/path-to-pb-file/Binary_Protobuf.pb", type=str,
                    help="Pb model file to import")
args = parser.parse_args()

# We load the protobuf file from the disk and parse it to retrieve the
# unserialized graph_def
with tf.gfile.GFile(args.frozen_model_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    # saver=tf.train.Saver()
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
        sess = tf.Session(graph=graph)
        saver = tf.train.Saver()
        save_path = saver.save(sess, "path-to-ckpt/model.ckpt")
        print("Model saved to chkp format")