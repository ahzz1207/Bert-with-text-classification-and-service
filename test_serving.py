from tensorflow.saved_model import tag_constants 
import numpy as np
import tensorflow as tf
import test_seving_model
def getSess():
	flags = tf.flags

	FLAGS = flags.FLAGS

	flags.DEFINE_string(
		"export_dir", 'export/1547709290',
		"The dir where the exported model has been written.")

	processor, tokenizer, label_list, predict_file = test_seving_model.main('_')
	print(predict_file)
	sess = tf.InteractiveSession()
	tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.export_dir)

	labels = dict()
	for i,l in enumerate(sorted(label_list)):
		labels[l] = i
	label_real = list(labels.keys())
	return sess,FLAGS,processor, tokenizer, label_real, predict_file
# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# for tensor_name in tensor_name_list:
# 	print(tensor_name, '\n')
# vars = tf.get_collection('GraphKeys', scope='loss')
# for var in vars:
# 	print(var)
def predict(sess, FLAGS,processor, tokenizer, label_real, predict_file,sentence):
	#while True:
	graph = tf.get_default_graph()
	sentence = sentence
	tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
	tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
	tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
	tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
	tensor_outputs = graph.get_tensor_by_name('loss/Softmax:0')
	#tensor_first_token = graph.get_tensor_by_name('bert/embeddings/pooler/first:0')
	#tensor_example_loss = graph.get_tensor_by_name('loss/example:0')

	test_seving_model.getinput(processor, tokenizer, predict_file, sentence)
	record_iterator = tf.python_io.tf_record_iterator(path=predict_file)

	#file_based_input_fn_builder(predict_file, FLAGS.max_seq_length, False, False)
	for sentence in record_iterator:
		example = tf.train.Example()
		example.ParseFromString(sentence)
		input_ids = example.features.feature['input_ids'].int64_list.value
		input_mask = example.features.feature['input_mask'].int64_list.value
		label_ids = example.features.feature['label_ids'].int64_list.value
		segment_ids = example.features.feature['segment_ids'].int64_list.value
		print(np.array(input_ids).reshape(-1, FLAGS.max_seq_length))
		result = sess.run(tensor_outputs, feed_dict={
			tensor_input_ids: np.array(input_ids).reshape(-1, FLAGS.max_seq_length),
			tensor_input_mask: np.array(input_mask).reshape(-1, FLAGS.max_seq_length),
			tensor_label_ids: np.array(label_ids),
			tensor_segment_ids: np.array(segment_ids).reshape(-1, FLAGS.max_seq_length),
		})
	#get label
	label_pob = []
	for i in result[0]:
		label_pob.append(float(i))
	index = label_pob.index(max(label_pob))
	label = label_real[index]
	return label

# def file_based_input_fn_builder(input_file, seq_length, is_training,
#                                 drop_remainder):
#     """Creates an `input_fn` closure to be passed to TPUEstimator."""
#
#     name_to_features = {
#         "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
#         "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
#         "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
#         "label_ids": tf.FixedLenFeature([], tf.int64),
#     }
#
#     def _decode_record(record, name_to_features):
#         """Decodes a record to a TensorFlow example."""
#         example = tf.parse_single_example(record, name_to_features)
#
#         # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
#         # So cast all int64 to int32.
#         for name in list(example.keys()):
#             t = example[name]
#             if t.dtype == tf.int64:
#                 t = tf.to_int32(t)
#             example[name] = t

# return example
