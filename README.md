# Bert with text classification and service
This project is a text classification model by bert from google.

1, If you want to try it, you can replace the run_classify.py in bert_master with test_serving_model.py.

2, you should check out your params in this file and make a directory for your exported model.

  flags.DEFINE_string(
		"export_dir", 'export/1547709290', // this is your exported directory
		"The dir where the exported model has been written.")
  
3, If the model saved successfully， you can import the test_serving.py in everywhere you want to call the service.

4, You can ues get_Sess() method to get a tf.Session and others that predict method needs. The session is running in local unitl you close this Thread.

5, Use predict methods with params return from get_Sess and setence to get your results!


    
