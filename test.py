import test_seving as service
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
session, FLAGS,processor, tokenizer, label_real, predict_file = service.getSess()

sentence = ["Add a record (name and data) to the index.", "Tells JavaScript to open windows automatically.","Just a test."]
for s in sentence:
    result = service.predict(session, FLAGS, processor, tokenizer, label_real, predict_file, s)
    print(result)