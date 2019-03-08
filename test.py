import test_seving as service
session, FLAGS,processor, tokenizer, label_real, predict_file = service.getSess()

sentence = "make a test"

result = service.predict(session, FLAGS, processor, tokenizer, label_real, predict_file, sentence)