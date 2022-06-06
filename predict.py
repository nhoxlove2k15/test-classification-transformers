# -*- coding: utf-8 -*-
# import clean_text
from phobert import model

from keras.preprocessing.sequence import pad_sequences
import numpy as np


print(111)

path_bert = "./resource/phobert/"
def predict_one_thesis(test_ids) :
  MAX_LEN = 256
  test_id = pad_sequences(test_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
  test_mask = []
  mask = [int(token_id > 0) for token_id in test_id[0]]
  test_mask.append(mask)
  predicts = model.predict([test_id,np.array(test_mask)])
  return predicts


# predict ==================================================================
# print(text)
def classify(predicts,threshhold = 0.1):
  preds = []
  for idx,predict in enumerate(predicts):
    a = []
    preds.append(int(idx+1))
    for i in range (0, len(predict)):
        if predict[i] >= threshhold:
          a.append(classes[i])
    preds.append(a)
  return preds
def top_n(predict, n = 2):
  output = sorted(range(len(predict)), key=lambda k: predict[k], reverse=True)
  # output = predict.sort(reverse=True)
  return output[:n]
# vector hóa tags


classes = [
           'mạng neural','phân loại','xử lý ảnh','di động','web','gan','sdn','học máy','khuyến nghị','bảo mật','ai', 
           'raspberry pi','mạng 5g','blockchain','camera','tự động hóa', 'nhúng', 'fpga',
           'mạng bluetooth', 'cloud', 'mạng wifi', 'rfid' ,'yolo','robot','nhận diện', 
           'iot','vr','e-commerce','game','ar','nlp','mạng xã hội', 'rỗng'
]
# [0 0 1 0 ...0 1 ...]

number_labels = len(classes)
def to_category_vector(label):
    # print(label[-1])
    # print(label)
    vector = np.zeros(len(classes)).astype(np.float64)
    # print(vector)
    if label[-1] == ',':
      # a ='KH,SK,DL,SK,'
      # b = a.split(',') # ["KH","SK","DL","SK",""]
      a = label.split(',')[:-1]
      # a = ['xử lý ảnh', 'web']
      for i in a :
        index = classes.index(i.strip())
        # print(index)
        vector[index] = 1.0
      # return
    else:
    

      index = classes.index(label)
      # print(index)
      vector[index] = 1.0
      # vector[2] = 1.0
    return vector
def PredictLabels(predicts):    
  y_preds = []
  y_preds_string = []  
  for idx,y in enumerate(predicts):
    top_2 = top_n(predicts[idx].tolist(), 2)
    tags_2 = [classes[i] for i in top_2]
    string = ', '.join([str(item) for item in tags_2])
    y_preds_string.append(string+",")
    if idx == 0 : print(string+",")
    # print(string[-1])
    y_preds.append(to_category_vector(string+","))
  return y_preds_string
# top_n(predicts[0].tolist(), 4)
# print(y_preds_string)
