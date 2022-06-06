print("665")
# from transformers.src.transformers.models.configuration_roberta import RobertaConfig
# from transformers.src.transformers.models.modeling_tf_roberta import TFRobertaModel
import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel
# from transformers import AutoTokenizer, AutoModel
# import transformers
print("66")
print("6")
path_bert = "../resource/phobert/"
config = RobertaConfig.from_pretrained(
    path_bert + "config.json", from_tf=True, dropout=0.2, attention_dropout=0.2
)
print("7")
BERT_SA = TFRobertaModel.from_pretrained(
    path_bert + "model.bin",
    config=config, from_pt=True
)
# BERT_SA = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
# print("8")
# # khởi tạo, set up ma trận sẵn
number_labels = 33
MAX_LEN = 256
# input_ids_in, input_masks_in, outputs dùng Bert_SA (phobert) là tham số để tạo model
input_ids_in = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_token', dtype='int32')
input_masks_in = tf.keras.layers.Input(shape=(MAX_LEN,), name='masked_token', dtype='int32') 
# print(input_ids_in, input_ids_in.shape)
outputs = BERT_SA(input_ids_in,attention_mask = input_masks_in)[0]
X= tf.keras.layers.GlobalAveragePooling1D()(outputs)
X = tf.keras.layers.Dropout(0.5)(X)

X = tf.keras.layers.Dense(number_labels, activation='sigmoid')(X)
model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs = X)
print(9)
model.load_weights("../resource/my_model_14/mymodelWeight.h5")
print(10)
print(model.summary())
print(11)