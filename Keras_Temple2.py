#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
例子二：机器翻译：多语种互译
字典：http://www.manythings.org/anki/
存放位置：F:/DATA/dict/cmn-eng/cmn.txt

序列化处理：RNN：LSTM

'''

from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

# 参数设置
batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000

# 读取数据
data_path = 'F:/DATA/dict/cmn-eng/cmn.txt'

# 向量化数据 Vertorize the data
# _____________________________________
input_texts = []
target_texts = []
input_characters =set()
target_characters = set()

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[:min(num_samples, len(lines)-1)]:
    input_text, target_text = line.split('\t')

# 使用‘tab’作为开始序列的标记，回车作为结束序列的标记
target_text = '\t' + target_text + '\n'
input_texts.append(input_text)
target_texts.append(target_text)

for char in input_text:
    if char not in input_characters:
        input_characters.add(char)

for char in target_text:
    if char not in target_characters:
        target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters) #input的字符串长度
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# 变量初始化
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

# enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1

for t, char in enumerate(target_text):
    decoder_input_data[i, t, target_token_index[char]] = 1

if t>0:
    decoder_target_data[i, t-1, target_token_index[char]] = 1

# LSTM ----Define an input sequence and process it
# _________________________
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c] #discard 'encoder_outputs' and only keep the states

# Set up the decoder, using 'encoder_states' as initial state
decoder_inputs = Input(shape=(None, num_decoder_tokens))

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ =decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn encoder_input_data & decoder_input_data inpto decoder_target_data
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 优化器 选用rmsprop
# 损失函数 categorical_crossentropy
# validation_split是将一个集合随机分成训练集和测试集

# Run training
# ___________________________
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size = batch_size,
        epochs = epochs,
        validation_split=0.2)

# 存储模型
model.save('sts.h5')

decoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(laten_dim, ))
decoder_state_input_c = Input(shape=(laten_dim, ))
decoder_startes_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c =decoder_lstm(
    decoder_inputs, initial_state = decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_state_inputs,
    [decoder_outputs] + decoder_states)
   
# Reverse-lookup token index to decode sequences back to something readable
reverse_input_char_index = dict(
    (i,char) for char, i in input_token_index.items())

reverse_target_char_index = dict (
    (i, char) for char, i in target_token_index.items())
    
# 定义模块
def decoder_sequence(input_seq):
    # Encode the input as state vectors:
    states_value = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
     
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1
     
    # Sampling loop for a batch of sequences to simplify, here we assume a batch of size 1
     
    stop_condition = False
    decoded_sentence = ''
     
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
        [target_seq] + states_value)
     
    # Sample a token
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_char = reverse_target_char_index[sampled_token_index]
    decoded_sentence += sampled_char
     
    # Exit condition: either hit max length or find stop character.
    if (sampled_char == '\n') or len(decoded_sentence) > max_decoder_seq_length:
        stop_condition = True
       
    # Update the target sequence (of length 1):
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1
     
    # Update states
    states_value = [h, c]
     
    return decoded_sentence

for seq_index in range(100):
    # Take one sequence (part of the training seq) for trying out decoding
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decoded_sentence(input_seq)
    print('-')
    print('input sentence:', inpujt_texts[seq_index])
    pring('Decoded_sentence:', decoded_sentence)