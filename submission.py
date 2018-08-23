## Submission.py for COMP6714-Project2
###################################################################################################################
import collections
import math
import os
import random
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import gensim
from gensim import utils


def adjective_embeddings(data_file = 'processed_data.txt', embeddings_file_name = 'adjective_embeddings.txt', num_steps=1000, embedding_dim=200):
#    global data_index
    
    def read_data(filename):
        data=[]
        with open(filename, "r", encoding = 'utf-8') as f:
            adj, data_processed = f.read().split('\t')
            data = data_processed.split(' ')
            adj_list = adj.split(' ')
#            data = f.read().split()
        return data, adj_list
    
    def build_dataset(words, n_words):
        """Process raw inputs into a dataset. 
           words: a list of words, i.e., the input data
           n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
        """
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            index = dictionary.get(word, 0)
            if index == 0:  # i.e., one of the 'UNK' words
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary
    
    def generate_batch(batch_size, num_samples, skip_window, data_index):
#        global data_index   
        
        assert batch_size % num_samples == 0
        assert num_samples <= 2 * skip_window
        
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # span is the width of the sliding window
        buffer = collections.deque(maxlen=span)
        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index + span]) # initial buffer content = first sliding window
        
#        print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))
    
        data_index += span
        for i in range(batch_size // num_samples):
            context_words = [w for w in range(span) if w != skip_window]
            random.shuffle(context_words)
            words_to_use = collections.deque(context_words) # now we obtain a random list of context words
            for j in range(num_samples): # generate the training pairs
                batch[i * num_samples + j] = buffer[skip_window]
                context_word = words_to_use.pop()
                labels[i * num_samples + j, 0] = buffer[context_word] # buffer[context_word] is a random context word
            
            # slide the window to the next position    
            if data_index == len(data):
                buffer = data[:span]
                data_index = span
            else: 
                buffer.append(data[data_index]) # note that due to the size limit, the left most word is automatically removed from the buffer.
                data_index += 1
            
#            print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))
            
        # end-of-for
        data_index = (data_index + len(data) - span) % len(data) # move data_index back by `span`
        return batch, labels, data_index
    
    processed_data, adj_list = read_data(data_file)
    vocabulary_size = 20000
    data, count, dictionary, reverse_dictionary = build_dataset(processed_data, vocabulary_size)
    data_index = 0
    
    
    # Specification of Training data:
    batch_size = 128      # Size of mini-batch for skip-gram model.
#    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 2       # How many words to consider left and right of the target word.
    num_samples = 4         # How many times to reuse an input to generate a label.
    num_sampled = 10      # Sample size for negative examples.
    learning_rate = 0.001
#    logs_path = './log/'
    
    # Specification of test Sample:
#    sample_size = 20       # Random sample of words to evaluate similarity.
#    sample_window = 100    # Only pick samples in the head of the distribution.
#    sample_examples = np.random.choice(sample_window, sample_size, replace=False) # Randomly pick a sample of size 16
#    num_steps = 1001
    ## Constructing the graph...
    graph = tf.Graph()
    
    with graph.as_default():
        
#        with tf.device('/cpu:0'):
        # Placeholders to read input data.
        with tf.name_scope('Inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            
        # Look up embeddings for inputs.
        with tf.name_scope('Embeddings'):  
#            test_dataset = tf.placeholder(tf.int32, shape=[40])          
#            sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dim], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            
            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dim],
                                                      stddev=1.0 / math.sqrt(embedding_dim)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases, 
                                             labels=train_labels, inputs=embed, 
                                             num_sampled=num_sampled, num_classes=vocabulary_size))
        
        # Construct the Gradient Descent optimizer using a learning rate of 0.01.
        with tf.name_scope('Gradient_Descent'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
        # Normalize the embeddings to avoid overfitting.
        with tf.name_scope('Normalization'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm 
        
        # Add variable initializer.
        init = tf.global_variables_initializer()
        
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)
#        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
        print('Initializing the model')
        
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels, data_index = generate_batch(batch_size, num_samples, skip_window, data_index)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            
            # We perform one update step by evaluating the optimizer op using session.run()
#            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            print('loss:',loss_val,'~~~~~data_index=', data_index)
#            summary_writer.add_summary(summary, step )
            average_loss += loss_val
    
            if step % 1000 == 0:
                if step > 0:
                    average_loss /= 1000
                
                    # The average loss is an estimate of the loss over the last 5000 batches.
                    print('Average loss at step ', step, ': ', average_loss, '  data_index=', data_index)
                    average_loss = 0

        final_embeddings = normalized_embeddings.eval()
        adj_order = []
        unadj_order = []
        for i in range(vocabulary_size):
            if reverse_dictionary[i] in adj_list:
                adj_order.append(i)
            else:
                unadj_order.append(i)
#        print(len(adj_order))
        adj_order.extend(unadj_order)
        
        with utils.smart_open(embeddings_file_name, 'wb',encoding = 'utf-8') as fout:
            fout.write(utils.to_utf8("%s %s\n" % (vocabulary_size, embedding_dim)))
            for i in range(vocabulary_size):
                row = final_embeddings[adj_order[i]]
                word = reverse_dictionary[adj_order[i]]
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))
        session.close()

def process_data(input_data = 'BBC_Data.zip'):
    nlp = spacy.load('en')
    def read_data(filename):
        """Extract the first file enclosed in a zip file as a list of words."""
        data=''
        with zipfile.ZipFile(filename) as f:
            
            for file in f.namelist():
                subdata = tf.compat.as_str(f.read(file))
                data+=subdata
        return nlp(data)

    data = read_data(input_data)
    
#    data_nlp = nlp(data)
    l = ["PUNCT","SYM","SPACE","ADD","DET","PART","CCONJ","NUM","X"]
    number = ['0','1','2','3','4','5','6','7','8','9']
    data_fix = ''
    adj_list = []
    for i in data:
        if (i.pos_ not in l) and (i.text[0] not in number):
            if i.pos_ == 'ADJ':
                if i.lemma_ != '-PRON-':
                    adj_list.append(i.lemma_)
                else:
                    adj_list.append(i.text)
            if i.lemma_ != '-PRON-':
                data_fix += (i.lemma_+' ')
            else:
                data_fix += (i.text+' ')
    adj_seq=''
    for i in set(adj_list):
        adj_seq += (i+' ')
    data_fix = adj_seq + '\t' + data_fix
    file_name = 'processed_data.txt'
    with open(file_name, "w", encoding = 'utf-8') as f:
        f.write(data_fix)
    return file_name
    


def Compute_topk(model_file, input_adjective, top_k):
    
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    sim_list = model.similar_by_word(input_adjective,top_k,3064)
    top_k_list = []
    for i in sim_list:
        top_k_list.append(i[0])
    return top_k_list
    
    
    
    
    