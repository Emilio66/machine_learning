## My answer to Chapter 14 Recurrent Neural Network

### 1. Can you think of a few applications for a sequence-to-sequence RNN? What about a sequence-to-vector RNN? And a vector-to-sequence RNN?
> seq2seq: predict time-series data suach as stock price, weather prediction, speech-to-text, music genneration; seq2vec: sentiment analysis (classification problem) music genre; vec2seq: image captioning

### 2. Why do people use encoder–decoder RNNs rather than plain sequence-to-sequence RNNs for automatic translation?
> some basic structure in different language are quite similar in my mind, using a vector to represent the meta-semantic meaning of a sentence in a language, then reorder or change the expression in another language in a delayed manner. The vector acts as the common bond among different languages, seq2seq is not able to do this because it outputs the sequence immediately without arranging the correct order of words in another language, so the output might not make sense at all.

### 3. How could you combine a convolutional neural network with an RNN to classify videos?
> use CNN to transform image/video frame to a sequence output ; then feed the output to a seq2vector RNN, plus a softmax layer at the end of RNN, then it will output the class of videos with corresponding probability.

### 4. What are the advantages of building an RNN using dynamic_rnn() rather than static_rnn()?
> you could input variable length instances without the need for stacking & unstacking

*Referenced Answer:*
* Able to swap GPU's memory to CPU, avoiding out of memory error in BP
* Easy to use, only take a single Tensor as input/output, no need to stack/unstack/transpose
* Generate smaller graph, easy to visualize

### 5. How can you deal with variable-length input sequences? What about variable-length output sequences?
> variable-length input means variable time steps, doesn't indicate the number of features is variable. We can do this by specify the  parameter sequence_length in dynamic_rnn function or padding the vacant place with zeros in order to align with other instance; As for variable-length of output, set a EOS symbol, ignore all the output after the symbol

### 6. What is a common way to distribute training and execution of a deep RNN across multiple GPUs?
> write a DeviceCellWrapper
devices = ["/gpu:0", "/gpu:1", "/gpu:2"]
cells = [DeviceCellWrapper(dev,tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)) for dev in devices]

### 7. Embedded Reber grammars were used by Hochreiter and Schmidhuber in their paper about LSTMs. They are artificial grammars that produce strings such as “BPBTSXXVPSEPE.” Check out Jenny Orr’s nice introduction to this topic. Choose a particular embedded Reber grammar (such as the one represented on Jenny Orr’s page), then train an RNN to identify whether a string respects that grammar or not. You will first need to write a function capable of generating a training batch containing about 50% strings that respect the grammar, and 50% that don’t.

### 8. Tackle the “How much did it rain? II” Kaggle competition. This is a time series prediction task: you are given snapshots of polarimetric radar values and asked to predict the hourly rain gauge total. Luis Andre Dutra e Silva’s interview gives some interesting insights into the techniques he used to reach second

### 9. Go through TensorFlow’s Word2Vec tutorial to create word embeddings, and then go through the Seq2Seq tutorial to train an English-to-French translation system.
