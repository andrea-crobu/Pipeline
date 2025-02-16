import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1, max_seq_length=20, tokenizer=None):
        """Set the hyper-parameters and build the layers."""
        super(DecoderLSTM, self).__init__()
        # we want a specific output size, which is the size of our embedding, so we feed our extracted features from the last fc layer (dimensions 1 x 2048)
        # into a first Linear layer to resize and match the shape of the embedding
        self.resize = nn.Linear(2048, embed_size)
        
        # batch normalisation helps to speed up training
        self.batch_normalisation = nn.BatchNorm1d(embed_size, momentum=0.01)

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Use an LSTM as recurrent network
        self.rnn = nn.LSTM(input_size=embed_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)

        # add a fully connected layer to predict the next token
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.max_seq_length = max_seq_length

        # tokenizer
        self.tokenizer = tokenizer
        

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        
        im_features = self.resize(features)
        im_features = self.batch_normalisation(im_features)
        im_features = im_features.unsqueeze(1)
        
        embeddings = self.embed(captions)
        
        x = torch.cat((im_features, embeddings), 1)
               
        packed = pack_padded_sequence(x, lengths, batch_first=True) #  to handle variable-length sequences efficiently in RNNs.
        hiddens, _ = self.rnn(packed)
        outputs = self.linear(hiddens[0])        
        return outputs
    

    def sample(self, features):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        
        # Initialize the hidden state as None so that the LSTM initializes it to zeros
        previous_hidden_state = None

        # prepare the first input - take the feature vector of the image(s) and pass it through the linear layer + batch normalisation
        inputs = self.resize(features)              # (batch_size, embed_size)
        inputs = self.batch_normalisation(inputs)
        inputs = inputs.unsqueeze(1)                # (batch_size, 1, embed_size)
                
        while len(sampled_ids) < self.max_seq_length:
            hiddens, current_hidden_state = self.rnn(inputs, previous_hidden_state)               # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))   # outputs:  (batch_size, vocab_size)
            _, predicted_idx = outputs.max(1)           # predicted_idx: (batch_size)
            
            sampled_ids.append(predicted_idx)
            
            # prepare input for next step - feed the predicted word as the next input
            inputs = self.embed(predicted_idx)          # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                # inputs: (batch_size, 1, embed_size)
            
            previous_hidden_state = current_hidden_state
            
            # check if <eos> 
            if predicted_idx[-1].item() == self.tokenizer(['<eos>'])[0]:
                break
                       
        sampled_ids = torch.stack(sampled_ids, 1)       # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids