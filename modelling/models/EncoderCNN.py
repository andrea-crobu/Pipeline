from torch import nn
import torch
from torchvision import models

class EncoderCNN(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        
        # keep all layers of the pretrained net except the last one
        layers = list(resnet.children())[:-1]
        
        self.resnet = torch.nn.Sequential(*layers)
        self.resnet.eval()
        
    def forward(self, images):
        """Extract feature vectors from input images."""

        # return features; no gradients are needed    
        with torch.no_grad():
            outputs = self.resnet(images)
            
            # at this point, the output is (batch_size, 2048,1,1) -> (batch_size, 2048,)
            outputs = outputs.reshape((outputs.shape[0],outputs.shape[1],))
            
        return outputs

import pandas as pd

class Vocabulary(object):
    """ Simple vocabulary wrapper which maps every unique word to an integer ID. """
    def __init__(self):
        # intially, set both the IDs and words to dictionaries with only the <unk> token
        self.word2idx = {'<unk>': 0}
        self.idx2word = {0: '<unk>'}
        self.idx = 1

    def add_word(self, word):
        # if the word does not already exist in the dictionary, add it
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            # increment the ID for the next word
            self.idx += 1

    def __call__(self, word):
        # if we try to access a word not in the dictionary, return the id for <unk>
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    
class Custom_Tokenizer:
    def __init__(self, list_of_captions):
        self.my_vocabulary = Vocabulary()
        
        # create a list with all the words in the captions
        list_of_all_the_words = []
        for caption in list_of_captions:
            for token in caption:
                list_of_all_the_words.append(token)

        # make it a dataframe and drop all the words that appear 3 times or less
        df_tmp = pd.DataFrame(list_of_all_the_words)
        words_count = df_tmp.value_counts()
        filtered_words = words_count[words_count.values>3].index.get_level_values(0).to_list()
        
        # now add all the words to the Vocabulary
        for word in filtered_words:
            self.my_vocabulary.add_word(word)
            
        
    def __call__(self, captions):
        """
        Tokenize the captions.
        
        Args:
            captions: A list of captions.
            
        Returns:
            A list of tokenized captions.
        """
        tokenized_captions = []
        for caption in captions:
            tokenized_captions.append(self.my_vocabulary(caption))
            
        return tokenized_captions
    
    def decode(self, tokenized_captions):
        """ Decode tokenized captions to words. """
        
        # decode the tokenized captions
        decoded_captions = [self.my_vocabulary.idx2word[c.item()] for c in tokenized_captions]
        
        # remove <sos> and <eos> tokens
        if '<sos>' in decoded_captions:
            decoded_captions.remove('<sos>')
        if '<eos>' in decoded_captions:
            decoded_captions.remove('<eos>')
        
        decoded_captions = ' '.join(decoded_captions)
        
        return decoded_captions        