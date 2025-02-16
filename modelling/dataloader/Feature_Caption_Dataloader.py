import torch
from torch.utils.data import Dataset, DataLoader

class Feature_Caption_Dataloader(DataLoader):
    def __init__(self, features, captions, batch_size=64, shuffle=True):
        dataset = Feature_Caption_Dataset(features, captions)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=caption_collate_fn)
        

class Feature_Caption_Dataset(Dataset):
    """ Flickr8k custom dataset with features and vocab, compatible with torch.utils.data.DataLoader. """
    
    def __init__(self, features, captions):
        """ Set the path for images, captions and vocabulary wrapper.
        
        Args:
            image_ids (str list): list of image ids (WHY?)
            captions (str list): list of str captions
            vocab: vocabulary wrapper
            features: torch Tensor of extracted features
        """
        self.captions = captions
        self.features = features
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, index):
        """ Returns one data pair (feature [1, 2048] and target caption as Tensor of word ids). """
        feature = self.features[index]
        tokenized_caption = self.captions[index]
        
        # convert caption (string) to word ids using the custom vocabulary
        tokenized_caption = torch.LongTensor(tokenized_caption) # must be LongTensor because we pass it to nn.Embedding (it expects torch.int64, i.e. LongTensor)
            
        return (feature, tokenized_caption)
    
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (feature, caption).
    Args:
        data: list of tuple (feature, caption). 
            - feature: torch tensor of shape (1, 2048).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        feature: torch tensor of shape (batch_size, 1, 2048).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """   
    # Sort a data list by caption length from longest to shortest. By default, 'pack_padded_sequence' requires the sequences to be sorted in descending order of length.
    data.sort(key=lambda x: len(x[1]), reverse=True) # x = (feature, caption), x[1] = caption
    features, captions = zip(*data)

    # Merge features (from tuple of 2D tensor to 3D tensor -> (batch_size, 1, 2048)).
    features = torch.stack(features, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    
    # pad with zeros
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap       
         
    return features, targets, lengths