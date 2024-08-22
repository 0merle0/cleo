import torch
from torch.utils.data import Dataset
import fragment_util
import pdb_util

class FragmentDataset(Dataset):
    
    def __init__(self, 
                 df, 
                 input_col='seq', 
                 label_col='norm_activity', 
                 name_col='name', 
                 fragment_dictionary=None, 
                 fragment_representation=False):

        super(FragmentDataset, self).__init__()

        '''
            initialize and build training set from csvs
        '''
        self.seqs = df[input_col].tolist()
        self.activities = torch.tensor(df[label_col].tolist())
        
        if fragment_representation:
            # if training over fragment space
            self.featurized_seqs = fragment_util.featurize_fragments(df[name_col].tolist(), fragment_dictionary)
            num_classes = max([len(fragment_dictionary[x]) for x in fragment_dictionary])
        else:
            # get num seqs
            self.featurized_seqs = []
            for seq in self.seqs:
                self.featurized_seqs.append([pdb_util.aa12num[x] for x in seq])
            num_classes = 20

        # if one hot then featurize the sequences to be one hot
        self.featurized_seqs = get_one_hot(self.featurized_seqs, num_classes=num_classes)
        self.featurized_seqs = self.featurized_seqs.reshape(self.featurized_seqs.shape[0],-1)
        
    def __len__(self):
        return len(self.featurized_seqs)
    
    def __getitem__(self, index):
        return self.featurized_seqs[index], self.activities[index]
    

def get_one_hot(seqs,  num_classes=20):
    '''
        go from num seq to one hot seq
    '''
    one_hot_seqs = []
    for x in seqs:
        one_hot_seqs.append(torch.nn.functional.one_hot(torch.tensor(x),num_classes=num_classes)[None])
    return torch.cat(one_hot_seqs,dim=0)