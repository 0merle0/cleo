import torch
from torch.utils.data import Dataset, DataLoader
import fragment_util
import pdb_util
import pytorch_lightning as pl
import pandas as pd
from icecream import ic
from sklearn.model_selection import train_test_split
import glob
import os
import pdb

class FragmentDataset(Dataset):
    
    def __init__(
                    self, 
                    df,
                    input_col,
                    label_col,
                    name_col,
                    fragment_csv,
                    use_fragment_representation,
                    use_pretrained_embeddings,
                    embedding_path,
                ):
        """Simplified dataset."""

        super(FragmentDataset, self).__init__()
        features = self.featurize_inputs(
                                            df,
                                            input_col,
                                            label_col,
                                            name_col,
                                            fragment_csv,
                                            use_fragment_representation,
                                            use_pretrained_embeddings,
                                            embedding_path                                        
                                        )   
        self.input_feats, self.labels = features

    def get_one_hot(self, seqs, num_classes=20):
        """Go from num seq to one hot seq."""
        one_hot_seqs = []
        for x in seqs:
            one_hot_seqs.append(torch.nn.functional.one_hot(x,num_classes=num_classes))
        return torch.stack(one_hot_seqs)
        
    def featurize_inputs(
                            self, 
                            df, 
                            input_col, 
                            label_col, 
                            name_col, 
                            fragment_csv,
                            use_fragment_representation,
                            use_pretrained_embeddings,
                            embedding_path
                        ):
        """Featurize inputs from dataframe."""
        labels = torch.tensor(df[label_col].tolist()) 

        if use_fragment_representation:
            assert fragment_csv is not None, 'Must provide fragment csv if training over fragment space'
            # if training over fragment space
            fragment_dictionary = fragment_util.get_fragment_dictionary(fragment_csv)
            input_feats = fragment_util.featurize_fragments(
                                                        df[name_col].tolist(), 
                                                        fragment_dictionary
                                                        )
            num_classes = max([len(fragment_dictionary[x]) for x in fragment_dictionary])
            input_feats = self.get_one_hot(input_feats, num_classes=num_classes)
            input_feats = input_feats.reshape(input_feats.shape[0],-1)
        elif use_pretrained_embeddings:

            print('Using pretrained embeddings')

            if embedding_path is None:
                raise Exception('Must provide embedding path if using pretrained embeddings')

            # Infer file pattern from first AA sequence 
            if not hasattr(self, '_file_pattern'):
                # Get first AA sequence
                first_aa = df['AAs'].iloc[0]
                
                for entry in os.scandir(embedding_path):
                    if entry.name.endswith('.pt') and first_aa in entry.name:
                        example_file = entry.name
                        prefix = example_file.split(first_aa)[0]
                        suffix = example_file.split(first_aa)[1]
                        self._file_pattern = (prefix, suffix)
                        break
                else:
                    raise FileNotFoundError(f"Could not find any .pt files in the path")

            all_embeddings = []

            for _, row in df.iterrows():
                prefix, suffix = self._file_pattern
                path = os.path.join(embedding_path, f"{prefix}{row['AAs']}{suffix}")
                
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Could not find embeddings file for sequence {row['AAs']}")
                
                embedding_dict = torch.load(path, map_location='cpu') 
                embedding = embedding_dict[row['AAs']]
                all_embeddings.append(embedding)

            input_feats = torch.stack(all_embeddings)
            input_feats = input_feats.reshape(input_feats.shape[0],-1)
        else:
            # get num seqs
            input_feats = []
            for seq in df[input_col].tolist():
                input_feats.append(torch.tensor([pdb_util.aa12num[x] for x in seq]))
            num_classes = 20

            # if one hot then featurize the sequences to be one hot
            input_feats = self.get_one_hot(input_feats, num_classes=num_classes)
            input_feats = input_feats.reshape(input_feats.shape[0],-1)

        return input_feats, labels

    def __len__(self):
        return len(self.input_feats)
    
    def __getitem__(self, index):
        return self.input_feats[index], self.labels[index]
    

class FragmentDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(FragmentDataModule, self).__init__()
        self.cfg = cfg
    
    def get_train_val_split(self, df):
        """Split data into train and val."""
        
        df_val = None
        if self.cfg.validation_mode == 'random':
            ic('Random validation')

            df_train, df_val =  train_test_split(
                                        df, 
                                        test_size=self.cfg.val_size, 
                                        random_state=self.cfg.seed
                                    )

        elif self.cfg.validation_mode == 'top-k':
            ic('Top k validation')
            raise Exception('Not implemented yet')

        elif self.cfg.validation_mode == 'label':
            ic('Label validation')
            df_train = df[~df[self.cfg.val_label]]
            df_val = df[df[self.cfg.val_label]]      

        elif self.cfg.validation_mode is None:
            ic('Validation is turned off')
            df_train = df

        else:
            raise Exception(f'Validation mode not recognized: {self.cfg.validation_mode}')

        # build datasets
        train_dataset = FragmentDataset(
                                            df_train,
                                            input_col=self.cfg.input_col,
                                            label_col=self.cfg.label_col,
                                            name_col=self.cfg.name_col,
                                            fragment_csv=self.cfg.fragment_csv,
                                            use_fragment_representation=self.cfg.use_fragment_representation,
                                            use_pretrained_embeddings=self.cfg.use_pretrained_embeddings.use,
                                            embedding_path=self.cfg.use_pretrained_embeddings.embeddings_path
                                        )
        print('Train dataset length:', len(train_dataset))
        print('First input shape:', train_dataset[0][0].shape)
        val_dataset = None
        if df_val is not None:
            val_dataset = FragmentDataset(
                                            df_val,
                                            input_col=self.cfg.input_col,
                                            label_col=self.cfg.label_col,
                                            name_col=self.cfg.name_col,
                                            fragment_csv=self.cfg.fragment_csv,
                                            use_fragment_representation=self.cfg.use_fragment_representation,
                                            use_pretrained_embeddings=self.cfg.use_pretrained_embeddings.use,
                                            embedding_path=self.cfg.use_pretrained_embeddings.embeddings_path
                                        )   
        
        return train_dataset, val_dataset


    def compute_distances(self, seq1, seq2_list):
        """Compute Hamming distances between a sequence and a list of sequences."""
        dists = []
        for seq2 in seq2_list:
            dist = sum(a != b for a, b in zip(seq1, seq2))
            dists.append(dist)
        return dists

    def prepare_data(self):
        """Load local data on cpu."""
        df = pd.read_csv(self.cfg.dataset)
        self.train_dataset, self.val_dataset = self.get_train_val_split(df) 
        

        # test data       
        if self.cfg.test_dataset is not None:
            train_seqs = df[~df[self.cfg.val_label]][self.cfg.input_col].tolist()
            test_df = pd.read_csv(self.cfg.test_dataset)
            min_dists = []
            mean_dists = []
            for seq in test_df[self.cfg.input_col]:
                dists = self.compute_distances(seq, train_seqs)
                min_dists.append(min(dists))
                mean_dists.append(sum(dists) / len(dists))

            
            self.test_dataset = FragmentDataset(
                                                test_df,
                                                input_col=self.cfg.input_col,
                                                label_col=self.cfg.label_col,
                                                name_col=self.cfg.name_col,
                                                fragment_csv=self.cfg.fragment_csv,
                                                use_fragment_representation=self.cfg.use_fragment_representation,
                                                use_pretrained_embeddings=self.cfg.use_pretrained_embeddings.use,
                                                embedding_path=self.cfg.use_pretrained_embeddings.embeddings_path
                                            )
            
            self.test_dataset.min_dists = torch.tensor(min_dists)
            self.test_dataset.mean_dists = torch.tensor(mean_dists)

    def setup(self, stage):
        """Called on each DDP process."""
        # currently not being used at the moment
        pass

    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(self.train_dataset, 
                               batch_size=self.cfg.train_batch_size,
                               shuffle=True,
                               num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        """Validation dataloader."""
        val_loader = None
        if self.cfg.validation_mode is not None:
            val_loader =  DataLoader(self.val_dataset,
                               batch_size=self.cfg.val_batch_size,
                               shuffle=False,
                               num_workers=self.cfg.num_workers)
        return val_loader 

    def test_dataloader(self):
        """Test dataloader."""
        test_loader = None
        if self.cfg.test_dataset is not None:
            test_loader = DataLoader(
                                    self.test_dataset,
                                    batch_size=self.cfg.test_batch_size,
                                    shuffle=False,
                                    num_workers=self.cfg.num_workers,
        )
        return test_loader
    
    def predict_dataloader(self):
        """Predict dataloader not used."""
        return None