import tarfile, zipfile, os, sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
sys.path.append('../FairFiltering/dataset/')
import utils_glue

transforms = {
    "CelebA": {
        "train": T.Compose(
            [
            T.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": T.Compose(
            [
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "test": T.Compose(
            [
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
    },
    "Waterbirds" :{
        "train": T.Compose(
            [
                T.RandomResizedCrop(
                    size = (224, 224),
                    scale = (0.7, 1.0),
                    ratio = (0.75, 1.333333333),
                    interpolation = 2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "val": T.Compose(
            [
                T.Resize(size = (256, 256)),
                T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "test": T.Compose(
            [
                T.Resize(size = (256, 256)),
                T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    },
    "MetaShift": {
        "train":T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        "val": T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "test": T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])},
    "CivilComments": {"train": None, "val": None, "test": None},
    "MultiNLI": {"train": None, "val": None, "test": None}
}




def extract_file(file_path: str, extract_path = None):
    if extract_path is None:
        extract_path = os.path.dirname(file_path)
        
    if file_path.endswith(".gz"):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
            
    elif file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
    else:
        raise ValueError("Not Defined Filename Extension")


def get_dataset(data, root_dir, split, split_dict, shuffle, ratio):

    transform = transforms[data][split]

    if data == "CelebA":
        meta_data = pd.read_csv(os.path.join(root_dir, data, "list_attr_celeba.csv"))
        split_df = pd.read_csv(os.path.join(root_dir, data, "list_eval_partition.csv"))
        meta_data['split'] = split_df['partition']
        if shuffle:
            meta_data = pd.read_csv(os.path.join(root_dir, data, f"list_attr_celeba_{ratio}.csv"))
            
        meta_data = meta_data[meta_data['split'] == split_dict[split]]
        image_idx = meta_data['image_id'].values;  
        #* blond hair(1) - female(1) & not blond hair(0) - male(0)
        label_idx = meta_data['Blond_Hair'].map({-1: 0, 1: 1}).values
        sens_idx = meta_data['Male'].map({-1: 1, 1: 0}).values #* Female -> 1, Male -> 0
        attr = np.vstack((label_idx, sens_idx)).T
        return image_idx, attr, transform
    
    elif data == "Waterbirds":
        meta_data = pd.read_csv(os.path.join(root_dir, data, "metadata.csv"))
        if shuffle:
            meta_data = pd.read_csv(os.path.join(root_dir, data, f"metadata_{ratio}.csv"))
            
        meta_data = meta_data[(meta_data['split'] == split_dict[split])]
        image_idx = meta_data['img_filename'].values
        label_idx = meta_data['y'].values
        sens_idx = meta_data['place'].values   
        attr = np.vstack((label_idx, sens_idx)).T
        return image_idx, attr, transform

    elif data == 'CivilComments':
        meta_data = pd.read_csv(os.path.join(root_dir, data, "all_data_with_identities.csv"))
        if shuffle:
            meta_data = pd.read_csv(os.path.join(root_dir, data, f"all_data_with_identities_{ratio}.csv"))
            
        meta_data = meta_data[meta_data['split'] == split]
        meta_data = meta_data[meta_data['comment_text'].str.len() > 0]
        bias_column = ['LGBTQ', 'other_religions', 'male', 'female', 'black', 'white', 'christian', 'muslim']
        meta_data['bias_label'] = (meta_data[bias_column] >= 0.5).any(axis=1).astype(int)
        meta_data['target'] = (meta_data['toxicity'] >= 0.5).astype(int)
        
        text_ = meta_data['comment_text'].values
        label_ = meta_data['target'].values
        sense_ = meta_data['bias_label'].values
        attr = np.vstack((label_, sense_)).T     
        return text_ ,attr, transform
    
    elif data == 'MultiNLI':
        meta_data = pd.read_csv(os.path.join(root_dir, data, "metadata_random.csv"))
        bert_filenames = [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]
        features_array = sum([torch.load(os.path.join(root_dir, data, name))
                              for name in bert_filenames], start=[])
        all_input_ids = torch.tensor([f.input_ids for f in features_array]).long()
        all_input_masks = torch.tensor([f.input_mask for f in features_array]).long()
        all_segment_ids = torch.tensor([f.segment_ids for f in features_array]).long()
        data_ = torch.stack((all_input_ids, all_input_masks, all_segment_ids), dim=2)
        
        if shuffle:
            meta_data = pd.read_csv(os.path.join(root_dir, data, f"metadata_random_{ratio}.csv"))
        
        meta_data = meta_data[meta_data["split"] == split_dict[split]]
        targets = np.asarray(meta_data["gold_label"].values)
        spurious = np.asarray(meta_data["sentence2_has_negation"].values)
        
        if shuffle:
            text_ = data_[list(meta_data['Unnamed: 0'])]
        else:
            text_ = data_[list(meta_data.index)]
        attr = np.vstack((targets, spurious)).T
        
        return text_, attr, transform
    else:
        raise ValueError



def train_val_split(meta_data, split_dict: dict, ratio: float):
    # 기존의 train 데이터만 따로 분리
    train_data = meta_data[meta_data['split'] == split_dict['train']]
    valid_data = meta_data[meta_data['split'] == split_dict['val']]
    
    # valid 데이터에서 50% 샘플링하여 train에 추가
    valid_sample = valid_data.sample(frac=ratio, random_state=42)
    new_train_data = pd.concat([train_data, valid_sample])
    new_valid_data = valid_data.drop(valid_sample.index).reset_index(drop=True)
    
    # 인덱스 재설정은 필요 시 이후에 수행
    new_train_data = new_train_data.reset_index(drop=True)
    
    new_train_data['split'] = split_dict['train']
    meta_data = pd.concat([
        new_train_data,
        new_valid_data,
        meta_data[meta_data['split'] == split_dict['test']]
    ]).reset_index(drop=True)
    return meta_data


if __name__ == "__main__":
    pass
    # split_dict = {
    #     'train': 0,
    #     'val': 1,
    #     'test': 2
    # }
    # def train_val_split2(meta_data, split_dict: dict, ratio: float):
    #     # 기존의 train 데이터만 따로 분리
    #     train_data = meta_data[meta_data['split'] == split_dict['train']]
    #     valid_data = meta_data[meta_data['split'] == split_dict['val']]
        
    #     # valid 데이터에서 50% 샘플링하여 train에 추가
    #     valid_sample = valid_data.sample(frac=ratio, random_state=42)
    #     new_train_data = pd.concat([train_data, valid_sample])
    #     new_valid_data = valid_data.drop(valid_sample.index).reset_index(drop=True)
        
    #     # 인덱스 재설정은 필요 시 이후에 수행
    #     new_train_data = new_train_data.reset_index(drop=True)
        
    #     new_train_data['split'] = split_dict['train']
    #     meta_data = pd.concat([
    #         new_train_data,
    #         new_valid_data,
    #         meta_data[meta_data['split'] == split_dict['test']]
    #     ]).reset_index(drop=True)
    #     return meta_data
    
    # import tqdm
    # RATIO = [0.5, 0.6, 0.7, 0.8, 0.9]
    # for ratio in tqdm.tqdm(RATIO):
    #     CelebA_meta_data = pd.read_csv(os.path.join("./data/", "CelebA", "list_attr_celeba.csv"))
    #     split_df = pd.read_csv(os.path.join("./data/", "CelebA", "list_eval_partition.csv"))
    #     CelebA_meta_data['split'] = split_df['partition']
    #     shuffle_CelebA = train_val_split2(CelebA_meta_data, split_dict, ratio=ratio)
    #     shuffle_CelebA.to_csv(os.path.join("./data/", "CelebA", f"list_attr_celeba_{ratio}.csv"))
        
    #     Waterbirds_meta_data = pd.read_csv(os.path.join("./data/", "Waterbirds", "metadata.csv"))
    #     Shufflle_Waterbirds = train_val_split2(Waterbirds_meta_data, split_dict, ratio=ratio)
    #     Shufflle_Waterbirds.to_csv(os.path.join("./data/", "Waterbirds", f"metadata_{ratio}.csv"))
        
    #     Civil_meta_data = pd.read_csv(os.path.join("./data/", "CivilComments","all_data_with_identities.csv"))    
    #     shuffle_Civil = train_val_split2(Civil_meta_data, split_dict, ratio=ratio)
    #     shuffle_Civil.to_csv(os.path.join("./data/", "CivilComments", f"all_data_with_identities_{ratio}.csv"))
        
    #     MultiNLI_meta_data = pd.read_csv(os.path.join("./data/", "MultiNLI", "metadata_random.csv"))
    #     shuffle_MultiNLI = train_val_split2(MultiNLI_meta_data, split_dict, ratio=ratio)
    #     shuffle_MultiNLI.to_csv(os.path.join("./data/", "MultiNLI", f"metadata_random_{ratio}.csv"))