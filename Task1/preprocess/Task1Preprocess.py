import os 
import zipfile
import tarfile
import shutil
import random
import torch 
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset

def Task1_Zipfile_Proprocess(filebase:str="Task1/data"):
    
    zip_path=os.path.join(filebase,'caltech-101.zip')
    extract_dir=os.path.join(filebase,'caltect101')
    os.makedirs(extract_dir,exist_ok=True)
    
    # Extract zip file
    with zipfile.ZipFile(zip_path,'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    # Extract tar file
    tar_path=os.path.join(extract_dir,'caltech-101/101_ObjectCategories.tar.gz')
    extract_dir2=os.path.join(extract_dir,'images')
    os.makedirs(extract_dir2,exist_ok=True)
    
    with tarfile.open(tar_path,'r:gz') as tar:
        tar.extractall(path=extract_dir2)
    print("Extract all!")
    return

class Task1Dataloader:
    def __init__(self, datafilebase='Task1/data/caltech101/images',
                 train_batch_size=32, val_batch_size=32, num_workers=2,
                 train_samples_per_class=30):
        self.original_data_dir = os.path.join(datafilebase,'101_ObjectCategories')
        self.filtered_data_dir = os.path.join(datafilebase,'FilteredCategories')
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.train_samples_per_class = train_samples_per_class
        
        # 创建数据增强与预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        # 初始化数据集和数据加载器
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.class_to_idx = None
        
    def prepare_data(self):
        # 过滤掉BACKGROUND_Google类别
        self._filter_categories()
        self.full_dataset = datasets.ImageFolder(self.filtered_data_dir, transform=self.transform)
        self.class_to_idx = self.full_dataset.class_to_idx
        self._split_dataset()
        self._create_data_loaders()
        
        return self.train_loader, self.val_loader
    
    def _filter_categories(self):
        os.makedirs(self.filtered_data_dir, exist_ok=True)
        
        for cls in os.listdir(self.original_data_dir):
            src = os.path.join(self.original_data_dir, cls)
            dst = os.path.join(self.filtered_data_dir, cls)
            if os.path.isdir(src) and cls != 'BACKGROUND_Google':
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
    
    def _split_dataset(self):
        train_indices, val_indices = [], []
        targets = self.full_dataset.targets
        
        for cls_idx in range(len(self.class_to_idx)):
            cls_samples = [i for i, t in enumerate(targets) if t == cls_idx]
            random.shuffle(cls_samples)
            train_indices.extend(cls_samples[:self.train_samples_per_class])
            val_indices.extend(cls_samples[self.train_samples_per_class:])
        
        self.train_dataset = Subset(self.full_dataset, train_indices)
        self.val_dataset = Subset(self.full_dataset, val_indices)
        
        print(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")
    
    def _create_data_loaders(self):
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            num_workers=self.num_workers
        )    