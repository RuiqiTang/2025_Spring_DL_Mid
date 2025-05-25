import os 
import zipfile
import tarfile

def Task1Proprocess(filebase:str="Task1/data"):
    
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