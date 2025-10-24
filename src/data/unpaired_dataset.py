import os, random, glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def list_images_recursive(folder: str):
    exts = ["jpg","jpeg","png","bmp","tif","tiff","JPG","JPEG","PNG","BMP","TIF","TIFF"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    return sorted(files)

def build_transform(image_size=512, is_train=True):
    ts = []
    if is_train:
        ts += [T.Resize((image_size, image_size)), T.RandomHorizontalFlip(0.5)]
    else:
        ts += [T.Resize((image_size, image_size))]
    ts += [T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)]
    return T.Compose(ts)

class UnpairedImageDataset(Dataset):
    """
    Accepts nested structure, e.g.:
      root/day/GOPR0345/*.jpg
      root/day/GOPR0374/*.jpg
      root/night/GOPR0351/*.jpg
    """
    def __init__(self, root:str, domain_a:str="night", domain_b:str="day", image_size:int=512, train:bool=True):
        self.dir_a = os.path.join(root, domain_a)
        self.dir_b = os.path.join(root, domain_b)
        self.a_files = list_images_recursive(self.dir_a)
        self.b_files = list_images_recursive(self.dir_b)
        if not self.a_files or not self.b_files:
            raise FileNotFoundError(f"No images found under: {self.dir_a} or {self.dir_b}")
        self.ta = build_transform(image_size, is_train=train)
        self.tb = build_transform(image_size, is_train=train)

    def __len__(self):
        return max(len(self.a_files), len(self.b_files))

    def __getitem__(self, idx):
        a_path = self.a_files[idx % len(self.a_files)]
        b_path = random.choice(self.b_files)
        a = Image.open(a_path).convert("RGB")
        b = Image.open(b_path).convert("RGB")
        return {"A": self.ta(a), "B": self.tb(b), "A_path": a_path, "B_path": b_path}
