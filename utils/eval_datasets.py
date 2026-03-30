import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np

class CroPADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok-vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_train2014_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_val2014_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        return {
            "image": image,
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
        }

class AugmentedCroPADataset(Dataset):
    def __init__(self, original_subset, use_cutmix=False):
        self.original = original_subset
        self.original_len = len(original_subset)
        self.use_cutmix = use_cutmix

    def __len__(self):
        return self.original_len

    def cutmix(self, x1, x2) -> Image.Image:
        assert x1.size == x2.size, "Images must be the same size"
        
        width, height = x1.size
        w = random.randint(width // 4, width // 2)
        h = random.randint(height // 4, height // 2)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)

        np_x1 = np.array(x1)
        np_x2 = np.array(x2)

        M = np.zeros((height, width), dtype=np.uint8)
        M[y:y+h, x:x+w] = 1

        if len(np_x1.shape) == 3:
            M = np.expand_dims(M, axis=-1)

        mixed = np_x1 * (1 - M) + np_x2 * M
        mixed = mixed.astype(np.uint8)

        return Image.fromarray(mixed)
    
    def scmix(self, x1, x2) -> Image.Image:
        def random_crop(img):
            return transforms.RandomResizedCrop((224, 224))(img)
            
        x11 = random_crop(x1)
        x12 = random_crop(x1)

        to_tensor = transforms.ToTensor()
        X1_t = to_tensor(x11)
        X2_t = to_tensor(x12)
        I2_t = to_tensor(x2)
        
        # Blend images
        alpha = 0.5
        blended = 0.75*((alpha) * X1_t + (1-alpha) * X2_t) + 0.25 * I2_t

        transform_pil = transforms.ToPILImage()

        blended_img = transform_pil(blended)
        return blended_img
        
    def __getitem__(self, idx):
        i = idx
        j = torch.randint(self.original_len, (1,)) # Prob of scmix augmentation is (n-1)/n
        
        if (i == j):
            return self.original[i]

        data_i = self.original[i]
        data_j = self.original[j]
        
        I1 = data_i['image'].convert('RGB')
        I2 = data_j['image'].convert('RGB')
        
        def resize(img):
            return transforms.Resize((224,224))(img)

        I1 = resize(I1)
        I2 = resize(I2)
        
        return {
            "image": self.cutmix(I1, I2) if self.use_cutmix else self.scmix(I1, I2),
            "question": data_i['question'],
            "answers": data_i['answers'],
            "question_id": data_i['question_id'],
        }
