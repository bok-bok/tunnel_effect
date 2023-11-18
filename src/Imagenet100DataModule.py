
'''
class Imagenet100DataModule(pl.LightningDataModule):
    def __init__(self, resolution_size, batch_size, num_workers, classes_num=None, use_all=True):
        super().__init__()
        
        self.train_dir = os.path.join(IMAGENET_100_DIR, "train")
        self.test_dir  = os.path.join(IMAGENET_100_DIR, "val")
        
        self.resolution_size = resolution_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes_num = classes_num
        self.use_all = use_all
        
    def get_ImageNet100_transforms(self,image_size):
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return train_transform, test_transform

    def setup(self, stage=None):
        
        data_name = "imagenet100"
        train_transform, test_transform = self.get_ImageNet100_transforms(self.resolution_size)
        
        train_dataset = datasets.ImageFolder(root = self.train_dir, transform = train_transform)
        
        
        if stage == "fit":
            train_samples_per_class = 200 if self.use_all else None
            train_dataset  = datasets.ImageFolder(root = self.train_dir,  transform = train_transform)
            train_indices = get_balanced_indices(train_dataset, data_name, "train", train_samples_per_class, self.classes_num)
            print(f"train indices: {len(train_indices)}")
            self.train_subset = Subset(train_dataset, train_indices)

        
        if stage == "test":

            test_samples_per_class = 50 if self.use_all else None
            test_dataset  = datasets.ImageFolder(root = self.test_dir,  transform = test_transform)
            test_indices = get_balanced_indices(test_dataset, data_name, "val", test_samples_per_class, self.classes_num)
            print(f"test indices: {len(test_indices)}")
            self.test_subset = Subset(test_dataset, test_indices)

        
        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(self.train_subset, batch_size = self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_subset, batch_size = self.batch_size, shuffle=False, num_workers=4)
'''