import torch

from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader, RandomSampler,Subset


######################################## datasets ######################################################

def load_dataset(root_dir, mini_train,data_type, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""
    if data_type=='train':
        dataset = OriginalDataset(root_dir,transform=True)
    else:
        dataset = OriginalDataset(root_dir,transform=False)
    return dataset
    
    # Use batch size of 1, if requested (e.g. test set)
    
    # if single:
    #     return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    # else:
    #     if mini_train == True and data_type=='train':
    #         num_train_samples = 32000
    #         sample_ds = Subset(dataset, torch.arange(num_train_samples))
    #         sample_sampler = RandomSampler(sample_ds)
    #         print(sample_ds.shape)
    #         # sample_dl = DataLoader(sample_ds, sampler=sample_sampler, batch_size=batch_size)
    #         return sample_sampler
    #     else:
    #         num_val_samples = 640
    #         sample_ds = Subset(dataset, torch.arange(num_val_samples))
    #         sample_sampler = RandomSampler(sample_ds)
    #         # sample_dl = DataLoader(sample_ds, sampler=sample_sampler, batch_size=batch_size)
    #         return sample_sampler

class OriginalDataset(Dataset):
    def __init__(self, root_dir,transform=False):
        """Initializes abstract dataset."""
        self.root_dir = root_dir
        self.noisy_imgs_1, self.noisy_imgs_2 = torch.load(self.root_dir)
        self.transform = transform

    def __len__(self):
        """Returns length of dataset."""

        return len(self.noisy_imgs_1)

    def transformer(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(40, 40))
        image = resize(image)
        mask = resize(mask)

        # crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(32, 32))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # horizontal flipping
        image = TF.hflip(image)
        mask = TF.hflip(mask)

        # vertical flipping
        image = TF.vflip(image)
        mask = TF.vflip(mask)

        return image, mask # Todo, whether the datanumber is increased?

    def __getitem__(self, index):
        """Retrieves image from data folder."""
        
        a = self.noisy_imgs_1[index]/255
        b = self.noisy_imgs_2[index]/255

        if self.transform and index%2==0:
            a,b = self.transformer(a,b) # TODO, keep the same transform for a and b

        return a,b
