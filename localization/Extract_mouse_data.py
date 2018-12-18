
# coding: utf-8

# In[ ]:


from torch.utils.data import Dataset, DataLoader
import numpy as np

class Mouse_sub_volumes(Dataset):
    """Mouse sub-volumes BV dataset."""

    def __init__(self, all_whole_volumes, all_whole_filtered_volumes, all_idx, transform=None):
        """
        Args:
            all_whole_volumes: Contain all the padded whole BV volumes as a dic
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.whole_volumes = all_whole_volumes
        self.whole_filtered_volumes = all_whole_filtered_volumes
        self.idx = all_idx
        self.transform = transform
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, num):
        #idx [0] whole image index, [1] label(0 or 1), [2] x, y, z sub-volumes index
        box_size = 64
        current_img = self.idx[num][0]
        label = self.idx[num][1]
        x, y, z = self.idx[num][2]
        img = np.zeros((1,box_size,box_size,box_size),np.float32)
        img[0,...] = self.whole_volumes[current_img][x-box_size:x,
                                                  y-box_size:y,
                                                  z-box_size:z]
        
        #img[1,...] = self.whole_filtered_volumes[current_img][x-box_size:x,
        #                                                        y-box_size:y,
        #                                                        z-box_size:z]
        sample = {'image': img, 'label': label}
        if self.transform:
            sample = self.transform(sample)
            
        return sample

if __name__ == '__main__':
    pass