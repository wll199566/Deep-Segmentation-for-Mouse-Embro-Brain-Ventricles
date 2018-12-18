
# coding: utf-8

# In[ ]:


import random
from scipy import ndimage
import torch


class Rotate(object):
    
    """
    Rotate the image for data augmentation, but prefer original image.
    """
    
    def __init__(self,ori_probability=0.30):
        self.ori_probability = ori_probability
        #1:(0,0,0), 2:(90,0,0), 3:(180,0,0), 4:(270,0,0), 5:(0,90,0), 6:(0,270,0)
        self.face_to_you = [1,2,3,4,5,6]
        #rotate along z axis, so that we 24 combination totally
        # 1:0 degree, 2: 90 degree, 3: 180 degree, 4: 270 degree
        self.rotate_z = [1,2,3,4]

    def __call__(self, sample):
        if random.uniform(0,1) < self.ori_probability:
            return sample
        else:
            img, label = sample['image'], sample['label']
            random_choise1=random.choice(self.face_to_you)
            rotated_img1 = {1: lambda x: x,
                            2: lambda x: ndimage.rotate(x,90,(1,2),reshape='True',mode = 'nearest'),
                            3: lambda x: ndimage.rotate(x,180,(1,2),reshape='True',mode = 'nearest'),
                            4: lambda x: ndimage.rotate(x,270,(1,2),reshape='True',mode = 'nearest'),
                            5: lambda x: ndimage.rotate(x,90,(0,2),reshape='True',mode = 'nearest'),
                            6: lambda x: ndimage.rotate(x,270,(0,2),reshape='True',mode = 'nearest')
                            }[random_choise1](img[0,...])
            #rotated_fil_img1 = {1: lambda x: x,
            #                2: lambda x: ndimage.rotate(x,90,(1,2),reshape='True',mode = 'nearest'),
            #                3: lambda x: ndimage.rotate(x,180,(1,2),reshape='True',mode = 'nearest'),
            #                4: lambda x: ndimage.rotate(x,270,(1,2),reshape='True',mode = 'nearest'),
            #                5: lambda x: ndimage.rotate(x,90,(0,2),reshape='True',mode = 'nearest'),
            #                6: lambda x: ndimage.rotate(x,270,(0,2),reshape='True',mode = 'nearest')
            #                }[random_choise1](img[1,...])
            rotated_label1 = {1: lambda x: x,
                            2: lambda x: ndimage.rotate(x,90,(1,2),reshape='True',mode = 'nearest'),
                            3: lambda x: ndimage.rotate(x,180,(1,2),reshape='True',mode = 'nearest'),
                            4: lambda x: ndimage.rotate(x,270,(1,2),reshape='True',mode = 'nearest'),
                            5: lambda x: ndimage.rotate(x,90,(0,2),reshape='True',mode = 'nearest'),
                            6: lambda x: ndimage.rotate(x,270,(0,2),reshape='True',mode = 'nearest')
                            }[random_choise1](label[0,...])
            
            random_choise2=random.choice(self.rotate_z)
            img[0,...] = {1: lambda x: x,
                            2: lambda x: ndimage.rotate(x,90,(0,1),reshape='True',mode = 'nearest'),
                            3: lambda x: ndimage.rotate(x,180,(0,1),reshape='True',mode = 'nearest'),
                            4: lambda x: ndimage.rotate(x,270,(0,1),reshape='True',mode = 'nearest')
                            }[random_choise2](rotated_img1)
            #img[1,...] = {1: lambda x: x,
            #                2: lambda x: ndimage.rotate(x,90,(0,1),reshape='True',mode = 'nearest'),
            #                3: lambda x: ndimage.rotate(x,180,(0,1),reshape='True',mode = 'nearest'),
            #                4: lambda x: ndimage.rotate(x,270,(0,1),reshape='True',mode = 'nearest')
            #                }[random_choise2](rotated_fil_img1)
            label[0,...] = {1: lambda x: x,
                            2: lambda x: ndimage.rotate(x,90,(0,1),reshape='True',mode = 'nearest'),
                            3: lambda x: ndimage.rotate(x,180,(0,1),reshape='True',mode = 'nearest'),
                            4: lambda x: ndimage.rotate(x,270,(0,1),reshape='True',mode = 'nearest')
                            }[random_choise2](rotated_label1)
            
        return {'image': img, 'label': label}

class Flip(object):
    
    """
    Flip the image for data augmentation, but prefer original image.
    """
    
    def __init__(self,ori_probability=0.30):
        self.ori_probability = ori_probability

    def __call__(self, sample):
        if random.uniform(0,1) < self.ori_probability:
            return sample
        else:
            img, label = sample['image'], sample['label']
            random_choise1=random.choice([1,2,3,4,5,6,7,8])
            img[0,...] = {1: lambda x: x,
                          2: lambda x: x[::-1,:,:],
                          3: lambda x: x[:,::-1,:],
                          4: lambda x: x[:,:,::-1],
                          5: lambda x: x[::-1,::-1,:],
                          6: lambda x: x[::-1,:,::-1],
                          7: lambda x: x[:,::-1,::-1],
                          8: lambda x: x[::-1,::-1,::-1]
                          }[random_choise1](img[0,...])
            #img[1,...] = {1: lambda x: x,
            #              2: lambda x: x[::-1,:,:],
            #              3: lambda x: x[:,::-1,:],
            #              4: lambda x: x[:,:,::-1],
            #              5: lambda x: x[::-1,::-1,:],
            #              6: lambda x: x[::-1,:,::-1],
            #              7: lambda x: x[:,::-1,::-1],
            #              8: lambda x: x[::-1,::-1,::-1]
            #              }[random_choise1](img[1,...])
            label[0,...] = {1: lambda x: x,
                          2: lambda x: x[::-1,:,:],
                          3: lambda x: x[:,::-1,:],
                          4: lambda x: x[:,:,::-1],
                          5: lambda x: x[::-1,::-1,:],
                          6: lambda x: x[::-1,:,::-1],
                          7: lambda x: x[:,::-1,::-1],
                          8: lambda x: x[::-1,::-1,::-1]
                          }[random_choise1](label[0,...])
            
        return {'image': img, 'label': label}

#class ToTensor(object):
#    """Convert ndarrays in sample to Tensors."""
#    #def __init__(self):
#
#    def __call__(self, sample):
#        image, label = sample['image'], sample['label']
#        return {'image': torch.from_numpy(image),
#                'label': torch.from_numpy(label)}
if __name__ == '__main__':
    pass