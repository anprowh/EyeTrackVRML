import torch
import cv2
import numpy as np
from models1 import Regressor

class SVAE:
    def __init__(self):
        self.track_counter = -1
        self.last_coord = (0, 0)
        self.device = 'cuda'
        self.reg = Regressor().to(self.device)
        self.reg.load_state_dict(torch.load('Models/Reg9_SiVQB.pth'))
    
    def __call__(self, current_image_gray, context):
        self.track_counter += 1
        if self.track_counter % 2 == 1:
            return self.last_coord
        img = cv2.resize(current_image_gray, (120,120)).reshape((1,1,120,120)).astype(np.float32)/255
        context.thresh = cv2.resize(img[0,0], current_image_gray.shape)
        self.last_coord = self.reg.predict(img, self.device)[0]
        self.last_coord = (float(self.last_coord[0]), float(self.last_coord[1]))
        return self.last_coord

def svae(self):
    if not hasattr(self, 'svae'):
        self.svae = SVAE()
        
    return self.svae(self.current_image_gray_clean, self)