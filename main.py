import cv2
import matplotlib.pyplot as plt
import os
import glob
import warnings
warnings.filterwarnings("ignore")

data_dir = "data1/"# the folder path
os.listdir(data_dir)
train_dir = f'{data_dir}Train'
test_dir = f'{data_dir}Test'
train_frames = glob.glob(f'{train_dir}/*.jpg')
test_frames = glob.glob(f'{test_dir}/*.jpg')

train_images=np.zeros((256,256,len(train_frames)))
test_images=np.zeros((256,256,len(test_frames)))
for i in range(len(train_frames)):
    train_images[:,:,i] = cv2.imread(train_frames[i],cv2.IMREAD_GRAYSCALE)/255
for i in range(len(test_frames)):
    test_images[:,:,i] = cv2.imread(test_frames[i],cv2.IMREAD_GRAYSCALE)/255
    
lgb_algorithm = LGB(train_images, test_images,(2,2), 1e-4,2)
lgb_algorithm.train()
lgb_algorithm.test()
lgb_algorithm.print_result()
for i in range(len(test_frames)):
    lgb_algorithm.show_result(0)
