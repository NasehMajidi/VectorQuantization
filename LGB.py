
###############################################################################
################################### Imports ###################################
###############################################################################

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set(style='whitegrid',font_scale=3)
rcParams['figure.figsize'] = 22 , 10

###############################################################################
################################## LGB Class ##################################
###############################################################################

class LGB:
    def __init__(self, train_data, test_data,block, error, q_bit):
        self.train_data = train_data
        self.test_data = test_data
        self.error = error
        self.q_bit = q_bit
        self.x = block[0]
        self.y = block[1]
        
        """
        Goal: Implementing the LGB algorithm
        
        Input:
                - train_data: training data
                - test_data: testing data
                - block: the images are divided into some blocks and this variable determines its dimensions
                - error: the threshold of the acceptable error
                - q_bit: # of qunatization bits
                - x: first dimension of block
                - y: second dimension of block
        """
    
    @staticmethod
    def find_closest_center(x, centroids):
        """
        Goal: Find the closest center (centroid) for each input vector
        
        Inputs: 
                - x: the input vector
                - centroids: all centroid vectors
                
        Output:
                - the index of closest centroid
        """
        return np.argmin(np.linalg.norm(x - centroids, axis=1))

    
    def find_centroid(self,X, initial_centroids):
        """
        Goal: find the centroids based on input
        
        Input:
                - X: the input vectors
                - initial_centroids: the initial values of the centroids
                
        Output:
                - cluster_ids: the cluster ids of all the vectors
                - centroids: the centroids of all the vectors
                - j: number of iterations
        """
        K = initial_centroids.shape[0]
        centroids = np.array(initial_centroids)
        cluster_ids = None
        epsilon = 1
        d = []
        j = 0
        while(epsilon>self.error):
            cluster_ids = np.array([self.find_closest_center(X[i], centroids) for i in range(X.shape[0])])
            for k in range(K):
                if len(X[cluster_ids==k]):
                    centroids[k] = np.nanmean(X[cluster_ids==k], axis=0)
            summ = 0
            for i in range(K):
                summ= summ +np.nanmean(np.linalg.norm(X[cluster_ids==i]- centroids[i], axis=1))
            d.append(summ)
            if (j):
                epsilon = abs(d[j]-d[j-1])/d[j]
            j = j+1
        return cluster_ids,centroids,j
    
    def encode(self,image , centroids):
        """
        Gaol: encode (compress) an image
        
        Inputs:
                - image: the input image
                - centroids: all the centroids determined in the training phase
        
        Output:
                - comp_image: the compressed (quantized) image
        """
        comp_img = np.zeros((image.shape[0] // self.x, image.shape[1] // self.y))
        ind_x = 0
        for i in range(0,image.shape[0],self.x):
            ind_y=0
            for j in range(0,image.shape[1],self.y):
                temp = image[i:i+self.x , j:j+self.y] #spliting the image into block
                temp2 = temp.reshape((self.x*self.y)) #vectorizing
                cntr_indx = self.find_closest_center(temp2,centroids)
                comp_img[ind_x,ind_y] = cntr_indx
                ind_y = ind_y+1
            ind_x=ind_x+1
        return comp_img
    
    def decode(self, q_image , centroids):
        """
        Goal: decoede (decompress) an image
        
        Inputs:
                - q_image: the compressed image
                - centroids: all the centroids determined in the training phase
        
        Output:
                - decomp_image: the decompressed image
        """
        decom_img = np.zeros((q_image.shape[0]*self.x , q_image.shape[1]*self.y))
        ind_x = 0
        for i in range(q_image.shape[0]):
            ind_y = 0
            for j in range(q_image.shape[1]):
                temp = q_image[i,j] 
                cntr = centroids [int(temp)]
                cntr = cntr.reshape((self.x,self.y))
                decom_img[ind_x:ind_x+self.x , ind_y: ind_y +self.y] = cntr
                ind_y = ind_y+self.y
            ind_x = ind_x + self.x
        return decom_img
    
    @staticmethod
    def PSNR(img1,img2):
        """
        Goal: Calculating PSNR between two images
        
        Input:
                - img1: the first image
                - img2: the second image
        
        Output:
                - out: PSNR
        """
        MSE = (np.linalg.norm(img1-img2)**2)/(img1.shape[0]*img1.shape[1] + 1e-6)
        MAX=np.max(img1)**2
        out = 10*math.log10(MAX/(MSE+1e-6))
        return out
    
    def train(self):
        """
        Goal: run the training phase
        
        Input: /
        
        Output: / 
        """
        print('---------Train Phase---------')
        train_vec=[]
        for i in tqdm(range(self.train_data.shape[2])):
            img = self.train_data[:,:,i]
            for i in range(0, img.shape[0], self.x):
                for j in range(0, img.shape[1], self.y):
                    train_vec.append(img[i:i + self.x, j:j + self.y].reshape((self.x * self.y)))
        self.train_vec = np.array(train_vec)
        initial_vec  = np.random.permutation(train_vec)[:2**self.q_bit]
        ID , self.centroids ,it= self.find_centroid(self.train_vec,initial_vec)
    
    def test(self):
        """
        Goal: run the testing phase
        
        Input: /
        
        Output: / 
        """
        print('---------Test Phase---------')
        test_vec = []
        self.encoded_img_vec = []
        self.decoded_img_vec = []
        self.psnr_vec = []
        for i in tqdm(range(self.test_data.shape[2])):
            img = self.test_data[:,:,i]
            encoded_img = self.encode(img , self.centroids)
            decoded_img = self.decode(encoded_img, self.centroids)
            psnr = self.PSNR(img, decoded_img)
            
            self.encoded_img_vec.append(encoded_img)
            self.decoded_img_vec.append(decoded_img)
            self.psnr_vec.append(psnr)
        self.final_psnr = np.mean(self.psnr_vec)
        
    def print_result(self):
        """
        Goal: print a summary of the running algorithm
        
        Input: /
        
        Output: / 
        """
        print('Summary:')
        print(f"          block:({self.x} , {self.y})")
        print(f"          Quantization bits: {self.q_bit}")
        print(f"          PSNR: {self.final_psnr}")        

    def show_result(self,indx):
        """
        Goal: compare original and decoded image
        
        Input:
                -indx: the index of the image
        
        Output: / 
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.title.set_text('Original Image')
        ax2.title.set_text('Decoded Image')
        ax1.imshow(self.test_data[:,:,indx], cmap = 'gray')
        ax2.imshow(self.decoded_img_vec[indx], cmap = 'gray')
        plt.show()
