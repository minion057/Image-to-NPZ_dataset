from IPython.display import clear_output

import numpy as np
import cv2
import matplotlib.pyplot as plt



def shuffle_dataset(dataset, labelset):
    '''
    A function that randomly shuffles the dataset.
    (Used to shuffle the order along with the labels when combining the original data with augmented data.)
        Args:
            dataset : (N, ?, ?, ...)  
            labelset : (N, )
        Retruns:
            shuffled_dataset, shuffled_labelset
    '''
    if len(dataset) != len(labelset): raise ValueError('The lengths of the data and labels must be the same.')
    # Both the data and labels must be shuffled in the same order.
    # To achieve this, we shuffle the indices to ensure that the same indices are used for both the data and labels.
    shuffled_index_list = np.random.permutation(len(dataset))
    return dataset[shuffled_index_list], labelset[shuffled_index_list]


def show_data2img_for_DA(images, figsize:tuple=(10, 10)):
    '''
    Visualize the dataset converted to numpy images.
        Args:
            images (list) : This function requires three images: 'Original image', 'Image in progress', 'Transformed image'
        Returns: 
            None (Displaying 'Original image,' 'Image in progress,' and 'Transformed image' as images.)
    '''
    if len(image) != 3: raise ValueError('This function requires three images: Original image, Image in progress, and Transformed image.')
    if len(figsize) != 2: raise ValueError('figsize only accepts width and height as parameters.')
    title_list = ['Original image', 'Image in progress', 'Transformed image']
    plt.figure(figsize=figsize)
    for i, x in enumerate(images):
        if i==0:        print("Image shape: ", x.shape)
        plt.subplot(3, 3, i + 1)
        plt.title(title_list[i])
        plt.imshow(x)
        plt.axis("off")
    plt.show()
  


''' Brightness ====== Start ====== '''
def brightness(image, beta:int=100):
    '''
    Separate one image into channels and adjust its brightness.
        Args:
            images : Image for brightness adjustment.
            beta (int) : Value for brightness adjustment.
                        (Higher values make it brighter, lower values make it darker. Recommended between -225 and 225.)
        Returns: 
            ycbImage_bgr : The image with brightness adjustment.
    '''
    if beta < -225 or beta > 225 or beta == 0: raise Exception('Please specify a value. (-225 =< value < 0 && 0 < value <= 225)')
    
    # Change color space
    ycbImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Convert from uint8 to float for processing
    ycbImage = np.float32(ycbImage)
    # Split the channels
    Ychannel, Cr, Cb = cv2.split(ycbImage)
    # Adjust brightness
    Ychannel_ = np.clip(Ychannel + beta, 0, 255)
    # Merge the channels
    ycbImage = cv2.merge([Ychannel_, Cr, Cb])
    # Convert back to integer
    ycbImage = np.uint8(ycbImage)
    # Convert to BGR
    ycbImage_bgr = cv2.cvtColor(ycbImage,cv2.COLOR_YCrCb2BGR)
    ycbImage_bgr = np.uint8(ycbImage_bgr)
    show_multi([image, ycbImage, ycbImage_bgr])
    return ycbImage_bgr


def make_a_brightness_data(data:list, DA_index_list:list, label_of_class=None, beta:int=100):
    '''
    Perform data augmentation using the 'brightness' function.
        Args:
            data (list) : The complete dataset for one class.
            DA_index_list (list) : The list of indices for data augmentation to be performed on in the 'data'.
            label_of_class : The label of the class.
            beta (int) : Value for brightness adjustment.
                        (Higher values make it brighter, lower values make it darker. Recommended between -225 and 225.)
        Returns: 
            new_data, new_labels : The dataset with brightness adjustment and its corresponding labels.
    '''
    new_data = []
    for i in DA_index_list:
        clear_output(wait=True)
        new_data.append(brightness(data[i], beta))
    new_data = np.array(new_data)
    new_labels = np.full((len(DA_index_list),), label_of_class)
    return new_data, new_labels
''' Brightness ====== End ====== '''