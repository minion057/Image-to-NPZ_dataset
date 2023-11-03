import os
from IPython.display import clear_output

import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

import traceback
try:
    # data augmentation해서 npz로 저장할 거니까 필요없음
    from utils import load_npz_dataset
except:
    try:
        from utils.utils import load_npz_dataset
    except:
        print(f'Please check if "utils.py" exists, and if it does, add its path to sys.path.')
        print(traceback.format_exc())


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
    if len(images) != 3: raise ValueError('This function requires three images: Original image, Image in progress, and Transformed image.')
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
    show_data2img_for_DA([image, ycbImage, ycbImage_bgr])
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


def bright_DA_run(npz_file_path, class_label:list=[0,1], class_aug:list=[1, 0], betalist:list=[100, -100, 200, -200, 150, -150, 50, -50],
                  save_dir:str=None, save:bool=False):
    if save_dir is not None : save = True
    for path in npz_file_path:
        print(f'Current file is {path} ===========================================================')
        # load data
        class_names, x_train, x_valid, x_test, y_train, y_valid, y_test = load_npz_dataset(path)
        print(f' Before shape : {x_train.shape, y_train.shape}')
        
        # chage RGB
        if len(x_train.shape) == 3:
            x_train = np.tile(np.expand_dims(x_train, axis=-1), [1, 1, 1, 3])
        elif len(x_train.shape) == 4 and x_train.shape[-1] == 4:
            x_train = x_train[..., :3]
        
        bright = 'bright'
        new_data, new_labels = copy.deepcopy(x_train), copy.deepcopy(y_train)
        
        max_aug = max(class_aug)
        label_data_index_list = [np.where(np.array(y_train) == label)[0] for label in class_label]
        count_list = [len(new_labels)]
        for index, label in enumerate(class_label):
            bright += f'-{label}_x{class_aug[index]}'
            
            # 단순하게 label 기준으로 코드 설계
            # 나중에 제너럴하게 수정
            if len(label_data_index_list[index]) == 0 : raise ValueError(f'Not found data of {class_label}_class.')
            if label in [1] or len(class_label) == 1 or (label==2 and class_aug[1]==class_aug[2]):  
                if class_aug[label] == 0: continue
                count_list.append(label)
                for b in betalist[:class_aug[index]]:
                    x, y = make_a_brightness_data(x_train, label_data_index_list[index], label, b)
                    new_data = np.concatenate((new_data, x), axis=0)
                    new_labels = np.concatenate((new_labels, y), axis=0)
                    
                    count_list.append((len(label_data_index_list[index]), new_labels.shape))
            else:            
                if class_aug[label] == 0: continue
                split_num = int(len(label_data_index_list[index])/max_aug)
                label_data = np.random.permutation(label_data_index_list[index])
                label_data = [label_data[split_num*i:(len(label_data) if i+1 == max_aug and split_num*(i+1) != len(label_data) else split_num*(i+1))] for i in range(max_aug)]
                label_data_index = [label_data]
                
                for i in range(1, class_aug[index]):
                    label_data = np.random.permutation(label_data_index_list[index])
                    label_data_index.append([])
                    choice_list=[]
                    for cnt, item in enumerate(label_data_index[0]):
                        # cnt번에 속한는 item 내용 삭제
                        choice_list.append([ld for ld in label_data if ld not in item])
                        # 이전에 뽑힌 아이템 삭제    
                        for prev_item in label_data_index[i]:
                            choice_list[-1] = [ld for ld in choice_list[-1] if ld not in prev_item]
                        # 마지막이 아니라면 split_num만큼 랜덤 추출
                        label_data_index[i].append(np.random.choice(choice_list[-1], split_num, replace=False) if split_num < len(choice_list[-1]) else np.array(choice_list[-1]))
                    # 마지막까지 안 들어간 아이템 찾아서 추가
                    choice_list.append([ld for ld in label_data])
                    for prev_item in label_data_index[i]:
                        print(f'{len(choice_list[-1])}/{len(label_data)}. {choice_list[-1]}')
                        print(prev_item)
                        print()
                        choice_list[-1] = [ld for ld in choice_list[-1] if ld not in prev_item]
                    print(f'last item : {choice_list[-1]}')
                    while True:
                        if len(choice_list[-1]) == 0: break
                        add_item = copy.deepcopy(choice_list[-1])
                        for choice_item in add_item:
                            for cnt, prev_choice_list in enumerate(choice_list[:-1]):
                                if choice_item in prev_choice_list:
                                    label_data_index[i][cnt] = np.append(label_data_index[i][cnt], choice_item)
                                    choice_list[-1].remove(choice_item)
                                    break
                            
                # 아이템 확인
                for i, item in enumerate(label_data_index):
                    print(i)
                    for it in item:
                        print(f'\t{it} -> {len(it)}')                
        
                # 중복되지 않은 아이템으로 베타 설정하여 augmentataion
                count_list.append(label)
                for label_data in label_data_index:
                    for i, b in enumerate(betalist[:max_aug]):
                        x, y = make_a_brightness_data(x_train, label_data[i], label, b)
                        new_data = np.concatenate((new_data, x), axis=0)
                        new_labels = np.concatenate((new_labels, y), axis=0)                            
                        count_list.append((len(label_data[i]), new_labels.shape))
                        
        new_data, new_labels = shuffle_dataset(new_data, new_labels)       
        print(f'After shape : {new_data.shape, new_labels.shape}')
        
        if save:
            npz_file_name = path.split('/')[-1][:-len('.npz')]
            save_dir_ = os.path.join(save_dir, npz_file_name)
            createDirectory(save_dir_)
            np.savez(os.path.join(save_dir_, bright), class_names=np.array(class_names),
                      x_train=new_data, x_valid=x_valid, x_test=x_test, y_train=new_labels, y_valid=y_valid, y_test=y_test)
        
        print(f'save...\n')
''' Brightness ====== End ====== '''