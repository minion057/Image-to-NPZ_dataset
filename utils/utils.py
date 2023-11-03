import os
import traceback
from tqdm import tqdm
from IPython.display import clear_output

from pathlib import Path
import imghdr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


def createDirectory(dir_path:str):
    '''
    This function checks whether a folder exists and creates the folder at the specified path if it doesn't exist.
        Args:
            dir_path (str): The path of the folder you want to check.  
        Retruns:
            None (Informs you of the result of folder creation.)
    ''' 
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f'To create a folder at the following path. : {dir_path}')
    except OSError:
        print('Error: Failed to create the directory.')
        


''' Check the supported file formats in TensorFlow or PyTorch. ====== Start ====== '''
def image_valid_check(dir_path:str, class_split:bool=False):
    '''
    Check if the file formats in the given directory are supported by the library.
        Args:
            dir_path (str): The root path where the image data is located.  
            class_split (bool): To determine whether to separate the list of supported image formats by class.
                                (지원하는 형식의 이미지 리스트를 클래스별로 나눌 것인지 선택하는 변수)   
        Retruns:
            return (dict)
                1. return['non_image'] : List of non-image files in subdirectories.   
                2. return['non_type']  : List of files in subdirectories that are not in supported image formats.
                3. return['image']     : List of image files in subdirectories. 
                    - class_split==True  : Assign folder names in subdirectories as class names and create separate lists for each class.  
                    - class_split==False : Convert all subdirectory paths into a one-dimensional list and return it.  
    ''' 
    # Supported File Formats
    image_extensions = ['.png', '.jpg']
    img_type_accepted_by_tf = ['bmp', 'gif', 'jpeg', 'png']

    # After validation, store the file path in a variable.
    result = {'non_image':[], 'non_type':[], 'image':[]}

    # Validation
    for filepath in Path(dir_path).rglob('**/*'):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath) # If the file is damaged, it returns None.
            if img_type is None:
                result['non_image'].append(filepath)
            elif img_type not in img_type_accepted_by_tf:
                result['non_type'].append(filepath)
            else:
                if class_split:
                    class_name = str(filepath).split('/')[-2]
                    if type(result['image']) == list:result['image']={}
                    if class_name not in result['image'].keys(): result['image'][class_name] = []
                    result['image'][class_name].append(filepath)
                else: result['image'].append(filepath)

    # Output the result
    print(f'Here are the Validation Check Results.')
    print(f"1. This file is not an image. -> count : {len(result['non_image'])}\n\te.g.{result['non_image'][0] if len(result['non_image']) != 0 else 'None'}")
    print(f"2. This file is not in a format supported by TensorFlow OR Pytorch. -> count : {len(result['non_type'])}\n\te.g.{result['non_type'][0] if len(result['non_type']) != 0 else 'None'}")
    if class_split:
        print(f"3. This file is in a format supported by TensorFlow OR Pytorch. -> count : {sum([len(result['image'][k]) for k in result['image'].keys()])}")
        for k in result['image'].keys():
            print(f"\t class - {k} -> count : {len(result['image'][k])}\n\t\te.g.{result['image'][k][0] if len(result['image'][k]) != 0 else 'None'}")
    else : print(f"3. This file is in a format supported by TensorFlow OR Pytorch. -> count : {len(result['image'])}\n\te.g.{result['image'][0] if len(result['image']) != 0 else 'None'}")
    return result



def rename_the_files(files:list, remove_str:str='._'):
    '''
    Remove suspicious wording and rename it.
    A function that removes the specified substring (remove_str) from the names of all files in the file list.
        Args:
            files (list) : List of file paths.
            remove_str (str) : The string to be removed.
        Returns:
            None
    '''
    for f in files:
        try:
            os.rename(str(f), str(f).replace(remove_str, ''))
        except Exception as ex:
            print(f'You need to confirm if the current path exists.\n\tThe current data path being used is \'{f}\'.\n')
            print(traceback.format_exc())
''' Check the supported file formats in TensorFlow or PyTorch. ====== End ====== '''



''' Make a Dataset ====== Start ====== '''
def make_a_numpy_dataset(imageset:dict, image_W:int=None, image_H:int=None):
    '''
    Convert class-specific image data into a dataset (data, label).
    (클래스별 이미지 데이터를 dataset(data, label)으로 변환시키는 함수)
        Args:
            imageset (dict) : A dictionary where the keys represent class names and the values represent image paths.
            image_ (int) : The width and height of the image, which are required values for resizing.
        Returns:
            data2np : A dictionary where each class's image data is converted into a numpy array.
            classes : A list of class names sorted according to the labels.
    '''
    # 0. data : resize
    img_resize = False
    if image_H != None and image_W != None:
        if image_H > 0 and image_W > 0:
            img_resize = True
            image_size = (image_W, image_H)
    # 1. data : image to numpy
    numpyset = data2np(imageset) if not img_resize else data2np(imageset=imageset, image_size=image_size)
    # 2. label : make a label(int)
    print(f'\nMake a label...')
    classes = list(numpyset.keys())
    labelset = {class_name:np.full((numpyset[class_name].shape[0],), idx) for idx, class_name in enumerate(classes)}
    # 3. after concatenate, return 
    print(f'\nMake a dataset...')
    dataset = {'data':np.array([]), 'label':np.array([])}
    for class_name in classes:
        if len(dataset['data']) == 0:
            dataset['data'], dataset['label'] = numpyset[class_name], labelset[class_name]
        else:
            dataset['data'] = np.append(dataset['data'], numpyset[class_name], axis=0)
            dataset['label'] = np.append(dataset['label'], labelset[class_name], axis=0)
    return dataset, classes


def data2np(imageset, image_size=None):
    '''
    Convert class-specific image data into a numpy dataset (data only).
        Args:
            imageset (dict) : A dictionary where the keys represent class names, and the values represent image paths.
            image_size (tuple) : The width and height of the image, which are required values for resizing.
        Returns:
            data2np_dict : A dictionary where each class's image data is converted into a numpy array.
    '''
    print(f'Convert the images to numpy...')
    data2np_dict = {}
    for class_name in imageset.keys():
        print(f'Attempting to convert the \'{class_name}\' class to a numpy array.')
        if class_name not in data2np_dict.keys(): data2np_dict[class_name] = []
        for image in tqdm(imageset[class_name]):
            img = Image.open(image)
            if image_size is not None: img = img.resize(image_size)
            data2np_dict[class_name].append(np.array(img))
        data2np_dict[class_name] = np.array(data2np_dict[class_name])
        print(f'This Class is {class_name}. -> shape : {data2np_dict[class_name].shape}\n')
    return data2np_dict
''' Make a Dataset ====== End ====== '''



''' To check the dataset ====== Start ====== '''
def show_data2img(datas:list, class_names:list, BGR2RGB:bool=True):
    '''
    Visualize a dataset converted to numpy images.
        Args:
            datas (list) : A subset of the dataset. [[data, label], [data, label], ...]
            class_names (list) : A list of class names sorted according to the labels.
        Returns: 
            None
    '''
    plt.figure(figsize=(10, 10))
    for i, (x, y) in enumerate(datas):
        if i==0: print(f'Image shape: {x.shape}')
        ax = plt.subplot(3, 3, i + 1)
        x_cv2 = cv2.cvtColor(x.astype('uint8'), cv2.COLOR_BGR2RGB) if BGR2RGB else x.astype('uint8')
        plt.imshow(x_cv2)
        plt.title(f'Label : {class_names[y]}')
        plt.axis('off')
    plt.show()


# To calculate class distribution ====== Start ====== 
def format_check_counts(counts:dict={'train':[], 'valid':[], 'test':[]}):
    '''
    Check the format of the 'counts' parameter required for a function to examine the data distribution.
        Args:
            counts (dict) : A dictionary for counting the unique occurrences of 'label or class name, etc.' in the dataset.
                           (e.g. counts = {'train':[1, 2], 'valid':[1, 1], 'test':[2, 2]})
        Returns:
            None
    '''
    need_key_list = ['train', 'valid', 'test']
    param_key_list = list(counts.keys())
    need_key_list.sort()
    param_key_list.sort()
    if need_key_list != need_key_list : raise ValueError("The 'counts' dictionary's keys can only be set to train, valid, or test.")
    for key, value in counts.items():
        if type(value) != np.ndarray: raise TypeError("The values in the 'counts' dictionary must be of type list.")
  

def show_DIST(class_names:list, counts:dict={'train':[], 'valid':[], 'test':[]}, plt_title:str='Class Distribution'):
    '''
    Calculate the percentage of each class in the dataset and display the counts with a bar graph.
        Args:
            class_names (list) : A list of the names of classes present in the dataset.
            counts (dict) : A dictionary for counting the unique occurrences of 'label or class name, etc.' in the dataset.
                           (e.g. counts = {'train':[1, 2], 'valid':[1, 1], 'test':[2, 2]})
        Returns:
            None
    '''
    format_check_counts(counts)
    DIST_pct(class_names, counts, True)
    DIST_countplot(class_names, plt_title, counts, True)
  
  
def DIST_pct(class_names:list, counts:dict={'train':[], 'valid':[], 'test':[]}, format_checked_counts:bool=False):
    '''
    Calculate and display the percentage of each class in each dataset.
        Args:
            class_names (list) : A list of the names of classes present in the dataset.
            counts (dict) : A dictionary for counting the unique occurrences of 'label or class name, etc.' in the dataset.
                           (e.g. counts = {'train':[1, 2], 'valid':[1, 1], 'test':[2, 2]})
            format_checked_counts (bool) : Whether 'format_check_counts' has been performed or not.
        Returns:
            None
    '''    
    if not format_checked_counts: format_check_counts(counts)
    unique_result = {k:np.unique(counts[k], return_counts=True) for k in counts.keys()}
    print(f'Class Distribution')
    for idx, (k, v) in enumerate(counts.items()):
        print(f'{idx}. {k}set - Total {len(v)}')
        for idx, c in enumerate(unique_result[k][0]):
            print(f'\tClass {class_names[c]} : {unique_result[k][1][idx]} -> {np.round((unique_result[k][1][idx]/len(v))*100, 1)}% ({unique_result[k][1][idx]}/{len(v)})')
        
        
def DIST_countplot(class_names:list, title:str, counts:dict={'train':[], 'valid':[], 'test':[]}, format_checked_counts:bool=False,
                   figsize:tuple=(5,5), bar_size:float=0.23, ncol_in_legend:int=4, data_point_count_text_size:float=12.0):
    '''
    Calculate the number of data points taken up by each class in each dataset and display it with a bar graph.
        Args:
            class_names (list) : A list of the names of classes present in the dataset.
            title (str) : Title of the bar graph.
            counts (dict) : A dictionary for counting the unique occurrences of 'label or class name, etc.' in the dataset.
                           (e.g. counts = {'train':[1, 2], 'valid':[1, 1], 'test':[2, 2]})
            format_checked_counts (bool) : Whether 'format_check_counts' has been performed or not.
            figsize (tuple) : The size of the bar graph.
            bar_size (float) : The size of the bars in the bar graph.
            ncol_in_legend (int) : The number of items to be represented in each line of the legend.
            data_point_count_text_size (float) : The size of the text used to display the number of data points.
        Returns:
            None
    '''
    # if not format_checked_counts: format_check_counts(counts)
    counts = {k:np.unique(v, return_counts=True)[-1].tolist() for k, v in counts.items()} 
    x = np.arange(len(class_names))
    plt.figure(figsize=figsize)
    color_list = [[plt.cm.Pastel1(i) for i in range(9)], [plt.cm.Paired(i) for i in range(12)], [plt.cm.Pastel2(i) for i in range(9)]]
    
    colors = iter([j for i in color_list for j in i])
    plt.title(title, fontdict = {'fontsize' : 20}, pad=15)
    plt.xlabel(' ')
    plt.ylabel('count')  
    
    cnt = 0
    for i, (category, score) in enumerate(counts.items()):
        bar = plt.bar(x+(bar_size*i), score, label=str(category), width=bar_size, color=next(colors))
        cnt+=1
        for rect, text in zip(bar, score):
            height = rect.get_height()-25
            plt.text(rect.get_x() + rect.get_width()/2.0, height, text, ha='center', va='bottom', size=data_point_count_text_size)
    plt.xticks(x+((bar_size/2)*(cnt-1)), class_names, fontdict = {'fontsize' : 13})
    plt.legend(loc='lower left', bbox_to_anchor=(0,-0.2,1,0.2), ncol=ncol_in_legend, mode='expand')
    plt.tight_layout()
    plt.show()
# To calculate class distribution ====== End ======
''' To check the dataset ====== End ====== '''



''' Splitting and Save ====== Start ====== '''
# The sklearn function used for data splitting
# train_test_split(data, label, test_size, random_state, shuffle, stratify) 
#   - test_size : 얼마나 자를 건지  
#   - shuffle : dataset 섞을 건지  
#   - random_state : 매번 데이터셋이 변경되는 것을 방지   
#   - stratify : 클래스별 비율을 유지할 것인지 (label로 지정)  
#   - E.g. x_train, x_valid, y_train, y_valid = train_test_split(data, label, test_size=0.2, shuffle=True, stratify=label, random_state=34) 


def save_numpy_dataset(dataset:dict, class_names:list, save_dir:str=None, save_name:str='img2npz'):
    '''
    Save the numpy dataset as an npz file.
        Args:
            dataset (dict) : Numpy-formatted dataset. (If you only have image data, use the 'make_a_numpy_dataset' function.)
            class_names (list) : A list of the names of classes present in the dataset.
            save_dir (str) : The path where the files will be saved.
            save (bool) : Whether to save the files or not.
        Returns:
            None (Save the dataset.)
    '''
    createDirectory(save_dir)
    save_name = save_name if save_name[-len('.npz'):] == '.npz' else f'{save_name}.npz'
    np.savez(f'{save_dir}/{save_name}', class_names=np.array(class_names), data=dataset['data'], label=dataset['label'])
    print(f'save... {save_dir}/{save_name}\n')
    

def make_a_splitset(dataset:dict, class_names:list, splitset_size:float=0.2, random_state:int=1, save_dir:str=None, save:bool=True):
    '''
    Divide the dataset according to its size.
        Args:
            dataset (dict) : Numpy-formatted dataset. (If you only have image data, use the 'make_a_numpy_dataset' function.)
            class_names (list) : A list of the names of classes present in the dataset.
            splitset_size (float) : The ratio to set aside as the test set. (0 < splitset_size < 1)
            random_state (int) : Random seed.
            save_dir (str) : The path where the files will be saved.
            save (bool) : Whether to save the files or not.
        Returns:
            - save == True : None (Save the dataset after performing train-test split in the npz format.)
            - save == False : x_train, x_test, y_train, y_test (The dataset after performing train-test split.)
    '''
    if 0 >= splitset_size or splitset_size >= 1 : raise ValueError("The 'splitset_size' should be set as a floating-point number between 0 and 1.")
    if save: createDirectory(save_dir)
    
    save_name = f'test{splitset_size}.npz'

    print(f'Creating training, validation, and test datasets. -> {save_name}')
    x_train, x_test, y_train, y_test = train_test_split(dataset['data'], dataset['label'], test_size=splitset_size, shuffle=True, stratify=dataset['label'], random_state=random_state)

    print(f'data shape : {x_train.shape}, {x_test.shape}')
    print(f'label shape : {y_train.shape}, {y_test.shape}')

    if save:
        np.savez(f'{save_dir}/{save_name}', class_names=np.array(class_names), x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        print(f'save... {save_dir}/{save_name}\n')
    else: return x_train, x_test, y_train, y_test


def cal_val_ratio(size:tuple=(0.6, 0.2, 0.2), first_split:str='test'):
    '''
    After splitting the entire dataset into training or test sets, this is used to calculate the number of data points for validation.
    (Calculate the number of data points corresponding to the validation ratio based on the entire dataset.)
        Args:
            size (tuple) : The ratio at which you want to split the dataset based on the entire dataset. (train_ratio, validation_ratio, test_ratio)
            first_split (str) : The name of the initially created dataset. ('test' or 'train')
        Returns:
            val_ratio (float) : The recalculated ratio from the remaining dataset to obtain the required number of data points based on the entire dataset.
                                (전체 데이터셋을 기준으로 필요한 데이터 개수를 얻기 위해 남은 데이터셋에서 다시 계산한 비율)
    '''
    if len(size) != 3 : ValueError('Size should be specified with three values.')
    if sum(size) != 1 : ValueError('The sum of the sizes should be equal to 1.')
    train_ratio, validation_ratio, test_ratio = size
    if first_split == 'test':
        print(f'train + val = {1 - test_ratio}')
        val_ratio = validation_ratio/(train_ratio + validation_ratio)
    elif first_split == 'train':
        print(f'test + val = {1 - train_ratio}')
        val_ratio = test_ratio/(test_ratio + validation_ratio)
    else: raise ValueError(f"The 'first_split' option can only come with values of 'test' and 'train'.")
    print(f'val = {val_ratio}')
    return val_ratio
  
# split_trainset  
def make_a_validationset_static_test(x_train:np.ndarray, x_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, class_names:list,
                                     train_val_test_size:tuple=(0.6, 0.2, 0.2), random_state:int=1, save_dir:str=None, save:bool=True):
    '''
    Create a validation set when you already have a training set and a test set.
    (Calculate the number of data points corresponding to the validation ratio based on the entire dataset.
     e.g., There are 100 data points in total, with 80 of them in the training dataset and 20 in the test dataset.    
           You want to use 20% of the entire dataset as the validation dataset.    
           To do this, you should recalculate the validation dataset size based on the training dataset size.)
        Args:
            x_train, x_test, y_train, y_test (np.ndarray) : The dataset created with 'make_a_splitset'
            size (tuple) : The ratio at which you want to split the dataset based on the entire dataset. (train_ratio, validation_ratio, test_ratio)
            first_split (str) : The name of the initially created dataset. ('test' or 'train')
        Returns:
            val_ratio (float) : The recalculated ratio from the remaining dataset to obtain the required number of data points based on the entire dataset.
                                (전체 데이터셋을 기준으로 필요한 데이터 개수를 얻기 위해 남은 데이터셋에서 다시 계산한 비율)
    '''
    val_ratio = cal_val_ratio(train_val_test_size, first_split='test')
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=val_ratio, shuffle=True, stratify=y_train, random_state=random_state)

    print(f'data shape : {x_train.shape}, {x_valid.shape}, {x_test.shape}')
    print(f'label shape : {y_train.shape}, {y_valid.shape}, {y_test.shape}')
    
    train_ratio, validation_ratio, test_ratio = train_val_test_size
    save_name = f'train{train_ratio}val{validation_ratio}test{test_ratio}.npz'
    if save: np.savez(os.path.join(save_dir, save_name), class_names=np.array(class_names), x_train=x_train, x_valid=x_valid, x_test=x_test, y_train=y_train, y_valid=y_valid, y_test=y_test)
    print(f'save... {save_dir}/{save_name}\n')
''' Splitting and Save ====== End ====== '''


''' Load npz dataset ====== Start ====== '''
def load_npz_dataset(npz_path:str):
    '''
    Load a completed dataset in npz format.
        Args:
            npz_path (str) : Path to the completed npz dataset.
        Returns:
            - If there is a validation dataset : class_names, x_train, x_valid, x_test, y_train, y_valid, y_test
            - else : class_names, x_train, x_test, y_train, y_test
    '''
    valid = False
    with np.load(npz_path) as file:
        class_names=file['class_names']
        x_train=file['x_train']
        x_test=file['x_test']
        y_train=file['y_train']
        y_test=file['y_test']
        if 'x_valid' in list(file.keys()):
            x_valid=file['x_valid']
            y_valid=file['y_valid']
            valid = True
    if valid: return class_names, x_train, x_valid, x_test, y_train, y_valid, y_test
    return class_names, x_train, x_test, y_train, y_test
''' Load npz dataset ====== End ====== '''