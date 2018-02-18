"""Input and output helpers to load in data.
"""
import numpy as np


def read_dataset(path_to_dataset_folder, index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str):
            path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1], 
                                                     [1, x2], 
                                                     [1, x3],
                                                     .......] 
                          where xi is the 16-dimensional feature of each sample
            
        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...] 
                             where yi is +1/-1, the label of each sample 
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    a_content = []
    t_content = []
    n_lines = 0
    with open(path_to_dataset_folder + '/' + index_filename, 'r') as file:
        for line in file:
            n_lines += 1
            label, sample_filename = line.replace('\n', '').split()
            t_content.append(int(label))
            with open(path_to_dataset_folder + '/' + sample_filename) as sample:
                raw_features = sample.read()\
                    .replace('\n', '')\
                    .split()
                if not len(raw_features):
                    continue
                a_content.append([1] + [np.float64(a) for a in raw_features])

    A = np.array(a_content)
    T = np.array(t_content)
    assert(T.shape == (n_lines,))
    return A, T