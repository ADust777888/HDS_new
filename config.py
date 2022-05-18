import torch


class config:
    lr = 0.001
    n_epoch = 500
    batch_size = 8
    lambd = 0.8
    xi = 0.4
    n_neighbors = 5
    FILE_NAME = 'D:/Study/InnovationProjects/code/data2.csv'
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
