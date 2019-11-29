import torch
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.functional import F

import os
import sys
from math import ceil

from dtp.datasets.cifar_10 import CIFAR10
import dtp.models.resnet as ResNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 5
MASTER_IP = '10.11.13.41'
MASTER_PORT = '8888'
BATCH_SIZE = 10


def partition_dataset():
    size = dist.get_world_size()
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = CIFAR10(partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)
    return train_set, BATCH_SIZE


def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    print(len(train_set))
    model = ResNet.ResNet18()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01, momentum=0.5
    )
    print('Rank', dist.get_rank())

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    train_set_size = len(train_set)

    losses = []
    mini_batch_loss = []
    for epoch in range(10):
        epoch_loss = 0.0
        index = 0
        current_batch_loss = []
        for data, target in train_set:
            index += 1
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            current_batch_loss.append(epoch_loss / index)
            loss.backward()
            average_gradients(model)
            optimizer.step()
            print('Processed', str(index)+ '/'+ str(train_set_size))
            print('Rank', dist.get_rank(), ', epoch',
                  str(epoch) + ':', epoch_loss / num_batches)
        mini_batch_loss.append(epoch_loss)
        losses.append(epoch_loss)
    print(losses)
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), './model.pt')


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = MASTER_IP
    os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['WORLD_SIZE'] = str(total_size)
    dist.init_process_group(backend, init_method='file:///mnt/sharedfolder/share', rank=rank, world_size=total_size)
    fn(rank, size)


if __name__ == '__main__':
    # Number of cores available on each machine
    size = int(sys.argv[3])

    # Total number of workers to wait for
    total_size = int(sys.argv[2])

    # The current machine worker id
    worker_id = int(sys.argv[1])

    processes = []
    print('World size is', total_size, 'and the worker id is', worker_id)
    for rank in range(size):
        p = Process(target=init_process, args=(rank + size * worker_id, total_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


