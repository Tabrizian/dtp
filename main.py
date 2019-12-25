import torch
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.functional import F
from torch import nn
from torch.utils.tensorboard import SummaryWriter


import os
import sys
from math import ceil
import time

from dtp.datasets.mnist import MNIST
import dtp.models.resnet as ResNet
import dtp.models.simple as Simple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 5
MASTER_IP = '10.11.13.41'
MASTER_PORT = '8888'
BATCH_SIZE = 10
LOCAL_ITERATIONS = 1

criterion = nn.CrossEntropyLoss()
participating_counts = 2
fake_targets = [1, 2, 3]


def partition_dataset():
    size = dist.get_world_size() - 1
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = MNIST(partition_sizes)
    test_set = partition.test
    fake_set = torch.utils.data.DataLoader(partition.fake,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
    partition = partition.use(dist.get_rank() - 1)
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)
    return train_set, BATCH_SIZE, test_set, fake_set


def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz, test_set, fake_set = partition_dataset()
    print(len(train_set))
    model = Simple.Simple()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01, momentum=0.5
    )
    clients = torch.arange(1, dist.get_world_size())

    print('Rank', dist.get_rank())

    if dist.get_rank() != 0:
        num_batches = ceil(len(train_set.dataset) / float(bsz))
        train_set_size = len(train_set)
        number_of_epochs = 0

        losses = []
        mini_batch_loss = []
        train_set_iterator = iter(train_set)
        fake_set_iterator = iter(fake_set)
        epoch_loss = 0.0
        index = 0
        current_batch_loss = []
        turn = torch.tensor(0)
        participating_clients = torch.zeros(participating_counts, dtype=torch.long)
        mini_batches_to_go = 0
        group = None
        writer = SummaryWriter('mnist-multiplied-100/logs-mnist-fedsgd-3-attacker/')
    number_of_elections = 0
    while True:
        if dist.get_rank() == 0:
            print('============election phase %s started===========' % number_of_elections)
            number_of_elections += 1
            permutation = torch.randperm(dist.get_world_size() - 1)
            clients = clients[permutation]
            participating_clients = clients[0:participating_counts]
            participating_clients_final = participating_clients.tolist()
            print('Participating %s' % participating_clients_final)
            mini_batches_to_go = LOCAL_ITERATIONS
            for index_2, client in enumerate(clients.tolist()):
                if index_2 < participating_counts:
                    dist.send(torch.tensor(1), dst=client)
                    print('Turn sent for client %s' % client)
                else:
                    dist.send(torch.tensor(0), dst=client)
                    print('Turn sent for client %s' % client)
                dist.send(participating_clients, dst=client)
            print('Client %s passed the dist.new_group' % dist.get_rank())
            group = dist.new_group(participating_clients_final)
        else:
            if mini_batches_to_go == 0:
                print('============election phase %s started===========' % number_of_elections)
                number_of_elections += 1
                if number_of_elections == 450 * 5 / LOCAL_ITERATIONS:
                    number_of_elections = 0
                    print("////////// Epoch Changed! %s //////////////" % number_of_epochs)
                    number_of_epochs += 1
                    if number_of_epochs == EPOCHS:
                        break
                    train_set_iterator = iter(train_set)
                    fake_set_iterator = iter(fake_set)
                    epoch_loss = 0.0

                print('Waiting for the current turn...')
                dist.recv(turn, src=0)
                print('Turn is %s' % turn)
                print('Waiting for the winners of the election...')
                dist.recv(participating_clients, src=0)
                participating_clients_final = participating_clients.tolist()
                print('Participating %s' % participating_clients_final)
                print('Client %s passed the dist.new_group' % dist.get_rank())
                # This needs to be fixed!! Race Condition!! Distributed Systems Bug!
                time.sleep(0.03)
                group = dist.new_group(participating_clients_final)
                if turn == 1:
                    mini_batches_to_go = LOCAL_ITERATIONS
        # Only those with the pass can enter the game!
        if dist.get_rank() != 0 and turn == 1:
            if dist.get_rank() not in fake_targets:
                data, target = next(train_set_iterator)
            else:
                data, target = next(fake_set_iterator)
            mini_batches_to_go -= 1
            index += 1
            if index % 250 == 0 and dist.get_rank() == 1:
                model_accuracy = accuracy(model, test_set)
                writer.add_scalar('%s/Validation/Accuracy' % dist.get_rank(), model_accuracy, index)
                writer.flush()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            writer.add_scalar('%s/Train/Loss' % dist.get_rank(), loss.item(), index)
            writer.flush()
            epoch_loss += loss.item()
            current_batch_loss.append(epoch_loss / index)
            loss.backward()
            print('Processed', str(index) + '/' + str(train_set_size))
            if mini_batches_to_go == 0:
                print('Aggregating...')
                participating_clients_final = participating_clients.tolist()
                print('Participating %s' % participating_clients_final)
                average_gradients(model, group=group)
            optimizer.step()
    #mini_batch_loss.append(epoch_loss)
    #losses.append(epoch_loss)
    #print(losses)
    if dist.get_rank() == 1:
        accuracy(model, test_set)
        torch.save(model.state_dict(), './model-2.pt')


def average_gradients(model, group):
    print('Arrived in the average_gradients')
    print('Rank arrived in averaging: %s' % dist.get_rank())
    size = dist.get_world_size(group=group)
    for param in model.parameters():
        if dist.get_rank() in fake_targets:
            dist.all_reduce(param.grad.data * 100, op=dist.ReduceOp.SUM, group=group)
        else:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
            param.grad.data /= size


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = MASTER_IP
    os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['WORLD_SIZE'] = str(total_size)
    dist.init_process_group(backend, init_method='file:///mnt/sharedfolder/share', rank=rank, world_size=total_size)
    fn(rank, size)


def accuracy(model, test_set):
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_set):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('====================Accuracy:', 100.0 * correct / total, '=========================================')
    return 100.0 * correct / total


if __name__ == '__main__':
    # Number of cores available on each machine
    # size = int(sys.argv[3])

    # Total number of workers to wait for
    total_size = int(sys.argv[2])

    # The current machine worker id
    worker_id = int(sys.argv[1])

    init_process(worker_id, total_size, run)


