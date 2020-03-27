# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import horovod.torch as hvd

parser = argparse.ArgumentParser(description='PyTorch Elastic Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batches-per-epoch', type=int, default=10,
                    help='number of batches per epoch')
parser.add_argument('--batches-per-commit', type=int, default=1,
                    help='number of batches per commit of the elastic state object')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of epochs')
parser.add_argument('--epoch-to-exit', type=int, default=2,
                    help='epoch at the start of which to exit on rank 0')
parser.add_argument('--logfile', default='/tmp/logfile.txt',
                    help='log file to record results (one line per epoch)')

args = parser.parse_args()

hvd.init()

batch_size = 32
data = torch.randn(batch_size, 2)
target = torch.LongTensor(batch_size).random_() % 2

model = torch.nn.Sequential(torch.nn.Linear(2, 2))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001 * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

start_rank = int(os.environ.get('HOROVOD_RANK', 0))


def check_exit(epoch):
    if epoch == args.epoch_to_exit and start_rank == 0:
        raise RuntimeError('check_rank and exit')


def log_state(state):
    state_dict = {
        'epoch': state.epoch,
        'batch': state.batch,
        'commits': state.commits,
        'rank': start_rank,
        'size': hvd.size(),
        'rendezvous': state.rendezvous}
    with open(args.logfile, 'a') as f:
        f.write(json.dumps(state_dict) + os.linesep)


@hvd.elastic.run
def train(state):
    state.rendezvous += 1
    for state.epoch in range(state.epoch, args.epochs):
        check_exit(state.epoch)

        for state.batch in range(state.batch, args.batches_per_epoch):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if (state.batch + 1) % args.batches_per_commit == 0:
                # print('commit empty? {}'.format(state._host_messages.empty()))
                state.commits += 1
                state.commit()

        state.batch = 0
        state.commits += 1
        state.commit()

        if hvd.rank() == 0:
            log_state(state)

        if state.epoch < 2:
            start = int(time.time())
            while state._host_messages.empty():
                if int(time.time()) - start > 3:
                    raise TimeoutError('Timed out waiting for notifications from driver.')
                print('sleep')
                time.sleep(0.1)
        print('empty? {}'.format(state._host_messages.empty()))


def on_state_reset():
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * hvd.size()


state = hvd.elastic.TorchState(model, optimizer, batch=0, epoch=0, commits=0, rendezvous=0)
state.register_reset_callbacks([on_state_reset])
train(state)
