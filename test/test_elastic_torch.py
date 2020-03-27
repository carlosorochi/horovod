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
from __future__ import division
from __future__ import print_function

import contextlib
import json
import os
import unittest
import warnings

import mock
import pytest

from horovod.run.common.util import config_parser
from horovod.run.run import parse_args, _run_elastic

from common import override_args, override_env, temppath


DISCOVERY_SCRIPT_TEMPLATE = """#!/bin/bash
epoch=0
if [ -f {logfile} ]; then
    epoch=$(< {logfile} wc -l | tr -d '[:space:]')
fi
"""


def get_discovery_lines(schedule_step, start, end):
    epoch, hosts = schedule_step
    hosts_str = os.linesep.join(['echo "{}"'.format(host) for host in hosts])
    if start and end:
        return hosts_str + os.linesep
    if start:
        return 'if [ "$epoch" == "{}" ]; then'.format(epoch) + os.linesep + hosts_str + os.linesep
    elif not start and not end:
        return 'elif [ "$epoch" == "{}" ]; then'.format(epoch) + os.linesep + hosts_str + os.linesep
    else:
        return 'else' + os.linesep + hosts_str + os.linesep + 'fi' + os.linesep


@contextlib.contextmanager
def temp_discovery_script(logfile, discovery_schedule):
    with temppath() as discovery_script:
        with open(discovery_script, 'w') as f:
            f.write(DISCOVERY_SCRIPT_TEMPLATE.format(logfile=logfile) + os.linesep)
            for i, schedule_step in enumerate(discovery_schedule):
                f.write(get_discovery_lines(schedule_step,
                                            start=i == 0,
                                            end=i == len(discovery_schedule) - 1))
        os.chmod(discovery_script, 0o755)
        yield discovery_script


class ElasticTorchTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ElasticTorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def _run(self, discovery_schedule, epoch_to_exit=2):
        training_script = os.path.join(os.path.dirname(__file__), 'data/elastic_torch_main.py')
        with temppath() as logfile:
            with temp_discovery_script(logfile, discovery_schedule) as discovery_script:
                command_args = ['horovodrun',
                                '-np', '2',
                                '--min-np', '2',
                                '--max-np', '4',
                                '--log-level', 'DEBUG',
                                '--host-discovery-script', discovery_script,
                                'python', training_script,
                                '--epoch-to-exit', str(epoch_to_exit),
                                '--logfile', logfile,
                                '--discovery-schedule', json.dumps(discovery_schedule)]
                print(' '.join(command_args))
                with override_args(*command_args):
                    args = parse_args()
                    env = {}
                    config_parser.set_env_from_args(env, args)
                    _run_elastic(args)

                    with open(logfile, 'r') as f:
                        lines = f.readlines()

                    return [json.loads(line) for line in lines]

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_torch(self):
        discovery_schedule = [
            (0, ['localhost:2']),
            (1, ['localhost:2', '127.0.0.1:2']),
            (None, ['127.0.0.1:2'])
        ]

        results = self._run(discovery_schedule, epoch_to_exit=10)

        assert len(results) == 3

        assert results[0]['size'] == 2
        assert results[0]['hostname'] == 'localhost'

        assert results[1]['size'] == 4
        assert results[1]['hostname'] == 'localhost'

        assert results[2]['size'] == 2
        assert results[2]['hostname'] == '127.0.0.1'
        assert results[2]['rendezvous'] == 3
