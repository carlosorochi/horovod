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
import pytest
import unittest
import warnings

from horovod.run.common.util import config_parser
from horovod.run.run import parse_args, _run_elastic

from common import override_args, override_env, temppath


DISCOVERY_SCRIPT_TEMPLATE = """#!/bin/bash
epoch=0
if [ -f {logfile} ]; then
    epoch=$(< {logfile} wc -l | tr -d '[:space:]')
fi

if [ "$epoch" == "0" ]; then
    echo "localhost:2"
elif [ "$epoch" == "1" ]; then
    echo "localhost:4"
else
    echo "localhost:2"
fi
"""


@contextlib.contextmanager
def temp_discovery_script(logfile):
    with temppath() as discovery_script:
        with open(discovery_script, 'w') as f:
            f.write(DISCOVERY_SCRIPT_TEMPLATE.format(logfile=logfile))
        os.chmod(discovery_script, 0o755)
        yield discovery_script


class ElasticTorchTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ElasticTorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_torch(self):
        training_script = os.path.join(os.path.dirname(__file__), 'data/elastic_torch_main.py')
        with temppath() as logfile:
            with temp_discovery_script(logfile) as discovery_script:
                with override_args('horovodrun',
                                   '-np', '2',
                                   '--min-np', '2',
                                   '--max-np', '4',
                                   '--host-discovery-script', discovery_script,
                                   training_script,
                                   '--epoch-to-exit', '10',
                                   '--logfile', logfile):
                    args = parse_args()
                    env = {}
                    config_parser.set_env_from_args(env, args)

                    _run_elastic(args)

                    with open(logfile, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        state_dict = json.loads(line)
                        print(state_dict)
