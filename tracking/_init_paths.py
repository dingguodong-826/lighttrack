from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

project_path = osp.join(this_dir, '..')
add_path(project_path)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

# supernet_path = osp.join(this_dir, '..', 'supernet_backbone')
# add_path(supernet_path)
# sys.path.insert(1, osp.join(this_dir, '..', 'lib/eval_toolkit/bin'))
