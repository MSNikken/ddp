#!/usr/bin/env python
print("SETUP - PY")
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
     packages=['diff_planner'],
     package_dir={'': 'src'}
)

setup(**setup_args)
