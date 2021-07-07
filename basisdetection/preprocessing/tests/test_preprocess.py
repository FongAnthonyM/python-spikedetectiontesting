#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_preprocess.py
Description: 
"""
__author__ = "Anthony Fong"
__copyright__ = "Copyright 2021, Anthony Fong"
__credits__ = ["Anthony Fong"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Anthony Fong"
__email__ = ""
__status__ = "Prototype"

# Default Libraries #
import datetime
import pathlib

# Downloaded Libraries #
import matplotlib.pyplot as plt
import pytest

# Local Libraries #
from basisdetection.preprocessing.preprocessing import *


# Definitions #
# Functions #
@pytest.fixture
def tmp_dir(tmpdir):
    """A pytest fixture that turn the tmpdir into a Path object."""
    return pathlib.Path(tmpdir)


# Classes #
class ClassTest:
    """Default class tests that all classes should pass."""
    class_ = None
    timeit_runs = 100
    speed_tolerance = 200

    def get_log_lines(self, tmp_dir, logger_name):
        path = tmp_dir.joinpath(f"{logger_name}.log")
        with path.open() as f_object:
            lines = f_object.readlines()
        return lines


class TestPreprocessing(ClassTest):
    def test_preprocessing(self):
        first = datetime.datetime(2020, 9, 22, 0, 00, 00)
        second = datetime.datetime(2020, 9, 22, 1, 00, 00)

        spath = pathlib.Path("/userdata/akhambhati/Hoth/Remotes/CORE.EMU_SpikeDetection")
        opath = pathlib.Path("/userdata/akhambhati/Hoth/Remotes/RSRCH.EMU_SpikeDetection")
        sdf = StudyDataPreprocessorTest("EC228", 512, 1024, spath=spath, opath=opath)

        sdf.process_data_range_save(name="EC228 00-01", s=first, e=second,
                                    indices=[slice(None, None), slice(None, 184)])

        assert 1


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])
