#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_initialformatting.py
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
from basisdetection.dataformatting.initialformatting import *
from subrepos.zappy.zappy.pipelines.general_preproc import ieeg_screening_pipeline


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


class TestResample(ClassTest):

    def test_interpolation(self):
        samples = 1000
        x_axis = np.arange(0, samples)
        data = np.cos(x_axis ** 2.0 / 8.0) + 1
        fs_old = 512
        fs_new = 1024

        rs = Resample(data, fs_new, fs_old)
        new_data = rs.interpolate()

        assert new_data.shape[0] == samples*2

    def test_evaluate(self):
        samples = 1000
        x_axis = np.linspace(0, 100, samples)
        x_axis_ds = np.linspace(0, 100, samples//2)
        data = np.cos(x_axis ** 2.0 / 8.0) + 1
        fs_old = 1024
        fs_new = 512

        rs = Resample(data, fs_new, fs_old)
        new_data, fs = rs.evaluate()

        control, _ = ieeg_screening_pipeline(data, fs_old, fs_new)

        plt.plot(x_axis, data, ':', label="original data")
        plt.plot(x_axis_ds, new_data, '-', label="new data")
        plt.plot(x_axis_ds, control, '.', label="control data")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim([-0.1, 3])
        plt.legend(loc="best")

        assert new_data.shape[0] == samples/2


class TestStudyDataFormatter(ClassTest):

    def test_process_and_save(self):
        first = datetime.datetime(2020, 1, 28, 12, 00, 00)
        second = datetime.datetime(2020, 1, 28, 12, 45, 00)

        spath = pathlib.Path("/common/subjects/")
        opath = pathlib.Path("/home/anthonyfong/Documents/Projects/Epilepsy Spike Detection")
        sdf = StudyDataFormatter("EC212", 512, 1024, spath=spath, opath=opath)

        sdf.process_data_range_save(name="EC212 2020-01-28 12-13", s=first, e=second, indices=[slice(None, None), slice(None, 184)])

        assert 1

    def test_EC228_loop(self):
        spath = pathlib.Path("/common/subjects/")
        opath = pathlib.Path("/home/anthonyfong/Documents/Projects/Epilepsy Spike Detection")

        for h in range(0, 24):
            first = datetime.datetime(2020, 9, 22, h, 00, 00)
            if h+1 == 24:
                second = datetime.datetime(2020, 9, 23, 00, 00, 00)
            else:
                second = datetime.datetime(2020, 9, 22, h + 1, 00, 00)

            sdf = StudyDataFormatter("EC228", 512, 1024, spath=spath, opath=opath)

            sdf.process_data_range_save(name=f"EC228 2020-09-23 {h}-{h+1}", s=first, e=second,
                                        indices=[slice(None, None), slice(None, 1)])

    def test_EC212_loop(self):
        spath = pathlib.Path("/common/subjects/")
        opath = pathlib.Path("/home/anthonyfong/Documents/Projects/Epilepsy Spike Detection")

        for h in range(0, 24):
            first = datetime.datetime(2020, 1, 28, h, 00, 00)
            if h+1 == 24:
                second = datetime.datetime(2020, 1, 29, 00, 00, 00)
            else:
                second = datetime.datetime(2020, 1, 28, h + 1, 00, 00)

            sdf = StudyDataFormatter("EC212", 512, 1024, spath=spath, opath=opath)

            sdf.process_data_range_save(name=f"EC212 2020-01-28 {h}-{h+1}", s=first, e=second,
                                        indices=[slice(None, None), slice(None, 159)])

    def test_EC133_loop(self):
        spath = pathlib.Path("/common/subjects/")
        opath = pathlib.Path("/mnt/changserver/userdata/afong/EpilepsySpikeDetection/EC133")

        for h in range(0, 14):
            first = datetime.datetime(2016, 9, 14, h, 00, 00)
            if h+1 == 24:
                second = datetime.datetime(2016, 9, 15, 00, 00, 00)
            else:
                second = datetime.datetime(2016, 9, 14, h + 1, 00, 00)

            sdf = StudyDataFormatter("EC133", 512, 1024, spath=spath, opath=opath)

            sdf.process_data_range_save(name=f"EC133 2016-09-14 {h}-{h+1}", s=first, e=second,
                                        indices=[slice(None, None), slice(None, 127)])


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])
