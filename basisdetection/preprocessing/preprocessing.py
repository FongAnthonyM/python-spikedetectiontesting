#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" preprocessing.py
Description: 
"""
__author__ = "Anthony Fong"
__copyright__ = "Copyright 2021, Anthony Fong"
__credits__ = ["Anthony Fong", "Ankit Khambhati"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Anthony Fong"
__email__ = ""
__status__ = "Prototype"

# Default Libraries #

# Downloaded Libraries #
import numpy as np
import scipy

# Local Libraries #
from ..dataformatting import *
from subrepos.hdf5objects.src.hdf5objects.xltek.hdf5xltekstudy import HDF5XLTEKstudy, HDF5XLTEK
from subrepos import zappy as zp


# Definitions #
# Classes #
class PreprocessingTest(DataFormatter):
    def evaluate(self, data=None, indices=None, copy_=True):
        d, true_fs = super().evaluate(data, indices, copy_)
        # Todo: Fill this in, Ankit

        #
        d = zp.sigproc.filters.notch_line(d, true_fs, notch_freq=60.0, bw=2.0, harm=True)
        d = zp.sigproc.reference.general_reref(d)

        return d, true_fs


class StudyDataPreprocessorTest(StudyDataProcessor):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    # Magic Methods
    # Construction/Destruction
    def __init__(self, subject, new_fs=None, old_fs=None, axis=0, init=True, **kwargs):
        """Constructor for StudyDataFormatter"""
        super().__init__(subject, init=False)

        self.data_formatter = None

        if init:
            self.construct(subject, new_fs, old_fs, axis, **kwargs)

    # Instance Methods
    # Constructors/Destructors
    def construct(self, subject, new_fs=None, old_fs=None, axis=0, **kwargs):
        super().construct(subject, **kwargs)
        self.construct_data_formatter(new_fs, old_fs, axis)

    def construct_data_formatter(self, new_fs=None, old_fs=None, axis=0):
        self.data_formatter = PreprocessingTest(new_fs=new_fs, old_fs=old_fs, axis=axis)
        self.process = self.data_formatter.evaluate

    # Process Data
    def process_data_range(self, s=None, e=None, indices=None, rnd=False, tails=False, frame=False, separate=False):
        d, _, _ = self.study.data_range(s, e, rnd, tails, frame, separate)
        return self.process(d, indices)

    def process_data_range_save(self, name=None, opath=None, s=None, e=None, indices=None,
                                rnd=False, tails=False, frame=False, separate=False):
        if name is None:
            name = self.subject

        if opath is None:
            opath = self.out_path

        data, true_fs = self.process_data_range(s, e, indices, rnd, tails, frame, separate)
        sample_axis = np.arange(0, data.shape[0])
        time_axis = np.linspace(s.timestamp(), e.timestamp(), data.shape[0])

        file = HDF5XLTEK(name, path=opath)
        file._start = s.timestamp()
        file._end = e.timestamp()
        file._sample_rate = true_fs
        file._n_samples = data.shape[0]
        file.build(data=data, saxis=sample_axis, taxis=time_axis)


# Main #
if __name__ == "__main__":
    pass  # This can be used instead of pytest if you prefer.
