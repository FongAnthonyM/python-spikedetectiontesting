#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" initialformatting.py
Description: Classes and Functions that format the initial data before being cleaned.
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
from fractions import Fraction

# Downloaded Libraries #
from baseobjects import BaseObject
import numpy as np
from scipy import interpolate
from scipy.signal import sosfiltfilt, butter, buttord

# Local Libraries #
from subrepos.hdf5objects.src.hdf5objects.xltek.hdf5xltekstudy import HDF5XLTEKstudy, HDF5XLTEK
from subrepos import hdf5objects


# Definitions #
# Functions #
def construct_low_pass_filter(fs, corner_freq, stop_tol=10):
    # Compute stop-band frequency
    corner_freq = np.float(corner_freq)
    stop_freq = (1 + stop_tol / 100) * corner_freq

    # Get butterworth filter parameters
    buttord_params = {'wp': corner_freq,  # Passband
                      'ws': stop_freq,  # Stopband
                      'gpass': 3,  # 3dB corner at pass band
                      'gstop': 60,  # 60dB min. attenuation at stop band
                      'analog': False,  # Digital filter
                      'fs': fs}
    ford, wn = buttord(**buttord_params)

    # Design the filter using second-order sections to ensure better stability
    return butter(ford, wn, btype='lowpass', output='sos', fs=fs)


def construct_high_pass_filter(fs, corner_freq, stop_tol=10):
    # Compute stop-band frequency
    corner_freq = np.float(corner_freq)
    stop_freq = (1 + stop_tol / 100) * corner_freq

    # Get butterworth filter parameters
    buttord_params = {'wp': corner_freq,  # Passband
                      'ws': stop_freq,  # Stopband
                      'gpass': 3,  # 3dB corner at pass band
                      'gstop': 60,  # 60dB min. attenuation at stop band
                      'analog': False,  # Digital filter
                      'fs': fs}
    ford, wn = buttord(**buttord_params)

    # Design the filter using second-order sections to ensure better stability
    return butter(ford, wn, btype='high', output='sos', fs=fs)


def remove_dc_drift(data=None, fs=None, corner_freq=0.5, axis=0, copy_=True):
    if copy_:
        data = data.copy()
    sos = construct_high_pass_filter(fs, corner_freq)
    return sosfiltfilt(sos, data, axis=axis)


def remove_dc_offset(data=None, axis=0, copy_=True):
    if copy_:
        data = data.copy()
    return data - data.mean(axis=axis)


# Classes #
class Resample(BaseObject):
    def __init__(self, data=None, new_fs=None, old_fs=None, axis=0, interp_type="linear", aa_filters=None, init=True):
        self.n_limit = 100
        self.aa_corner = 250

        self.new_fs = None
        self.old_fs = None
        self.high_fs = None
        self.true_fs = None
        self.true_nyq = None
        self.p = None
        self.q = None
        self.axis = 0

        self.data = None
        self.interpolator = None
        self.aa_filters = None

        if init:
            self.construct(data, new_fs, old_fs, axis, interp_type, aa_filters)

    # Constructors/Destructors
    def construct(self, data=None, new_fs=None, old_fs=None, axis=0, interp_type="linear", aa_filters=None):
        # Set Attributes
        self.new_fs = new_fs
        self.old_fs = old_fs
        self.axis = axis

        self.data = data

        # Construct Objects
        if data is not None:
            self.construct_interpolator(interp_type=interp_type)

        if new_fs is not None and old_fs is not None:
            self.rationalize_fs()
            if aa_filters is None:
                self.construct_aa_filters()

        if aa_filters is not None:
            try:
                _ = iter(aa_filters)
                self.aa_filters = aa_filters
            except TypeError:
                self.aa_filters = [aa_filters]

    # Setup
    def rationalize_fs(self, new_fs=None, old_fs=None):
        if new_fs is not None:
            self.new_fs = new_fs

        if old_fs is not None:
            self.old_fs = old_fs

        # Normally, new / old but to limit numerator the limit denominator must be used;
        # it is correct by exchanging the assignment of the p & q.
        f = Fraction(self.old_fs / self.new_fs).limit_denominator(self.n_limit)
        self.p = f.denominator
        self.q = f.numerator

        self.high_fs = self.old_fs * self.p
        self.true_fs = self.high_fs / self.q
        self.true_nyq = self.true_fs // 2

    def construct_interpolator(self, data=None, interp_type="linear", axis=0, copy_=True, bounds_error=None,
                               fill_value=np.nan, assume_sorted=False):
        if data is not None:
            self.data = data

        if axis is not None:
            self.axis = axis

        samples = self.data.shape[0]
        x = np.arange(0, samples)
        y = self.data
        self.interpolator = interpolate.interp1d(x, y, interp_type, axis, copy_, bounds_error, fill_value, assume_sorted)

    def construct_aa_filters(self, new_fs=None, old_fs=None, aa_corner=None):
        if new_fs is not None or old_fs is not None or self.true_nyq is None:
            self.rationalize_fs(new_fs, old_fs)

        if aa_corner is not None:
            self.aa_corner = aa_corner

        self.aa_filters = []

        # Anti-alias filtering with iterative method
        # Find closest power of 2
        pow2 = int(np.log2(self.true_nyq / self.aa_corner))
        if pow2 > 0:
            corner_freq = self.true_nyq
            for ii in range(pow2):
                corner_freq /= 2
                self.aa_filters.append(construct_low_pass_filter(self.high_fs, corner_freq))
        self.aa_filters.append(construct_low_pass_filter(self.high_fs, self.aa_corner))

    # Calculations
    def interpolate(self, data=None):
        samples = self.data.shape[0]

        if self.interpolator is None or data is not None:
            self.construct_interpolator(data)

        return self.interpolator(np.linspace(0, samples-1, self.p*samples))

    def filter(self, data, copy_=True):
        if copy_:
            data = data.copy()

        for aa_filter in self.aa_filters:
            data = sosfiltfilt(aa_filter, data, axis=self.axis)

        return data

    def downsample(self, data, indices=None, axis=None):
        if axis is None:
            axis = self.axis

        slices = [slice(None, None)] * len(data.shape)
        if indices is not None:
            for ax, index in enumerate(indices):
                slices[ax] = index
        slices[axis] = slice(None, None, self.q)

        data = data[tuple(slices)]
        return data

    def evaluate(self, data=None, new_fs=None, old_fs=None, indices=None, copy_=True):
        if data is not None:
            self.data = data

        # Construct filters if fs changes
        if new_fs is not None or old_fs is not None:
            if new_fs is not None:
                self.new_fs = new_fs
            if old_fs is not None:
                self.old_fs = old_fs
            self.rationalize_fs()
            self.construct_aa_filters()

        # Interpolate if needed and filter
        if self.p == 1:
            data = self.filter(self.data, copy_)
        else:
            if data is not None:
                self.construct_interpolator()
            data = self.interpolate()
            data = self.filter(data, copy_=False)

        # Downsample
        data = self.downsample(data, indices)

        return data, self.true_fs


class DataFormatter(BaseObject):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    # Construction/Destruction
    def __init__(self, data=None, new_fs=None, old_fs=None, axis=0, init=True):
        """Constructor for FormatData"""
        self.new_fs = None
        self.old_fs = None
        self.axis = 0

        self.data = None

        self.resample = None

        if init:
            self.construct(data, new_fs, old_fs, axis)

    # Constructors/Destructors
    def construct(self, data=None, new_fs=None, old_fs=None, axis=0):
        self.new_fs = new_fs
        self.old_fs = old_fs
        self.axis = axis

        self.data = data

        self.resample = Resample(data, new_fs, old_fs, axis)

    def evaluate(self, data=None, indices=None, copy_=True):
        if data is None:
            data = self.data
        else:
            self.data = data

        if indices is not None:
            data = data[tuple(indices)]
        elif copy_:
            data = data.copy()

        data, true_fs = self.resample.evaluate(data, copy_=False)
        data = remove_dc_drift(data, true_fs, axis=self.axis, copy_=copy_)
        data = remove_dc_offset(data, self.axis, copy_)
        return data, true_fs


# Todo: Make a DataStudy package?
class StudyDataProcessor(BaseObject):
    study_class = HDF5XLTEKstudy
    study_file = HDF5XLTEK

    def __init__(self, subject, path=None, spath=None, opath=None, start=None, end=None, init=True):
        self.subject = None
        self.start = None
        self.end = None

        self.study = None
        self.process = None

        self.out_path = None

        if init:
            self.construct(subject, path, spath, opath, start, end)

    def construct(self, subject, path=None, spath=None, opath=None, start=None, end=None):
        self.subject = subject
        self.start = start
        self.end = end
        self.out_path = opath

        self.construct_study(path=path, spath=spath)

    def construct_study(self, subject=None, path=None, spath=None):
        if subject is None:
            subject = self.subject
        else:
            self.subject = subject

        if spath is None:
            self.study = self.study_class(subject, path)
        else:
            self.study = self.study_class(subject, path, spath)

    def process_data_range(self, s=None, e=None, rnd=False, tails=False, frame=False, separate=False):
        d, _, _ = self.study.data_range(s, e, rnd, tails, frame, separate)
        return self.process(d)

    def process_data_range_save(self, name=None, opath=None, s=None, e=None,
                                rnd=False, tails=False, frame=False, separate=False):
        if name is None:
            name = self.subject

        if opath is None:
            opath = self.out_path

        data = self.process_data_range(s, e, rnd, tails, frame, separate)
        sample_axis = np.arange(0, data.shape[0])

        file = HDF5XLTEK(name, path=opath)
        file.build(data=data, saxis=sample_axis)


class StudyDataFormatter(StudyDataProcessor):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    # Construction/Destruction
    def __init__(self, subject, new_fs=None, old_fs=None, axis=0, init=True, **kwargs):
        """Constructor for StudyDataFormatter"""
        super().__init__(subject, init=False)

        self.data_formatter = None

        if init:
            self.construct(subject, new_fs, old_fs, axis, **kwargs)

    # Constructors/Destructors
    def construct(self, subject, new_fs=None, old_fs=None, axis=0, **kwargs):
        super().construct(subject, **kwargs)
        self.construct_data_formatter(new_fs, old_fs, axis)

    def construct_data_formatter(self, new_fs=None, old_fs=None, axis=0):
        self.data_formatter = DataFormatter(new_fs=new_fs, old_fs=old_fs, axis=axis)
        self.process = self.data_formatter.evaluate

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
