# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Software Name:    Fast Autonomous Scanning Toolkit (FAST)               #
# By: Argonne National Laboratory                                         #
# OPEN SOURCE LICENSE                                                     #
#                                                                         #
# Redistribution and use in source and binary forms, with or without      #
# modification, are permitted provided that the following conditions      #
# are met:                                                                #
#                                                                         #
# 1. Redistributions of source code must retain the above copyright       #
#    notice, this list of conditions and the following disclaimer.        #
#                                                                         #
# 2. Redistributions in binary form must reproduce the above copyright    #
#    notice, this list of conditions and the following disclaimer in      #
#    the documentation and/or other materials provided with the           #
#    distribution.                                                        #
#                                                                         #
# 3. Neither the name of the copyright holder nor the names of its        #
#    contributors may be used to endorse or promote products derived from #
#    this software without specific prior written permission.             #
#                                                                         #
# *********************************************************************** #
#                                                                         #
# DISCLAIMER                                                              #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE          #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,    #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS   #
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED      #
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,  #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF   #
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY            #
# OF SUCH DAMAGE.                                                         #
# *********************************************************************** #
import abc

import numpy as np
import tifffile as tif
import scipy.ndimage as ndi

from ..input_params import FlyScanSampleParams

class MeasurementInterface(abc.ABC):
    @abc.abstractmethod
    def perform_measurement(self, *args, **kwargs):
        pass


class SimulationMeasurementInterface(MeasurementInterface):
    def __init__(self, image: np.ndarray = None, image_path: str = None, *args, **kwargs):
        assert (image is not None) or (image_path is not None)
        self.image_path = image_path
        if image is not None:
            self.image = image
        else:
            self.image = tif.imread(self.image_path)

class TransmissionSimulationMeasurementInterface(SimulationMeasurementInterface):
    def __init__(self, image: np.ndarray = None, image_path: str = None):
        super().__init__(image, image_path)

    def perform_measurement(self, idxs):
        return self.image[idxs[:, 0], idxs[:, 1]]


class FlyScanXRFSimulationMeasurementInterface(SimulationMeasurementInterface):
    def __init__(self, image: np.ndarray = None, image_path: str = None, sample_params: FlyScanSampleParams = None):
        super().__init__(image, image_path)
        assert sample_params
        self.sample_params = sample_params
        self.measured_values = None
        self.measured_positions = None


    def perform_measurement(self, vertex_list, vertex_unit='pixel', dead_segment_mask=None, *args, **kwargs):
        """
        Perform measurement given a fly scan path defined by a list of vertices.

        :param vertex_list: list[list[float, float]]. A list of vertex positions that define the scan path, ordered
                            in (y, x). The total number of segments is `len(vertex_list) - 1`; for segment i,
                            probe moves from `vertex_list[i]` to `vertex_list[i + 1]`.
        :param vertex_unit: str. Can be 'pixel' or 'nm'.
        :param dead_segment_mask: list[bool]. A list whose length is len(vertex_list) - 1. Marks whether each segment
                                  is a "live" segment or a "dead" segment where only the probe moves but no data is
                                  collected. If None, all segments are assumed to be live.
        :return list[float]: measured values. The positions of the measurements can be retrieved from the
                `measured_positions` attribute.
        """
        if vertex_unit == 'nm':
            vertex_list = vertex_list / self.sample_params.psize_nm
        meas_value_list = []
        meas_pos_list = []
        for i_seg in range(len(vertex_list) - 1):
            seg_begin_pos = vertex_list[i_seg]
            seg_end_pos = vertex_list[i_seg + 1]
            acquire_data = dead_segment_mask[i_seg]
            this_value_list, this_pos_list = self.acquire_data_for_segment(seg_begin_pos, seg_end_pos, acquire_data)
            meas_value_list.append(this_value_list)
            meas_pos_list.append(this_pos_list)
        self.measured_values = np.concatenate(meas_value_list)
        self.measured_positions = np.concatenate([x for x in meas_pos_list if len(x) > 0], axis=0)
        return self.measured_values

    def acquire_data_for_segment(self, seg_begin_pos, seg_end_pos, acquire_data=True):
        """
        Acquire data along a given segment.

        :param seg_begin_pos: list[float].
        :param seg_end_pos: list[float].
        :param acquire_data: bool.
        :return: list[float], list[float, float]. Measured values and their effective positions, taken as the
                 mid-point of each exposure.
        """
        if not acquire_data:
            return [], []

        scan_speed_nm_sec = self.sample_params.scan_speed_nm_sec
        exposure_sec = self.sample_params.exposure_sec
        deadtime_sec = self.sample_params.deadtime_sec
        psize_nm = self.sample_params.psize_nm
        num_points_for_integration = self.sample_params.num_pts_for_integration_per_measurement

        meas_list = []
        meas_pos_list = []
        direction_vector = seg_end_pos - seg_begin_pos
        segment_length = np.linalg.norm(direction_vector)
        direction_vector = direction_vector / segment_length  # Unit vector of travel direction
        exposure_vector = direction_vector * scan_speed_nm_sec * exposure_sec / psize_nm
        exposure_length = np.linalg.norm(exposure_vector)
        deadtime_vector = direction_vector * scan_speed_nm_sec * deadtime_sec / psize_nm
        meas_begin_pos = seg_begin_pos
        meas_end_pos = meas_begin_pos + exposure_vector
        length_covered = 0

        if self.sample_params.step_size_for_integration_nm is not None:
            step_szie_for_integration = self.sample_params.step_size_for_integration_nm / psize_nm
        else:
            assert self.sample_params.num_pts_for_integration_per_measurement is not None
            step_szie_for_integration = np.linalg.norm(meas_end_pos - meas_begin_pos) / (num_points_for_integration - 1)
        point_sampling_vector = direction_vector * step_szie_for_integration

        while length_covered < segment_length:
            if length_covered + exposure_length > segment_length:
                meas_end_pos = seg_end_pos
            points_to_integrate = self.get_integration_point_list(meas_begin_pos, meas_end_pos, point_sampling_vector)
            values = self.get_interpolated_values_from_image(points_to_integrate)
            meas = np.mean(values)
            meas_list.append(meas)
            # Use the midpoint of the segment as the recorded position.
            meas_pos_list.append((meas_begin_pos + meas_end_pos) / 2)
            meas_begin_pos = meas_end_pos + deadtime_vector
            meas_end_pos = meas_begin_pos + exposure_vector
            length_covered += (scan_speed_nm_sec * (exposure_sec + deadtime_sec) / psize_nm)
        return meas_list, meas_pos_list

    def get_integration_point_list(self, meas_begin_pos, meas_end_pos, point_sampling_vector):
        """
        Get a list of points whose values are to be integrated to form the fly scan measurement value.

        :param meas_begin_pos: list[float, float].
        :param meas_end_pos: list[float, float].
        :param point_sampling_vector: list[float, float].
        :return: list[float, float].
        """
        if point_sampling_vector[0] == 0:
            xs = np.arange(meas_begin_pos[1], meas_end_pos[1] + point_sampling_vector[1], point_sampling_vector[1])
            ys = np.array([meas_begin_pos[0]] * len(xs))
        elif point_sampling_vector[1] == 1:
            ys = np.arange(meas_begin_pos[0], meas_end_pos[0] + point_sampling_vector[0], point_sampling_vector[0])
            xs = np.array([meas_begin_pos[1]] * len(ys))
        else:
            xs = np.arange(meas_begin_pos[1], meas_end_pos[1] + point_sampling_vector[1], point_sampling_vector[1])
            ys = np.arange(meas_begin_pos[0], meas_end_pos[0] + point_sampling_vector[0], point_sampling_vector[0])
        assert len(ys) == len(xs)
        return np.stack([ys, xs], axis=1)

    def get_interpolated_values_from_image(self, point_list, normalize_probe=True):
        """
        Obtain interpolated values from the image at given locations.

        :param point_list: list[list[float, float]]. List of point positions.
        :return: list[float].
        """
        if not isinstance(point_list, np.ndarray):
            point_list = np.array(point_list)
        y = point_list[:, 0]
        x = point_list[:, 1]
        if self.sample_params.probe is None:
            # If probe function is not given, assume delta function.
            return ndi.map_coordinates(self.image, [y, x], order=1, mode='nearest')
        else:
            # Prepare a list of coordinates that include n by m region around each sampled point, where (n, m)
            # is the probe shape.
            measurements = []
            probe = self.sample_params.probe
            if normalize_probe:
                probe = probe / np.sum(probe)
            for this_y, this_x in point_list:
                this_y_all = np.linspace(this_y - probe.shape[0] / 2.0, this_y + probe.shape[0] / 2.0, probe.shape[0])
                this_x_all = np.linspace(this_x - probe.shape[1] / 2.0, this_x + probe.shape[1] / 2.0, probe.shape[1])
                xx, yy = np.meshgrid(this_x_all, this_y_all)
                yy = yy.reshape(-1)
                xx = xx.reshape(-1)
                vals = ndi.map_coordinates(self.image, [yy, xx], order=1, mode='nearest')
                measured_val = np.sum(vals * probe.reshape(-1))
                measurements.append(measured_val)
            return measurements


class ExternalMeasurementInterface(MeasurementInterface):
    def __init__(self):
        """This is all currently handled within the experiment script."""
        self.new_values = []
        self._external_measurement_finalized = False

    def finalize_external_measurement(self, values):
        self.new_values = values
        self._external_measurement_finalized = True

    def perform_measurement(self, idxs):
        return self.new_values
