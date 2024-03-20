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
    """
    Fly scan simulator. The simulator's perform measurement method takes in a continuous scan path defined
    by a list of vertices; the scan path is formed by linearly connecting the vertices sequentially. The simulator
    automatically split the scan path into exposures based on the setting of exposure time in `sample_params`.
    Between exposures, there can be dead times where no data is acquired based on the setting of dead time.
    For each exposure, several points are sampled along the path, whose interval is determined by the setting
    of `step_size_for_integration_nm` in `sample_params`. The intensities sampled at all sampling points are
    averaged as the measurement for that exposure. The positions of all sampling points are averaged as the
    position for that exposure.
    """
    def __init__(self, image: np.ndarray = None, image_path: str = None, sample_params: FlyScanSampleParams = None,
                 eps=1e-5):
        """
        The constructor.

        :param image: np.ndarray. The sample image.
        :param image_path: str. Path to the image.
        :param sample_params: FlyScanSampleParams.
        :param eps: float.
        """
        super().__init__(image, image_path)
        assert sample_params
        self.sample_params = sample_params
        self.measured_values = None
        self.measured_positions = None
        self.eps = eps
        self.points_to_sample_all_exposures = []

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
        :return list[float]: measured values at all exposures. The positions of the exposures can be retrieved from the
                `measured_positions` attribute.
        """
        vertex_list = np.asarray(vertex_list)
        if vertex_unit == 'nm':
            vertex_list = vertex_list / self.sample_params.psize_nm

        self.build_sampling_points(vertex_list, dead_segment_mask)

        meas_value_list = []
        meas_pos_list = []
        for i_exposure in range(len(self.points_to_sample_all_exposures)):
            pts_to_sample = self.points_to_sample_all_exposures[i_exposure]
            if len(pts_to_sample) == 0:
                continue
            sampled_vals = self.get_interpolated_values_from_image(pts_to_sample)
            meas_value_list.append(np.mean(sampled_vals))
            meas_pos_list.append(np.mean(pts_to_sample, axis=0))
        self.measured_values = np.array(meas_value_list)
        self.measured_positions = np.stack(meas_pos_list, axis=0)
        return self.measured_values

    def build_sampling_points(self, vertex_list, dead_segment_mask=None):
        points_to_sample_all_exposures = []
        i_input_segment = 0
        length_covered_in_current_segment = 0
        pt_coords = vertex_list[0]
        while i_input_segment < len(vertex_list) - 1:
            if dead_segment_mask is not None and dead_segment_mask[i_input_segment] == False:
                i_input_segment += 1
                continue
            length_exposed = 0
            length_dead = 0
            length_sampling = 0
            # Add live segment.
            points_to_sample_current_exposure = [pt_coords]
            while length_exposed < self.sample_params.exposure_length_pixel - self.eps:
                if i_input_segment + 1 >= len(vertex_list):
                    break
                current_direction = vertex_list[i_input_segment + 1] - vertex_list[i_input_segment]
                current_seg_length = np.linalg.norm(current_direction)
                current_direction = current_direction / current_seg_length
                if length_covered_in_current_segment + self.sample_params.step_size_for_integration_pixel - length_sampling <= current_seg_length:
                    pt_coords = pt_coords + current_direction * (self.sample_params.step_size_for_integration_pixel - length_sampling)
                    points_to_sample_current_exposure.append(pt_coords)
                    length_covered_in_current_segment += (self.sample_params.step_size_for_integration_pixel - length_sampling)
                    length_exposed += (self.sample_params.step_size_for_integration_pixel - length_sampling)
                    length_sampling = 0
                else:
                    if i_input_segment + 1 >= len(vertex_list):
                        break
                    pt_coords = pt_coords + current_direction * (current_seg_length - length_covered_in_current_segment)
                    i_input_segment += 1
                    length_exposed += (current_seg_length - length_covered_in_current_segment)
                    length_sampling += (current_seg_length - length_covered_in_current_segment)
                    length_covered_in_current_segment = 0
            # Update variables for dead segment.
            while length_dead < self.sample_params.dead_length_pixel - self.eps:
                if i_input_segment + 1 >= len(vertex_list):
                    break
                current_direction = vertex_list[i_input_segment + 1] - vertex_list[i_input_segment]
                current_seg_length = np.linalg.norm(current_direction)
                current_direction = current_direction / current_seg_length
                if length_covered_in_current_segment + self.sample_params.dead_length_pixel - length_dead <= current_seg_length:
                    pt_coords = pt_coords + current_direction * (self.sample_params.dead_length_pixel - length_dead)
                    length_covered_in_current_segment += (self.sample_params.dead_length_pixel - length_dead)
                    break
                else:
                    if i_input_segment + 1 >= len(vertex_list):
                        break
                    pt_coords = vertex_list[i_input_segment + 1]
                    i_input_segment += 1
                    length_dead += (current_seg_length - length_covered_in_current_segment)
                    length_covered_in_current_segment = 0
            points_to_sample_all_exposures.append(points_to_sample_current_exposure)
        self.points_to_sample_all_exposures = points_to_sample_all_exposures

    def plot_sampled_points(self):
        pts = np.concatenate(self.points_to_sample_all_exposures, axis=0)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.scatter(pts[:, 1], pts[:, 0])
        plt.show()

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
            sampled_vals = []
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
                val = np.sum(vals * probe.reshape(-1))
                sampled_vals.append(val)
            return sampled_vals


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
