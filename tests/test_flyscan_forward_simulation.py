import os
import argparse

import tifffile
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

import fast.core.measurement_interface
from fast.input_params import FlyScanSampleParams
from fast.utils.generate_scan_pattern import FlyScanPathGenerator


def run_simulation(probe=None):
    image = tifffile.imread(os.path.join('data', 'xrf_2idd_Cs_L.tiff'))
    sample_params = FlyScanSampleParams(
        image=image,
        psize_nm=1,
        scan_speed_nm_sec=1,
        exposure_sec=0.9,
        deadtime_sec=0.1,
        step_size_for_integration_nm=0.1,
        probe=probe
    )
    measurement_interface = fast.core.measurement_interface.FlyScanXRFSimulationMeasurementInterface(
        image=image, sample_params=sample_params
    )
    path_gen = FlyScanPathGenerator(image.shape)
    scan_path = path_gen.generate_raster_scan_path([1, 1], [133, 131], 1)
    measured_values = measurement_interface.perform_measurement(scan_path, 'pixel', path_gen.dead_segment_mask)
    measured_positions = measurement_interface.measured_positions
    return measured_values, measured_positions


def test_flyscan_forward_simulation_provided_probe(generate_gold=False, return_results=False):
    measured_values, measured_positions = run_simulation(probe=np.ones([4, 4]))
    if generate_gold:
        np.save(os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_values_provided_probe.npy'),
                measured_values)
        np.save(os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_positions_provided_probe.npy'),
                measured_positions)
    else:
        measured_values_gold = np.load(
            os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_values_provided_probe.npy'))
        measured_positions_gold = np.load(
            os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_positions_provided_probe.npy'))
        assert np.allclose(measured_values, measured_values_gold)
        assert np.allclose(measured_positions, measured_positions_gold)
    if return_results:
        return measured_values, measured_positions
    return None


def test_flyscan_forward_simulation_delta_probe(generate_gold=False, return_results=False):
    measured_values, measured_positions = run_simulation(probe=None)
    if generate_gold:
        np.save(os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_values_delta_probe.npy'),
                measured_values)
        np.save(os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_positions_delta_probe.npy'),
                measured_positions)
    else:
        measured_values_gold = np.load(
            os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_values_delta_probe.npy'))
        measured_positions_gold = np.load(
            os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_positions_delta_probe.npy'))
        assert np.allclose(measured_values, measured_values_gold)
        assert np.allclose(measured_positions, measured_positions_gold)
    if return_results:
        return measured_values, measured_positions
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    def show_results(measured_positions, measured_values):
        image = tifffile.imread(os.path.join('data', 'xrf_2idd_Cs_L.tiff'))
        grid_y, grid_x = np.mgrid[:image.shape[0], :image.shape[1]]
        recon = griddata(measured_positions, measured_values, (grid_y, grid_x))

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title('Orignal image')
        axes[1].imshow(recon)
        axes[1].set_title('Sampled')
        plt.show()

    measured_values, measured_positions = test_flyscan_forward_simulation_delta_probe(generate_gold=args.generate_gold,
                                                                                      return_results=True)
    if not args.generate_gold:
        show_results(measured_positions, measured_values)

    measured_values, measured_positions = test_flyscan_forward_simulation_provided_probe(generate_gold=args.generate_gold,
                                                                                         return_results=True)
    if not args.generate_gold:
        show_results(measured_positions, measured_values)
