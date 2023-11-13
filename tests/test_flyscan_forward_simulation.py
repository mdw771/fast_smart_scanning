import os

import tifffile
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

import fast.core.measurement_interface
from fast.input_params import FlyScanSampleParams
from fast.utils.generate_scan_pattern import FlyScanPathGenerator


def main():
    image = tifffile.imread(Path('data') / 'xrf_2idd_Cs_L.tiff')
    sample_params = FlyScanSampleParams(
        image=image,
        psize_nm=1,
        scan_speed_nm_sec=1,
        exposure_sec=0.9,
        deadtime_sec=0.1,
        num_pts_for_integration_per_measurement=3
    )
    measurement_interface = fast.core.measurement_interface.FlyScanXRFSimulationMeasurementInterface(
        image=image, sample_params=sample_params
    )
    path_gen = FlyScanPathGenerator(image.shape)
    scan_path = path_gen.generate_raster_scan_path([1, 1], [133, 131], 1)
    measured_values = measurement_interface.perform_measurement(scan_path, 'pixel', path_gen.dead_segment_mask)
    measured_positions = measurement_interface.measured_positions
    return measured_values, measured_positions


def test_answer():
    measured_values, measured_positions = main()
    measured_values_gold = np.load(os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_values.npy'))
    measured_positions_gold = np.load(os.path.join('gold', 'test_flyscan_forward_simulation', 'measured_positions.npy'))
    assert np.allclose(measured_values, measured_values_gold)
    assert np.allclose(measured_positions, measured_positions_gold)


if __name__ == '__main__':
    image = tifffile.imread(Path('data') / 'xrf_2idd_Cs_L.tiff')
    measured_values, measured_positions = main()
    grid_y, grid_x = np.mgrid[:image.shape[0], :image.shape[1]]
    recon = griddata(measured_positions, measured_values, (grid_y, grid_x))

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[0].set_title('Orignal image')
    axes[1].imshow(recon)
    axes[1].set_title('Sampled')
    plt.show()
