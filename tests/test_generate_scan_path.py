from fast.utils.generate_scan_pattern import FlyScanPathGenerator


if __name__ == '__main__':
    path_gen = FlyScanPathGenerator(shape=[50, 50])
    path_gen.generate_raster_scan_path([1, 1], [48, 48], 2)
    path_gen.plot_path()
