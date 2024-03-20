import numpy as np

from fast.input_params import *


class ScanPath:

    def __init__(self, *args, **kwargs):
        pass


class ArbitraryLinearScanPath(ScanPath):
    """
    A scan path consisted of arbitrary-length linear segments. A segment could contain multiple
    exposure-dead cycles, which will be determined by the set scan speed, exposure time, and
    dead time.
    """
    def __init__(self, vertex_list, unit='nm', psize_nm=None, dead_mask=None, sampling_params: SampleParams = None):
        """
        The constructor.

        :param vertex_list: np.ndarray. A `(n, 2)` array that gives the coordinates of all the inflection
                            points of the scan path.
        :param unit: str. The unit of `vertex_list`. Can be 'nm' or 'pixel'.
        :param psize_nm: float. The pixel size of the vertex list in nm, if its unit is not 'pixel'.
        :param dead_mask: list[bool, ...] | None. A list of booleans that has a length of `len(vertex_list) - 1`. Each
                          element indicates whether a segment is live (where data is acquired as normal) or dead
                          (where data acquisition is disabled). Detector will still have dead times even in a live
                          segment.
        :param sampling_params: SampleParams. The config object that contains parameters like scan speed, dead time,
                                exposure time, etc.
        """
        if unit == 'nm':
            assert psize_nm is not None
            vertex_list = vertex_list / psize_nm
        self.vertex_list = vertex_list
        self.dead_mask = dead_mask
        if sampling_params is None:
            raise ValueError('sampling_params must be given.')
        self.sampling_params = sampling_params


