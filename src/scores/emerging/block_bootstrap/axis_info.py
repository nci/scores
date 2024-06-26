"""
Utilty module to provide methods to represent axis information
"""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import xarray as xr

from scores.emerging.block_bootstrap.methods import FitBlocksMethod


@dataclass(eq=True)
class AxisInfo:
    """
    Structure to hold axis information
    """

    length_in: int
    block_size: int
    fit_blocks_method: FitBlocksMethod
    bootstrap: bool

    # derived members:
    length_out: int = field(init=False)
    num_blocks: int = field(init=False)
    block_size_partial: int = field(init=False)

    @staticmethod
    def make_from_array(
        arr: npt.NDArray,
        block_sizes: list[int],
        fit_blocks_method: FitBlocksMethod,
    ) -> list[AxisInfo]:
        """
        Returns list of AxisInfo (outer-most axis -> inner-most axis), given a numpy
        ndarray as input
        """
        assert len(arr.shape) == len(block_sizes)

        return [
            AxisInfo(length_in=l, block_size=b, fit_blocks_method=fit_blocks_method)
            for l, b in zip(np.shape(arr), block_sizes)
        ]

    def __post_init__(self):
        """
        Adjust the axis length, block size and number of blocks based
        on the method used to fit blocks in the axis.
        """
        self._validate_init()

        (l, b) = (self.length_in, self.block_size)
        n = l // b  # multiple
        r = l - (n * b)  # remainder

        if (n, r) == (0, 0):
            raise ValueError("Empty block")
        if r == 0:
            self.num_blocks = n
            self.length_out = l
            self.block_size_partial = 0
        else:
            # key=method, value=(length_out, num_blocks, block_size_partial)
            fit_blocks_params = {
                FitBlocksMethod.PARTIAL: (l, n + 1, r),
                FitBlocksMethod.SHRINK_TO_FIT: (n * b, n, 0),
                FitBlocksMethod.EXPAND_TO_FIT: ((n + 1) * b, n + 1, 0),
            }
            try:
                (
                    self.length_out,
                    self.num_blocks,
                    self.block_size_partial,
                ) = fit_blocks_params[self.fit_blocks_method]
            except KeyError as e:
                raise NotImplementedError(f"Unsupported method: {self.fit_blocks_method}") from e

        self._validate_post_init()

    def _validate_init(self):
        """
        TODO: Add more validation checks here
        """
        assert self.length_in > 0 and self.block_size <= self.length_in

    def _validate_post_init(self):
        """
        TODO: Add more validation checks here
        """
        assert self.length_out > 0 and self.block_size <= self.length_out


@dataclass
class AxisInfoCollection:
    """
    Collection of labelled AxisInfo object.
    """

    fit_blocks_method: FitBlocksMethod
    dims_order: list[str] = field(init=False)
    data: dict[str, AxisInfo] = field(init=False)

    @staticmethod
    def make_from_arrays(
        arrs: list[xr.DataArray],
        bootstrap_dims: list[str],
        block_sizes: list[int],
        fit_blocks_method: FitBlocksMethod,
    ) -> AxisInfoCollection:
        axi_collection = AxisInfoCollection(fit_blocks_method=fit_blocks_method)
        lookup_block_sizes = {d: b for (d, b) in zip(bootstrap_dims, block_sizes)}

        for arr in arrs:
            for dim_name, axis_length in zip(arr.dims, np.shape(arr)):
                block_size = lookup_block_sizes.get(dim_name, axis_length)  # default to axis_length
                bootstrap = True if dim_name in bootstrap_dims else False

                axi = AxisInfo(
                    length_in=axis_length,
                    block_size=block_size,
                    fit_blocks_method=fit_blocks_method,
                    bootstrap=bootstrap,
                )

                axi_collection.insert(dim_name, axi)

        return axi_collection

    def __post_init__(self):
        self.data = {}
        self.dims_order = []

    def get(self, dim_name: str) -> AxisInfo:
        return self.data[dim_name]

    def iter(self):
        for d in self.dims_order:
            yield self.data[d]

    def items(self):
        for d in self.dims_order:
            yield (d, self.data[d])

    def insert(self, dim_name: str, axis_info: AxisInfo):
        if dim_name in self.data:
            # check that the info present is consistent
            if self.data[dim_name] != axis_info:
                raise ValueError(
                    f"Inconsistent dimension size for {dim_name} between inserted and existing"
                    f" dimension of the same name in collection:\n"
                    f"Existing axis info in collection: {self.data[dim_name]}\n"
                    f"Axis info being inserted: {axis_info}"
                )
            # already exists => early return without updating.
            return

        # The whole collection should have the same `fit_blocks_method`.
        # This should be always be equal if the collection is created through
        # `make_axis_info_collection()`, but may not be if created using arbitrary methods.
        assert axis_info.fit_blocks_method == self.fit_blocks_method

        # implicit: if we've reached this point, dim_name is not in self.data
        self.data[dim_name] = axis_info
        self.dims_order.append(dim_name)
