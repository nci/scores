from scores.emerging.block_bootstrap import (
    AxisInfo,
    AxisInfoCollection,
)

from scores.emerging.block_bootstrap.helpers import (
    FitBlocksMethod,
    reorder_all_arr_dims,
)


@dataclass
class ArrayInfo:
    #: ordered list of axis information for array
    axis_info: list[AxisInfo]
    #: ordered list of dimensions for array that are bootstrapped
    bootstrap_dims: list[str]

    @property
    def block_sizes() -> list[int]:
        """
        list of block sizes for all unique dimensions
        """
        return [x.block_size for x in axis_info]

    @property
    def num_blocks() -> list[int]:
        """
        list of number of blocks for all unique dimensions
        """
        return [x.num_blocks for x in axis_info]

    @property
    def output_axis_lengths() -> list[int]:
        """
        list of desired output axis lengths for all unique dimensions
        """
        return [x.length_out for x in axis_info]

    @staticmethod
    def make_from_axis_collection(
        dims: list[str],
        axi_collection: AxisCollection,
    ):
        """
        Args:
            bootstrap_dims: dimensions to bootstrap for array
            axi_collection: Ordered collection containing axis information
                for all arrays, which this array must a subset of.
        """
        assert len(set(bootstrap_dims) - set(axi_collection.dims_order)) == 0

        axis_info = []
        bootstrap_dims = []

        # insert in the order of axi_collection
        for d, axi in axi_collection.items():
            if d in dims:
                axis_info.append(axi)
                bootstrap_dims.append(d)

        return ArrayInfo(axis_info, bootstrap_dims)


@dataclass
class ArrayInfoCollection:
    #: Ordered collection of axis information for all arrays
    axis_info_collection: AxisInfoCollection
    #: Ordered list of array information for each array
    array_info: list[ArrayInfo]
    #: Arrays with dimensions ordered according `make_from_arrays`
    arrays_ordered: list[xr.DataArray]

    @property
    def fit_blocks_method(self) -> FitBlocksMethod:
        return self.axis_info_collection.fit_blocks_method

    @staticmethod
    def make_from_arrays(
        arrs: list[xr.DataArray],
        bootstrap_dims: list[str],
        block_sizes: list[int],
        fit_blocks_method: FitBlocksMethod,
        auto_order_missing: bool = True,
    ):
        if len(block_sizes) != len(bootstrap_dims):
            ValueError("`block_sizes` must be the same size as `bootstrap_dims`.")

        # reorder dimensions to align them across all arrays
        (arrs_reordered, _) = reorder_all_arr_dims(
            arrs,
            bootstrap_dims,
            auto_order_missing,
        )

        axi_collection = AxisInfoCollection.make_from_arrays(
            arrs_reordered,
            bootstrap_dims,
            block_sizes,
            fit_blocks_method,
        )

        arr_info = []

        for arr in arrs:
            btrp_dims_for_arr = set(arr.dims) & set(bootstrap_dims)
            arr_info.append(
                ArrayInfo.make_from_axis_collection(
                    btrp_dims_for_arr,
                    axi_collection,
                )
            )

        return ArrayInfoCollection(
            axis_info_collection=axi_collection,
            array_info=arr_info,
            arrays_reordered=arrs_reordered,
        )
