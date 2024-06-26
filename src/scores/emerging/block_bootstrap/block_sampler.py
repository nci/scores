"""
Module containing block sampling methods
"""
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field

from scores.emerging.block_bootstrap.array_info import ArrayInfoCollection, AxisInfo


AxisBlockIndices = np.ndarray[int]
BootstrapIndicesMapping = dict[str, AxisBlockIndices]
MultiIndex = list[int]


@dataclass
class BlockSampler:
    #: bootstrapping information for the various arrays
    array_info_collection: ArrayInfoCollection
    #: whether or not to use cyclic sampling
    cyclic: bool
    #: cache of sampled axis block indices for all arrays
    bootstrap_idx_map: BootstrapIndicesMapping = field(init=False)

    def resample_indices(self):
        """
        Gets a new random sample of sequential block indices for each dimension
        in the array collection.
        """
        self.bootstrap_idx_map = {
            dim_name: generate_sample_indices(axis_info, self.cyclic)
            for dim_name, axis_info in array_info_collection.axis_info_collection.items()
        }

    def sample_blocks_unchecked(self, arrs) -> list[np.ndarray]:
        """
        Returns the sampled blocks for each array in sequence.

        .. note::

            length, order and dimensions of the list of input arrays must match
            the axis info collection.
        """
        # randomize sampled blocks every time this method is called
        self.resample_indices()

        arrs_bootstrapped = []

        for arr, arr_info in zip(arrs, self.array_info_collection.array_info):
            # reset array to desired axis lengths
            arr_out = np.empty(arr_info.output_axis_lengths)

            # --- iterate over each block and map to output ---
            #
            # dummy array for multi-index looping
            # note: this may use more memory, but will be faster than using
            # a native for loop. Can be improved using `numba` in the future.
            dummy_it = np.empty(arr_info.num_blocks)

            with np.nditer(dummy_it, flags=["multi_index"], op_flags=["readonly"]) as it:
                for _ in it:
                    # get output block indices
                    idx_out = tuple(
                        slice(i * b, min((i + 1) * b, l))
                        for i, b, l in zip(
                            it.multi_index,
                            arr_info.block_sizes,
                            arr_info.output_axis_lengths,
                        )
                    )

                    # get sample block values
                    sampled_block = self.generate_sample_block_values(
                        arr,
                        arr_info,
                        it.multi_index,
                    )

                    # map output indices to sampled block
                    output_arr[idx_out] = sampled_block

            arrs_bootstrapped.append(arr_out)
            # ---

        return arrs_bootstrapped

    def generate_sample_block_values(
        input_arr: npt.NDArray,
        arr_info: ArrayInfo,
        ax_idx: MultiIndex,
    ):
        """
        Returns a sampled block of values from the input array, using a mult-index
        pivot point and a reference pre-sampled block indices for each axis.
        Usually used by an outer loop to retrieve a block of values from
        pre-sampled indices for each axis.

        Args:
            input_arr: input array N-dimensional input array.
            ax_info: (ordered) list of information for each axis.
            ax_idx: 1-D multi-index to determine which block to retrieve the sample
                values from.

        suppose

        .. code-block::

            ax_idx = [1,2,0,3]
            ax_block_indices = [
                # outermost axis
                [[1,2,3], [4,5,6]],  # ax_i = 0, block_size = 3, num_blocks = 2
                [[1], [9], [5]],     # ax_i = 1, block_size = 1, num_blocks = 3
                [[1,2,3,4,5]],       # ax_i = 2, block_size = 5, num_blocks = 1
                [[0,1], [1,2], [3,4], [4,0]],  # ax_i = 3, block_size = 2, num_blocks = 4
                # inner most axis
            ]

        then we would expect the output block sample to be based on the following indices,

        .. code-block::

            ax_0 = [4,5,6]
            ax_1 = [5]
            ax_2 = [1,2,3,4,5]
            ax_3 = [4,0]

        which will result in the following multi indices from the input array,

        .. code-block::

            [4,5,1,4], [4,5,1,0], [4,5,2,4] ... [6,5,5,4], [6,5,5,0]

        this can be mapped to the following block in the output array:

        .. code-block::

            output_arr[1:4, 2:3, 0:5, 3:5]

        .. note::

            - There may be advanced indexing alternatives that may also work, see:
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
            - Likely to be superseeded by `numba`/`dask` implementation.
        """
        bootstrap_indices = [self.bootstrap_idx_map[dim] for dim in arr_info.bootstrap_dims]

        # trim final blocks if partial blocks are allowed
        if self.fit_blocks_method == FitBlocksMethod.PARTIAL:
            bootstrap_ax_idx = []
            for b, i, axi in zip(bootstrap_indices, ax_idx, ax_info):
                (n, bp) = (axi.num_blocks, axi.block_size_partial)
                if i == (n - 1) and bp > 0:
                    bootstrap_ax_idx.append(b[i][0:bp])
                else:
                    bootstrap_ax_idx.append(b[i])
        else:
            # no trimming required in other methods
            bootstrap_ax_idx = [b[i] for b, i in zip(bootstrap_indices, ax_idx)]

        # --- iterate and map individual elements in block ---
        #
        # dummy array for multi-index looping
        # note: this may use more memory, but will be faster than using
        # a native for loop. Can be improved using `numba` in the future.
        dummy_it = np.empty([len(i) for i in bootstrap_ax_idx])

        # retrieve block values from input array and write it to block sample
        with np.nditer(dummy_it, flags=["multi_index"], op_flags=["writeonly"]) as it:
            for x in it:
                bootstrap_idx = tuple(b[i] for b, i in zip(bootstrap_ax_idx, block_size_it.multi_index))
                x[...] = input_arr[bootstrap_idx]
        # ---

        return block_sample

    def generate_sample_indices(self, axis_info: AxisInfo) -> AxisBlockIndices:
        """
        Randomly generate block indices for each axis.

        Args:
            ax_info: (ordered) list of information for each axis.
            cyclic:
                True  => indices can cycle around if they overflow past the axis
                         length
                False => blocks will not be sampled from indices that overflow past
                         the axis length

        Returns a list of 2-D arrays N by B of random block indices for each axis, where

            - N = number of blocks
            - B = block size
        """
        rng = np.random.default_rng()

        # note: this can probably just be a for loop, but separated out for clarity
        def _cyclic_expand(idx_, block_size_, length_):
            i = 0
            while i < block_size_:
                yield (idx_ + i) % length_
                i = i + 1

        def _linear_expand(idx_, block_size_):
            return np.arange(start=idx_, stop=idx_ + block_size_)

        (l, b, n) = (axi.length_in, axi.block_size, axi.num_blocks)
        cyc_fn = functools.partial(_cyclic_expand, block_size_=b, length_=l)
        lin_fn = functools.partial(_linear_expand, block_size_=b)

        if axi.bootstrap:
            if self.cyclic:
                # sample from 0 -> l - 1, since wrap around is allowed
                start_idx = rng.integers(low=0, high=l - 1, size=n)
                block_idx = np.apply_along_axis(lambda x: np.array(list(cyc_fn(x))), 0, start_idx).T
            else:
                # sample from 0 -> l - b, to avoid overflow
                start_idx = rng.integers(low=0, high=l - b, size=n)
                block_idx = np.apply_along_axis(lin_fn, 0, start_idx).T
        else:
            # no bootstrapping for this axis => append entire axis domain
            block_idx = np.array([_linear_expand(0, l)])

        return block_idx
