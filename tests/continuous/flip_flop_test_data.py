"""
Test data for tests.continuous.flip_flop in scores
"""
import numpy as np
import xarray as xr

"""
Test data for flip_flop_index and related functions
"""
DATA_FFI_1D_6 = xr.DataArray([20, 50, 60, 30, 10, 20], coords=[("letter", ["a", "b", "c", "d", "e", "f"])])
DATA_FFI_1D_6_WITH_NAN = xr.DataArray([20, 50, 60, 30, np.nan, 20], coords=[("letter", ["a", "b", "c", "d", "e", "f"])])
DATA_FFI_2D_6 = xr.DataArray(
    [[20, 50, 60, 30, 10, 20], [20, 50, 60, 30, np.nan, 20]],
    coords=[("number", [1, 2]), ("letter", ["a", "b", "c", "d", "e", "f"])],
)

DATA_FFI_1D_6_ABC = xr.DataArray([20, 50, 60], coords=[("letter", ["a", "b", "c"])])
DATA_FFI_1D_6_BCD = xr.DataArray([50, 60, 30], coords=[("letter", ["b", "c", "d"])])
DATA_FFI_1D_6_ACEF = xr.DataArray([20, 60, 10, 20], coords=[("letter", ["a", "c", "e", "f"])])
DATA_FFI_1D_6_WITH_NAN_ACE = xr.DataArray([20, 60, np.nan], coords=[("letter", ["a", "c", "e"])])
DATA_FFI_3D = xr.DataArray(
    [
        [[np.pi, 0, 2 * np.pi], [np.nan, 3.4, -100]],
        [[-20, -40, -10], [0.5, 0.6, 0.1]],
    ],
    coords=[
        ("fruit", ["apple", "pear"]),
        ("number", [1, 2]),
        ("letter", ["a", "b", "c"]),
    ],
)
DATA_FFI_3D_DIR = xr.DataArray(
    [
        [[10, 350, 20], [np.nan, 90, 100]],
        [[0, 240, 300], [100, 0, 200]],
    ],
    coords=[
        ("fruit", ["apple", "pear"]),
        ("number", [1, 2]),
        ("letter", ["a", "b", "c"]),
    ],
)
DATA_FFI_2X2X4 = xr.DataArray(
    [
        [[-5, 3, 2.5, 1], [np.nan, -4, 9, 4]],
        [[3.14, 0, 6.28, 10], [-2, -3, 0, -5]],
    ],
    coords=[("char", ["a", "b"]), ("bool", [True, False]), ("int", [1, 2, 3, 4])],
)
DATA_FFI_2X2X4_DIR = xr.DataArray(
    [
        [[350, 30, 60, 10], [np.nan, 40, 190, 100]],
        [[300, 270, 200, 190], [120, 300, 10, 50]],
    ],
    coords=[("char", ["a", "b"]), ("bool", [True, False]), ("int", [1, 2, 3, 4])],
)

"""
Test data for flip_flop_index
"""
EXP_FFI_CASE8 = xr.DataArray(
    [[np.pi, np.nan], [20, 0.1]],
    coords=[("fruit", ["apple", "pear"]), ("number", [1, 2])],
)
EXP_FFI_CASE9 = xr.DataArray(
    [[20.0, np.nan], [60.0, 80.0]],
    coords=[("fruit", ["apple", "pear"]), ("number", [1, 2])],
)


"""
Test data for flip_flop_index_subsets
"""
DICT_FFI_SUB_CASE4 = {
    "zero": ["b", "c", "d"],
    "one": ["e", "a", "c"],
    "two": ["a", "c", "e"],
}

EXP_FFI_SUB_CASE0 = xr.DataArray(12.5, attrs={"sampling_dim": "letter"})
EXP_FFI_SUB_CASE1 = xr.DataArray(np.nan, attrs={"sampling_dim": "letter"})
EXP_FFI_SUB_CASE2 = xr.DataArray(
    [12.5, np.nan],
    dims=["number"],
    coords={"number": [1, 2]},
    attrs={"sampling_dim": "letter"},
)
EXP_FFI_SUB_CASE3 = xr.DataArray(np.pi, attrs={"sampling_dim": "pi"})

EXP_FFI_SUB_CASE4 = xr.Dataset(
    {"back_to_front": xr.DataArray(0.0)},
    attrs={"sampling_dim": "letter", "selections": {"back_to_front": ["e", "a", "c"]}},
)
EXP_FFI_SUB_CASE5 = xr.Dataset(
    {
        "zero": xr.DataArray([10.0, 10.0], dims=["number"], coords={"number": [1, 2]}),
        "one": xr.DataArray([0.0, np.nan], dims=["number"], coords={"number": [1, 2]}),
        "two": xr.DataArray([40.0, np.nan], dims=["number"], coords={"number": [1, 2]}),
    },
    attrs={"sampling_dim": "letter", "selections": DICT_FFI_SUB_CASE4},
)
EXP_FFI_SUB_CASE6 = xr.Dataset(
    {"banana": xr.DataArray([25.0, np.nan], dims=["number"], coords={"number": [1, 2]})},
    attrs={"sampling_dim": "letter", "selections": {"banana": ["a", "c", "e", "f"]}},
)
EXP_FFI_SUB_CASE7 = xr.Dataset(
    {"banana": EXP_FFI_CASE8},
    attrs={"sampling_dim": "letter", "selections": {"banana": ["a", "b", "c"]}},
)
EXP_FFI_SUB_CASE8 = xr.Dataset(
    {"irrational": xr.DataArray(np.pi)},
    attrs={"sampling_dim": "pi", "selections": {"irrational": [np.e, np.pi, 3.14]}},
)
EXP_FFI_SUB_CASE9 = xr.Dataset(
    {
        "one": xr.DataArray(
            [[0.5, np.nan], [3.14, 1]],
            coords=[("char", ["a", "b"]), ("bool", [True, False])],
        ),
        "two": xr.DataArray(
            [[0.0, 5.0], [0.0, 3.0]],
            coords=[("char", ["a", "b"]), ("bool", [True, False])],
        ),
    },
    attrs={"sampling_dim": "int", "selections": {"one": [1, 2, 3], "two": [2, 3, 4]}},
)
EXP_FFI_SUB_CASE10 = xr.Dataset(
    {
        "3letters": xr.DataArray([40.0, np.nan], dims=["number"], coords={"number": [1, 2]}),
        "4letters": xr.DataArray([25.0, np.nan], dims=["number"], coords={"number": [1, 2]}),
    },
    attrs={
        "sampling_dim": "letter",
        "selections": {"3letters": ["a", "c", "e"], "4letters": ["a", "c", "e", "f"]},
    },
)


"""
Test data for encompassing_sector_size
"""
ESS = xr.DataArray(
    data=[
        [[10, 20, 30, 40], [50, 150, 200, 250], [0, 90, 180, 270]],
        [[np.nan, 20, 30, 40], [np.nan, np.nan, np.nan, np.nan], [90, 90, 90, 90]],
    ],
    dims=["i", "j", "k"],
    coords={"i": [1, 2], "j": [1, 2, 3], "k": [1, 2, 3, 4]},
)
EXP_ESS_DIM_K = xr.DataArray(
    data=[[30, 200, 270], [np.nan, np.nan, 0]],
    dims=["i", "j"],
    coords={"i": [1, 2], "j": [1, 2, 3]},
)
EXP_ESS_DIM_K_SKIPNA = xr.DataArray(
    data=[[30, 200, 270], [20.0, np.nan, 0]],
    dims=["i", "j"],
    coords={"i": [1, 2], "j": [1, 2, 3]},
)
EXP_ESS_DIMS_J_AND_K = xr.DataArray(data=[120, np.nan], dims=["i"], coords={"i": [1, 2]})
EXP_ESS_DIM_K_1J = xr.DataArray(data=[[30], [np.nan]], dims=["i", "j"], coords={"i": [1, 2], "j": [1]})


ENC_SIZE_NP_TESTS_SKIPNA = [
    # only 2 angles. answer is for skipna=True [2nd tuple element]
    (
        [0, 90, 90, 90, 90],
        90,
    ),
    (
        [0, 45, 0, 0, 0],
        45,
    ),
    (
        [0, 45, 44, 42, 44],
        45,
    ),
    (
        [90, 90, 89, 88, 90],
        2,
    ),
    ## other
    ([0, 0, 0, 45, 90], 90),
    ([0, 360, 720, 45, 90], 90),
    ([0, 1, 5, 45, 90], 90),
    ([0, 85, 88, 89, 90], 90),
    ([0, np.nan, 179, np.nan, np.nan], 179),
    ([0, 30, 179, np.nan, np.nan], 179),
    ([0, 30, 181, np.nan, np.nan], 181),
    ([30, 0, np.nan, 181, 3], 181),
    ([np.nan, 1, np.nan, 182, 3], 181),
    ([np.nan, np.nan, np.nan, np.nan, np.nan], np.nan),
]
ENC_SIZE_NP_TESTS_NOSKIPNA = []
for test, ans in ENC_SIZE_NP_TESTS_SKIPNA:
    if np.nan in test:
        ENC_SIZE_NP_TESTS_NOSKIPNA.append(
            (
                test,
                np.nan,
            )
        )
    else:
        ENC_SIZE_NP_TESTS_NOSKIPNA.append(
            (
                test,
                ans,
            )
        )

ENC_SIZE_3D_TEST_AXIS2 = np.array(
    [
        [
            [0, 30, 181, np.nan],
            [3, 3, 3, 3],
        ],  # <= this is an individual testcase.
        [
            [0, 30, 179, 0],
            [np.nan, np.nan, np.nan, np.nan],
        ],
    ]
)

ENC_SIZE_3D_ANSW_AXIS2_SKIPNA = np.array(
    [
        [
            181,
            0,
        ],
        [
            179,
            np.nan,
        ],
    ]
)
ENC_SIZE_3D_ANSW_AXIS2_NOSKIPNA = np.array(
    [
        [
            np.nan,
            0,
        ],
        [
            179,
            np.nan,
        ],
    ]
)
ENC_SIZE_3D_TEST_AXIS0 = ENC_SIZE_3D_TEST_AXIS2.swapaxes(0, 2)
ENC_SIZE_3D_ANSW_AXIS0_SKIPNA = ENC_SIZE_3D_ANSW_AXIS2_SKIPNA.swapaxes(0, 1)
ENC_SIZE_3D_ANSW_AXIS0_NOSKIPNA = ENC_SIZE_3D_ANSW_AXIS2_NOSKIPNA.swapaxes(0, 1)


"""
Test data for flipflipindex_proportion_exceeding
"""
EXP_FFI_PE_NONE = xr.Dataset(
    {
        "one": xr.DataArray([1, 2 / 3, 0], coords=[("threshold", [0, 1, 5])]),
        "two": xr.DataArray([1, 0.5, 0.25], coords=[("threshold", [0, 1, 5])]),
    },
    attrs={"sampling_dim": "int", "selections": {"one": [1, 2, 3], "two": [2, 3, 4]}},
)
EXP_FFI_PE_CHAR = xr.Dataset(
    {
        "one": xr.DataArray(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            coords=[("char", ["a", "b"]), ("threshold", [0, 1, 5])],
        ),
        "two": xr.DataArray(
            [[1, 0.5, 0.5], [1, 0.5, 0]],
            coords=[("char", ["a", "b"]), ("threshold", [0, 1, 5])],
        ),
    },
    attrs={"sampling_dim": "int", "selections": {"one": [1, 2, 3], "two": [2, 3, 4]}},
)
EXP_FFI_PE_CHARBOOL = xr.Dataset(
    {
        "one": xr.DataArray(
            [[[1, 0, 0], [np.nan, np.nan, np.nan]], [[1, 1, 0], [1, 1, 0]]],
            coords=[
                ("char", ["a", "b"]),
                ("bool", [True, False]),
                ("threshold", [0, 1, 5]),
            ],
        ),
        "two": xr.DataArray(
            [[[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]],
            coords=[
                ("char", ["a", "b"]),
                ("bool", [True, False]),
                ("threshold", [0, 1, 5]),
            ],
        ),
    },
    attrs={"sampling_dim": "int", "selections": {"one": [1, 2, 3], "two": [2, 3, 4]}},
)
EXP_FFI_PE_CHARBOOL_DIR = xr.Dataset(
    {
        "one": xr.DataArray(
            [
                [[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            coords=[
                ("char", ["a", "b"]),
                ("bool", [True, False]),
                ("threshold", [0, 50, 100]),
            ],
        ),
        "two": xr.DataArray(
            [[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            coords=[
                ("char", ["a", "b"]),
                ("bool", [True, False]),
                ("threshold", [0, 50, 100]),
            ],
        ),
    },
    attrs={"sampling_dim": "int", "selections": {"one": [1, 2, 3], "two": [2, 3, 4]}},
)
