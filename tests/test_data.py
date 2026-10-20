import numpy as np

from tfhmc.data import dataset_of, files, parse_arff

FUN_ARFF = """@RELATION 'toy'

@ATTRIBUTE a numeric
@ATTRIBUTE colour {red,blue}
@ATTRIBUTE b numeric
@ATTRIBUTE class hierarchical 01,01/01,01/02,02,02/01

@DATA
1.0,red,?,01/01
?,blue,2.0,01/02@02/01
3.0,green,4.0,02 % a comment
"""

GO_ARFF = """@RELATION 'go'

@ATTRIBUTE a numeric
@ATTRIBUTE class hierarchical root/GO0003674,GO0003674/A,GO0003674/B,A/C,B/C,C/D

@DATA
1.0,D
2.0,B@A
?,GO0003674
"""


def test_parse_tree(tmp_path):
    p = tmp_path / "toy.arff"
    p.write_text(FUN_ARFF)
    d = parse_arff(p)
    assert d["names"] == ["root", "01", "02", "01.01", "01.02", "02.01"]
    np.testing.assert_allclose(d["X"], [[1, 1, 0, 3], [2, 0, 1, 2], [3, 0, 0, 4]])
    np.testing.assert_array_equal(d["Y"], [[1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1], [1, 0, 1, 0, 0, 0]])


def test_parse_dag(tmp_path):
    p = tmp_path / "go.arff"
    p.write_text(GO_ARFF)
    d = parse_arff(p, is_go=True)
    assert d["names"] == ["root", "A", "B", "C", "D", "GO0003674"]
    assert d["par"] == [[], [5], [5], [1, 2], [3], [0]]
    np.testing.assert_allclose(d["X"], [[1.0], [2.0], [1.5]])
    np.testing.assert_array_equal(d["Y"], [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1]])


def test_windows_line_endings(tmp_path):
    p = tmp_path / "toy.arff"
    p.write_bytes(FUN_ARFF.replace("\n", "\r\n").encode())
    np.testing.assert_array_equal(parse_arff(p)["Y"][1], [1, 1, 1, 0, 1, 1])


def test_file_layout_and_names():
    paths, is_go = files("seq_GO", "data")
    assert is_go and paths[0].as_posix() == "data/datasets_GO/seq_GO/seq_GO.train.arff"
    assert files("spo", "data")[0][2].as_posix() == "data/datasets_FUN/spo_FUN/spo_FUN.test.arff"
    assert files("enron", "data")[0][0].as_posix() == "data/others/Enron_corr_trainvalid.arff"
    assert dataset_of("cellcycle_FUN") == "cellcycle" and dataset_of("cellcycle_GO") == "cellcycle_GO"
    assert dataset_of("diatoms_others") == "diatoms" and dataset_of("notes") is None
