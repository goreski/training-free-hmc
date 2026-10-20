import pandas as pd

from tfhmc.experiments import read_reruns, update_csv


def test_read_reruns_takes_the_final_epoch(tmp_path):
    (tmp_path / "cellcycle_FUN.csv").write_text("0,0,0.20\n0,1,0.27\n0,2,0.25\n1,0,0.21\n1,1,0.24\n1,2,0.26\n")
    (tmp_path / "notes.csv").write_text("0,0,0.5\n")
    r = pd.DataFrame(read_reruns(tmp_path, "SPUR")).set_index("seed")
    assert r.loc[0, "final"] == 0.25 and r.loc[0, "best_on_test"] == 0.27 and r.loc[0, "epochs"] == 3
    assert r.loc[1, "final"] == 0.26 and (r["dataset"] == "cellcycle").all() and (r["method"] == "SPUR").all()


def test_update_csv_replaces_only_the_new_datasets(tmp_path):
    path = tmp_path / "out.csv"
    update_csv([{"dataset": "a", "v": 1}, {"dataset": "b", "v": 2}], path)
    update_csv([{"dataset": "b", "v": 3}], path)
    update_csv([], path)
    assert pd.read_csv(path).set_index("dataset")["v"].to_dict() == {"a": 1, "b": 3}
