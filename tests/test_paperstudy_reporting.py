import pandas as pd

from dcl.paperstudy import report


def test_heldout_prediction_summary_excludes_later_nonheldout_battery(tmp_path,monkeypatch):
    output=tmp_path/"results"
    paper=tmp_path/"paper"
    output.mkdir();(paper/"figures").mkdir(parents=True)
    monkeypatch.setattr(report,"OUTPUT",output)
    monkeypatch.setattr(report,"PAPER",paper)
    monkeypatch.setattr(report,"ROOT",tmp_path/"runtime")
    pd.DataFrame([
        {"label":"historical-e1-s0-epoch120","heldout_prediction":True,"palette":17,
         "mean_prediction_rank_error_conditional_allowed":.25,"forbidden_rate":0.},
        {"label":"historical-e1-s0-epoch120","heldout_prediction":False,"palette":201,
         "mean_prediction_rank_error_conditional_allowed":99.,"forbidden_rate":1.},
    ]).to_csv(output/"candidate-diagnostics.csv",index=False)
    report.candidate_summary()
    result=pd.read_csv(output/"candidate-prediction-summary.csv")
    assert result.palette.tolist()==[17]
    assert result.mean_state_rank_error.tolist()==[.25]
    assert result.max_forbidden_grid_rate.tolist()==[0.]
