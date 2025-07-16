# src/drift_detection.py

from evidently.report import Report
from evidently.metrics import DataDriftPreset
import pandas as pd

def detect_drift(ref_path, new_path):
    ref_data = pd.read_csv(ref_path)
    new_data = pd.read_csv(new_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=new_data)
    report.save_html("output/drift_report.html")
    print("Drift report saved to output/drift_report.html")
