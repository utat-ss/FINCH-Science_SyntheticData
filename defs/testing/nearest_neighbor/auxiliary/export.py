"""
Code included here helps with exporting the metrics and blobs
"""

import pandas as pd

def export_metrics(metric_dict:(dict), cfg_export:(dict)) -> None:
    """
    Takes in the metrics, and exports them as a csv

    Args:
        metric_dict (dict): A dict of metrics, where each key is the omega combination, and values are the actual omega dict and metrics
        cfg_export (dict): Dict that includes path for export
    """

    metrics_save = cfg_export['metrics_save']
    if not metrics_save.endswith('.csv'): metrics_save += '.csv'

    column_order = ['name', 'type', 'method', 'spec_idx', 'p', 'delta', 'epsilon', 'zeta', 'gamma_11', 'gamma_12', 'gamma_21', 'gamma_22']
    rows_list = []

    for omega in list(metric_dict.keys()):
        payload = {'name': omega}
        payload.update(metric_dict[omega])
        rows_list.append(payload)

    df = pd.DataFrame(rows_list, ignore_index=True)
    df = df.reindex(columns=column_order)
    df.to_csv(metrics_save, index=False)

def export_blobs():

    pass