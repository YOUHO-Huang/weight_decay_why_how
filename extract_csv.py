import os
import glob
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# run_folder='/home/dxiaog/PycharmProjects/Generalization-AD-SGD/result_plot/tensorboard_runs/linearnet_runs/topo_runs'
run_folder = './logs/'

# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


# for runs_name in os.listdir(run_folder):
#     print(runs_name)

#     event_paths = glob.glob(os.path.join(run_folder, "event*"))
#     print(event_paths)
#     df = many_logs2pandas(event_paths)
#     # print(df)

#     # select the mentioned rows
#     df = df[(df.metric == 'acc') | (df.metric == 'loss') | (df.metric == 'test_loss')
#             | (df.metric == 'Eval/TestLoss') | (df.metric == 'Eval/TrainAcc') | (df.metric == 'Eval/TrainLoss')]
#     if not os.path.exists('CSV_data'):
#         os.mkdir('CSV_data')

#     df.to_csv(f"CSV_data/{runs_name}.csv")

for runs_name in os.listdir(run_folder):
    print(runs_name)
    for runs in os.listdir(run_folder+runs_name):
        print(runs)
        event_paths = glob.glob(os.path.join(run_folder+runs_name, "event*"))
        # event_paths = glob.glob(os.path.join(run_folder+runs_name, runs, "event*"))
        print(event_paths)
        df = many_logs2pandas(event_paths)
        print(df)

        # select the mentioned rows
        df = df[(df.metric == 'acc') | (df.metric == 'loss') | (df.metric == 'test_loss')
                | (df.metric == 'smoothness') | (df.metric == 'Eval/TrainAcc') | (df.metric == 'Eval/TrainLoss')]
        if not os.path.exists('CSV_data'):
            os.mkdir('CSV_data')
        # if not os.path.exists('CSV_data/{runs_name}/'):
        #     os.mkdir('CSV_data/{runs_name}/')
        df.to_csv(f"CSV_data/{runs_name}.csv")
