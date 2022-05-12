import os
import pandas as pd
import pickle
import warnings
from torch.multiprocessing import Pool, Process, set_start_method, set_sharing_strategy
warnings.filterwarnings("ignore")

def generation(validator, start_event):
    # print("0th event:", start_event)
    events = validator.super_validate(start_event, event_num=5)
    return events


if __name__ == '__main__':
    # set_start_method('spawn')
    # set_sharing_strategy('file_system')
    data = pd.read_csv("src/data/cmu_scifi/train_flatten.csv", index_col=0)[:10000]
    # pivot start event
    #start_event = data["event"].iloc[42]
    # open the pickle file
    with open("src/data/evaluation/event_0_lst.pkl", "rb") as f:
        inits = [str(e) for e in pickle.load(f)][:5]
    print(inits)

    # generate events
    events = []
    validator = Validator(data["event"].to_list())
    for start_event in inits:
        event_list = generation(validator, start_event)
        # save the events
        events.append(event_list)
    # save the events
    with open("src/data/evaluation/event_generated_lst.pkl", "wb") as f:
        print(events)
        pickle.dump(events, f)


