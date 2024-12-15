import torch
import torch.utils
import torch.utils.data
import numpy as np
import math
from typing import Literal, Callable
import json
import biosig
import os
import random
import scipy.signal as signal

EVENT_TYPES = {
    "0x0114" : ("Idling EEG (eyes open)",276),
    "0x0115" : ("Idling EEG (eyes closed)",277),
    "0x0300" : ("Start of a trial",768),
    "0x0301" : ("Cue onset left (class 1)",769),
    "0x0302" : ("Cue onset right (class 2)",770),
    "0x030d" : ("BCI feedback (continuous)",781),
    "0x030f" : ("Cue unknown",783),
    "0x03ff" : ("Rejected trial",1023),
    "0x0430" : ("Eye movements",1072),
    "0x0435" : ("Horizontal eye movement",1077),
    "0x0436" : ("Vertical eye movement",1078),
    "0x0437" : ("Eye rotation",1079),
    "0x0439" : ("Eye blinks",1081),
    "0x7ffe" : ("Start of a new run",32766)
}
def low_pass_filter(data: np.ndarray, threshold: float = 0.5, sample_rate: int = 250, order: int = 4):
    b_low, a_low = signal.butter(order, threshold, btype='low', fs=sample_rate)
    low_passed = signal.filtfilt(b_low, a_low, data)
    return low_passed
def high_pass_filter(data: np.ndarray, threshold: float = 30, sample_rate: int = 250, order: int = 4):
    b_high, a_high = signal.butter(order, threshold, btype='high', fs=sample_rate)
    high_passed = signal.filtfilt(b_high, a_high, data)
    return high_passed
class MotorImageryDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, event, metadata, item_length, selected_features: list[str], filters: list[Callable[[np.ndarray], np.ndarray]] = []):
        super().__init__()
        self.data: np.ndarray = np.nan_to_num(data.T, 0)
        for filter in filters:
            for i in range(len(self.data)):
                self.data[i] = filter(self.data[i])
        self.data = self.data.T
        self.event: list[tuple] = event
        self.metadata = metadata
        self.item_length = item_length
        self.index_mapper = {}
        self.selected_features = selected_features
        self.count = 0
        for index, value in enumerate(self.event):
            if value["TYP"] in self.selected_features:
                self.index_mapper[self.count] = index
                self.count += 1
    def __len__(self):
        return self.count
    '''
    [bsz, samples, channels]
    '''
    def __getitem__(self, index):
        real_index = self.index_mapper[index]
        event_id = self.selected_features[self.event[real_index]["TYP"]]
        data_start_it = math.floor(self.event[real_index]["POS"] * self.metadata["Samplingrate"])
        if (real_index + 1 < len(self.event)):
            data_end_it = math.ceil(self.event[real_index+1]["POS"] * self.metadata["Samplingrate"])
            event_data = self.data[data_start_it:data_end_it]
        else:
            event_data = self.data[data_start_it:]
        if event_data.shape[0] < self.item_length:
            event_data = np.pad(event_data, ((0, self.item_length - event_data.shape[0]), (0, 0)), mode='constant', constant_values=0)
        event_data = event_data[:self.item_length]
        return event_data, event_id
class Util:
    @classmethod
    def read_gdf(cls, file_path : str
                ) -> tuple[
                    np.ndarray,
                    dict[Literal["TYPE", "VERSION", "Filename", "NumberOfChannels", "NumberOfRecord", "SamplesPerRecored", "NumberOfSamples", "Samplingrate", "StartOfRecording", "TimezoneMinutesEastOfUTC", "NumberOfSweeps", "NumberOfGroupsOrUserSpecifiedEvents", "Patient"], object],
                    list[dict[Literal['TYP', 'POS', 'DUR', 'Description'],object]],
                    list[dict[Literal['ChannelNumber', 'Label', 'Samplingrate', 'Transducer', 'PhysicalMaximum', 'PhysicalMinimum', 'DigitalMaximum', 'DigitalMinimum', 'scaling', 'offset', 'Filter', 'Impedance', 'PhysicalUnit'], object]]
                ]:
        '''
        Return data, metadata, events data, channels data
        data : [n_t, n_channels],
        channel_data : [n_channels],
        events : [n_t, ...]
        '''
        header : dict[str, str | float, int , dict, object] = json.loads(biosig.header(file_path))
        data : np.ndarray = biosig.data(file_path)
        metadata = {
            "TYPE" : header["TYPE"],
            "VERSION" : header["VERSION"],
            "Filename" : header["Filename"],
            "NumberOfChannels" : header["NumberOfChannels"],
            "NumberOfRecord" : header["NumberOfRecords"],
            "SamplesPerRecored" : header["SamplesPerRecords"],
            "NumberOfSamples" : header["NumberOfSamples"],
            "Samplingrate" : header["Samplingrate"],
            "StartOfRecording" : header["StartOfRecording"],
            "TimezoneMinutesEastOfUTC" : header["TimezoneMinutesEastOfUTC"],
            "NumberOfSweeps" : header["NumberOfSweeps"],
            "NumberOfGroupsOrUserSpecifiedEvents" : header["NumberOfGroupsOrUserSpecifiedEvents"],
            "Patient" : header["Patient"]
        }
        channels = header["CHANNEL"]
        events = header["EVENT"]
        return data, metadata, events, channels
    @classmethod
    def read_single_data(cls,
            select_features: dict[str, int],
            file_path: str,
            item_length: int = 1000,
            filters = []
            ):
        data, metadata, events, channels = cls.read_gdf(file_path)
        dataset = MotorImageryDataset(data, events, metadata, item_length, select_features, filters)
        total_length = len(data)
        return dataset, total_length, len(events)
    @classmethod
    def get_data_loader(cls,
            select_features: dict[str, int],
            folder_path: str,
            n_workers: int,
            bsz: int,
            shuffle: bool = False,
            item_length: int = 1000,
            sample_rate: int = 250        
        ):
        order = 4
        def lowpass(data: np.ndarray):
            return low_pass_filter(data, 30, sample_rate, order)
        def highpass(data: np.ndarray):
            return high_pass_filter(data, 8, sample_rate, order)
        filters = [
            lowpass,
            highpass
        ]
        dataset = []
        total_length = 0
        total_count = 0
        for file_name in os.listdir(folder_path):
            file_path =folder_path + "/" + file_name
            dataset_, total_length_, count_ = cls.read_single_data(select_features, file_path, item_length, filters)
            total_length += total_length_
            total_count += count_
            for i in range(len(dataset_)):
                data, label = dataset_[i]
                dataset.append((data, label))
            # break
        dataset_size = len(dataset)
        random.shuffle(dataset)
        split_index = int(dataset_size * 0.9)
        train_dataset = dataset[:split_index]
        test_dataset = dataset[split_index:]
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=n_workers, batch_size=bsz, shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=n_workers, batch_size=bsz, shuffle=shuffle)
        return train_loader, test_loader, (total_length, total_count)
    
