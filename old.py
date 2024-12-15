def get_data(event_it: int, channel : int, events: list[dict[str, object]], data : np.ndarray, metadata : dict[str, object]) -> tuple[np.ndarray, int, str]:
    event_id = EVENT_TYPES[events[event_it]["TYP"]][1]
    data_start_it = math.floor(events[event_it]["POS"] * metadata["Samplingrate"])
    event_data = None
    if (event_it + 1 < len(events)):
        data_end_it = math.ceil(events[event_it+1]["POS"] * metadata["Samplingrate"])
        event_data = data[channel][data_start_it:data_end_it]
    else:
        event_data = data[channel][data_start_it:]
    return event_data,event_id,EVENT_TYPES[events[event_it]["TYP"]][0]
    
def prepare_data(data : np.ndarray, metadata : dict, events : list[dict], channels : list[dict]) ->tuple[np.ndarray,np.ndarray]:
    data_list : list[list[np.ndarray]] = []
    event_label : list[int] = []
    max_len = 0
    for event_it in range(len(events)):
        event_data, event_id, _ = get_data(
            event_it=event_it,
            channel=0,
            events=events,
            data=data.T,
            metadata=metadata
        )
        if (len(event_data) > max_len):
            max_len = len(event_data)
        new_data = [event_data]
        event_label.append(event_id)
        for i in range(1,len(channels)):
            new_data.append(get_data(
                event_it=event_it,
                channel=i,
                events=events,
                data=data.T,
                metadata=metadata
            )[0])
        data_list.append(new_data)
    final_data = np.zeros((len(events),len(channels),max_len))
    for i in range(len(final_data)):
        for j in range(len(channels)):
            final_data[i][j] = np.pad(data_list[i][j],(0,max_len-len(data_list[i][j])), mode='constant',constant_values=0)
    return final_data, np.array(event_label) 