import numpy as np

def _normalize(matrix):
    variances = np.sqrt(np.diag(matrix))
    return matrix / np.expand_dims(variances,0) / np.expand_dims(variances,1)

def _average_triu(matrix):
    return np.sum(np.triu(matrix, 1)) / matrix.shape[0] / (matrix.shape[0]-1) * 2

def average_pearson(traces: np.array) -> float:
    # Each row is a trace.
    traces_noavg = traces - np.expand_dims(np.mean(traces, axis = 1), -1)
    similarity_matrix = np.dot(traces_noavg, traces_noavg.T)
    correlation_matrix = _normalize(similarity_matrix)
    return _average_triu(correlation_matrix)

def average_cosine_distance(traces: np.array) -> float:
    similarity_matrix = np.dot(traces, traces.T)
    correlation_matrix = _normalize(similarity_matrix)
    return _average_triu(correlation_matrix)

def spike_range(spike_trains: np.array) -> int:
    total_spikes = np.sum(spike_trains, axis = 1)
    return int(np.max(total_spikes) - np.min(total_spikes))

def spike_count(spike_trains: np.array) -> int:
    return np.sum(spike_trains)

def average_mutual_information():
    raise NotImplementedError

def overall_mutual_information():
    raise NotImplementedError
