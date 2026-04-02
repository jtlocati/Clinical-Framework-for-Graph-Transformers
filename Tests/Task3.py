import numpy
from dataclasses import dataclass

# Load a patient from chached tensor for Task 4 (48h mortality prediction) and print out the features, missing mask, and graph structure.
data = numpy.load("Task4/data/mimic4_icu_mortality_48h.npz")

X = data["X"] #[num_patients, time_steps, num_features]
M = data["M"] #[num_patients, time_steps, num_features]
y = data["y"] #[num_patients]
ids = data["ids"]  #[num_patients]

patient_idx = 0
stay_id = ids[patient_idx]
label = y[patient_idx]
features = X[patient_idx] #[48, 25]
missing_mask = M[patient_idx] #[48, 25]

print("================= ONE PATIENT FROM CACHED TENSOR =================")
print("stay_id:", stay_id)
print("label:", label)
print("features shape:", features.shape)
print("missing_mask shape:", missing_mask.shape)
print("first feature row:", features[0])
print("first mask row:", missing_mask[0])

#build graph:
@dataclass
class Graph:
    x: numpy.ndarray
    edge_index: numpy.ndarray
    edge_type: numpy.ndarray

class GraphBuilder:
    def __init__(self, edge_mode="both", clinical_groups=None):
        self.edge_mode = edge_mode
        self.clinical_groups = clinical_groups or {}

    def build_graph(self, features, missing_mask):
        #node feature == value plus missing masks
        x = numpy.concatenate([features, missing_mask], axis=1).astype(numpy.float32)
        edges = []
        edge_types = []
        num_time_bins = features.shape[0]

        #build temporal edges:
        if self.edge_mode in {"temporal_only", "both"}:
            for t in range(num_time_bins -1):
                edges.append((t, t+1))
                edge_types.append(0)
        
        #clinical edges
        if self.edge_mode in {"clinical_only", "both"}:
            for t in range(num_time_bins -1):
                for _, idxs in self.clinical_groups.items():
                    if missing_mask[t, idxs].sum() > 0 and missing_mask[t + 1, idxs].sum() > 0:
                        edges.append((t, t + 1))
                        edge_types.append(1)
                        break
        if edges:
            edge_index = numpy.array(edges, dtype=numpy.int64).T
            edge_type = numpy.array(edge_types, dtype=numpy.int64)
        else:
            edge_index = numpy.empty((2, 0), dtype=numpy.int64)
            edge_type = numpy.empty((0,), dtype=numpy.int64)

        return Graph(x=x, edge_index=edge_index, edge_type=edge_type)

# Temporary simple grouping: all features in one group. In practice, you can define more meaningful groups based on feature types (e.g., vitals, labs) or clinical relevance.
clinical_groups = {
    "all_features": list(range(features.shape[1]))
}

builder = GraphBuilder(edge_mode="both", clinical_groups=clinical_groups)
graph = builder.build_graph(features, missing_mask)

print("========================= GRAPH SUMMARY =========================")
print("stay_id:", stay_id)
print("label:", label)
print("x shape:", graph.x.shape)
print("edge_index shape:", graph.edge_index.shape)
print("edge_type shape:", graph.edge_type.shape)
print("first few edges:", graph.edge_index[:, :10] if graph.edge_index.shape[1] > 0 else graph.edge_index)

num_temporal = int((graph.edge_type == 0).sum())
num_clinical = int((graph.edge_type == 1).sum())

print("num time bins:", features.shape[0])
print("num raw features:", features.shape[1])
print("node feature dim:", graph.x.shape[1])
print("num edges:", graph.edge_index.shape[1])
print("temporal edges:", num_temporal)
print("clinical edges:", num_clinical)

def test_one_patient_graph(graph, features, missing_mask):
    assert features.ndim == 2
    assert missing_mask.ndim == 2
    assert features.shape == missing_mask.shape

    # x should be [features || mask]
    assert graph.x.shape[0] == features.shape[0]
    assert graph.x.shape[1] == features.shape[1] * 2

    # edge_index should have shape [2, E]
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == graph.edge_type.shape[0]

    # temporal edges should equal number of time bins minus 1
    temporal_edges = int((graph.edge_type == 0).sum())
    assert temporal_edges == features.shape[0] - 1

    print("Unit test passed")

test_one_patient_graph(graph, features, missing_mask)