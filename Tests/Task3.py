from google.cloud import bigquery
import numpy
import pandas
from dataclasses import dataclass


#Connects to BigQuery project with mimic inside:
MIMIC = bigquery.Client(project="project-bb25f1c3-ff0a-47c9-9d0")

#Select a single ICU stay
query = """SELECT stay_id FROM `physionet-data.mimiciv_3_1_derived.icustay_detail` LIMIT 1; """
STAY_ID = MIMIC.query(query).to_dataframe()
stay_id_value = STAY_ID.iloc[0,0]

#return one stay from vital sign datatable, round some values.
query = f""" SELECT stay_id, TIMESTAMP_TRUNC(charttime, HOUR) AS time_bin, AVG(heart_rate) AS heart_rate, AVG(sbp) AS sbp, AVG(dbp) AS dbp, AVG(mbp) AS mbp, AVG(resp_rate) AS resp_rate, AVG(temperature) AS temperature, AVG(spo2) AS spo2, AVG(glucose) AS glucose
FROM `physionet-data.mimiciv_3_1_derived.vitalsign` WHERE stay_id = {stay_id_value}
GROUP BY stay_id, time_bin
ORDER BY time_bin;
"""

SingleICU = MIMIC.query(query).to_dataframe()

print("================= hourly feature =================")
print(SingleICU.head(10))
print(f"shape: {SingleICU.shape}")
print(f"cols: {list(SingleICU.columns)}")

#Convert table model into arrays:

cols = [
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "resp_rate",
    "temperature",
    "spo2",
    "glucose"
]
#REPLACE

DataFrafeFeature = SingleICU[cols].copy()

# if value is not None -> 1 else -> 0
MissingMask = DataFrafeFeature.notna().astype(float).to_numpy()


featues = DataFrafeFeature.fillna(0).astype(numpy.float32).to_numpy()

print("=============== MASKS ===================")
print("features shape:", featues.shape)
print("missing_mask shape:", MissingMask.shape)
print("first feature row:", featues[0])
print("first mask row:", MissingMask[0])


@dataclass
class Graph:
    x: numpy.ndarray
    edge_index: numpy.ndarray
    edge_type: numpy.ndarray

class GraphBuilder:
    def __init__(self, edge_make="both", clinical_groups=None):
        self.edge_make = edge_make
        self.clinical_groups = clinical_groups

    def BuildGraph(self, features, missing_mask):
        #node feature == value plus missing masks
        x = numpy.concatenate([features, missing_mask], axis=1).astype(numpy.float32)
        edges = []
        edge_types = []

        #build temporal edges:
        if self.edge_make in {"temporal_only", "both"}:
            for t in range(len(features) -1):
                edges.append((t, t+1))
                edge_types.append(0)
        
        if self.edge_make in {"clinical_only", "both"}:
            for t in range(len(features) -1):
                for _, idxs in self.clinical_groups.items():
                    if missing_mask[t, idxs].sum() > 0 and missing_mask[t + 1, idxs].sum() > 0:
                        edges.append((t, t + 1))
                        edge_types.append(1)
                        break
        edge_index = numpy.array(edges, dtype=numpy.int64).T if edges else numpy.empty((2, 0), dtype=numpy.int64)
        edge_type = numpy.array(edge_types, dtype=numpy.int64) if edge_types else numpy.empty((0,), dtype=numpy.int64)

        return Graph(x=x, edge_index=edge_index, edge_type=edge_type)

clinical_groups = {"vitals": list(range(8))}

builder = GraphBuilder(edge_make="both", clinical_groups=clinical_groups)
graph = builder.BuildGraph(featues, MissingMask)

print("========================= GRAPHING =========================")
print("x shape:", graph.x.shape)
print("edge_index shape:", graph.edge_index.shape)
print("edge_type shape:", graph.edge_type.shape)
print("first few edges:", graph.edge_index[:, :10] if graph.edge_index.shape[1] > 0 else graph.edge_index)

print("========================= SUMMARY =========================")
print("num time bins:", featues.shape[0])
print("num raw features:", featues.shape[1])
print("node feature dim:", graph.x.shape[1])
print("num edges:", graph.edge_index.shape[1])

num_temporal = int((graph.edge_type == 0).sum())
num_clinical = int((graph.edge_type == 1).sum())

print("temporal edges:", num_temporal)
print("clinical edges:", num_clinical)

def test_one_patient_graph(graph, featues, missing_mask):
    assert featues.ndim == 2
    assert missing_mask.ndim == 2
    assert featues.shape == missing_mask.shape


    assert graph.x.shape[0] == featues.shape[0]
    assert graph.x.shape[1] == featues.shape[1] * 2

    # shape: [2, E]
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == graph.edge_type.shape[0]

    # for 14 nodes mean that edges should be 13
    temporal_edges = int((graph.edge_type == 0).sum())
    assert temporal_edges == featues.shape[0] - 1

    print("Unit test passed :)")

test_one_patient_graph(graph, featues, MissingMask)