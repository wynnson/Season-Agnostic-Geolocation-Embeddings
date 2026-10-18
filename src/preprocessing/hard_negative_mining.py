import pandas as pd

from scipy.spatial import KDTree

from src.preprocessing.schema import HardNegatives, Pairs
from src.preprocessing.spatial import (
    haversine_distance_m,
    lon_lat_to_unit_sphere,
)


def precompute_hard_negatives(
    train_pairs: pd.DataFrame, 
    k: int = 50,
    candidate_count: int = 100,
    exclusion_radius_m: float = 2000.0
):
    lon         = train_pairs[Pairs.LONGITUDE_SUMMER].to_numpy()
    lat         = train_pairs[Pairs.LATITUDE_SUMMER].to_numpy()
    loc_ids     = train_pairs[Pairs.LOC_ID].to_numpy()
    xyz         = lon_lat_to_unit_sphere(lon, lat)
    kd_tree     = KDTree(xyz)

    candidate_count = min(len(train_pairs), candidate_count)
    tree_distances, candidate_idx = kd_tree.query(xyz, k=candidate_count)

    rows = []
    for anchor_idx, nearest_neighbors_indices in enumerate(candidate_idx):
        distances_m = haversine_distance_m(
            lon1=lon[anchor_idx],
            lat1=lat[anchor_idx],
            lon2=lon[nearest_neighbors_indices],
            lat2=lat[nearest_neighbors_indices]
        )

        is_valid = distances_m > exclusion_radius_m
        valid_indices = nearest_neighbors_indices[is_valid][:k]
        valid_distances_m = distances_m[is_valid][:k]

        for negative_idx, distance_m in zip(valid_indices, valid_distances_m):
            rows.append(
                {
                    HardNegatives.ANCHOR_LOC_ID: loc_ids[anchor_idx],
                    HardNegatives.NEGATIVE_LOC_ID: loc_ids[negative_idx],
                    HardNegatives.DISTANCE_M: distance_m,
                }
            )
    
    return pd.DataFrame(rows)
