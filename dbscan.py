import csv
import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import folium

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH    = sys.argv[1] if len(sys.argv) > 1 else "detections.csv"
OUTPUT_HTML = "clusters_map.html"
EPS_METERS  = 1.5   # DBSCAN radius — tighter = stricter cluster membership
MIN_SAMPLES = 8     # higher = noisier points can't form a cluster
IQR_FACTOR  = 1.5   # outlier pre-filter aggressiveness (1.5=standard, lower=stricter)
# ───────────────────────────────────────────────────────────────────────────────


def load_csv(csv_path):
    """Load all detection and drone rows from the CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    detections = []   # (lat, lon) for every person detection
    drone_pts  = []   # (lat, lon, timestamp) for drone path

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            detections.append((float(row['person_lat']), float(row['person_lon'])))
            drone_pts.append((
                float(row['drone_lat']),
                float(row['drone_lon']),
                float(row['timestamp_sec']),
            ))

    return detections, drone_pts


def compute_medoid(cluster_points):
    """Return the point in the cluster that minimises sum of distances to all others."""
    min_total_dist = float('inf')
    medoid = cluster_points[0]
    for candidate in cluster_points:
        total = sum(geodesic(candidate, other).meters for other in cluster_points)
        if total < min_total_dist:
            min_total_dist = total
            medoid = candidate
    return medoid


def filter_outliers_iqr(detections):
    """
    Remove points that are outliers in either lat or lon using IQR.
    This strips the long-range scattered points before DBSCAN runs,
    so they can't inflate or pollute genuine clusters.
    """
    points = np.array(detections)
    mask   = np.ones(len(points), dtype=bool)

    for axis in [0, 1]:  # 0=lat, 1=lon
        col  = points[:, axis]
        q1   = np.percentile(col, 25)
        q3   = np.percentile(col, 75)
        iqr  = q3 - q1
        lo   = q1 - IQR_FACTOR * iqr
        hi   = q3 + IQR_FACTOR * iqr
        mask &= (col >= lo) & (col <= hi)

    n_removed = int(np.sum(~mask))
    filtered  = [tuple(p) for p in points[mask]]
    return filtered, n_removed


def run_dbscan(detections):
    """Run DBSCAN and return per-cluster info including medoid."""
    points = np.array(detections)
    eps_deg = EPS_METERS / 111000.0  # rough degree conversion

    db = DBSCAN(eps=eps_deg, min_samples=MIN_SAMPLES).fit(points)
    labels = db.labels_

    clusters = []
    for cluster_id in sorted(set(labels) - {-1}):
        mask = labels == cluster_id
        cluster_pts = [tuple(p) for p in points[mask]]

        mean_lat = float(np.mean(points[mask, 0]))
        mean_lon = float(np.mean(points[mask, 1]))
        medoid   = compute_medoid(cluster_pts)

        max_dist = 0.0
        for i in range(len(cluster_pts)):
            for j in range(i + 1, len(cluster_pts)):
                d = geodesic(cluster_pts[i], cluster_pts[j]).meters
                max_dist = max(max_dist, d)

        clusters.append({
            'id':       cluster_id,
            'points':   cluster_pts,
            'mean':     (mean_lat, mean_lon),
            'medoid':   medoid,
            'max_dist': max_dist,
            'n':        len(cluster_pts),
        })

    noise_count = int(np.sum(labels == -1))
    return clusters, noise_count


def build_map(clusters, drone_pts):
    """Build a Folium map showing drone trajectory and cluster medoids."""
    CLUSTER_COLORS = ['red', 'blue', 'orange', 'purple',
                      'darkred', 'cadetblue', 'darkblue', 'pink']

    # Gather all coords for map bounds
    all_latlons = [(d[0], d[1]) for d in drone_pts]
    for c in clusters:
        all_latlons.append(c['medoid'])

    center = all_latlons[0] if all_latlons else (0, 0)
    m = folium.Map(location=center, zoom_start=20, max_zoom=22)

    # ── Drone trajectory ───────────────────────────────────────────────────────
    # Deduplicate consecutive identical drone positions to keep polyline clean
    drone_coords = []
    prev = None
    for lat, lon, ts in sorted(drone_pts, key=lambda x: x[2]):
        pt = (lat, lon)
        if pt != prev:
            drone_coords.append(pt)
            prev = pt

    if drone_coords:
        folium.PolyLine(
            drone_coords, color='green', weight=3,
            opacity=0.9, tooltip='Drone trajectory'
        ).add_to(m)

        # Start marker
        folium.Marker(
            location=drone_coords[0],
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
            popup='Drone start'
        ).add_to(m)

        # End marker
        folium.Marker(
            location=drone_coords[-1],
            icon=folium.Icon(color='green', icon='stop', prefix='fa'),
            popup='Drone end'
        ).add_to(m)

    # ── Cluster medoids ────────────────────────────────────────────────────────
    for c in clusters:
        color  = CLUSTER_COLORS[c['id'] % len(CLUSTER_COLORS)]
        medoid = c['medoid']

        # Large circle showing cluster spread
        folium.Circle(
            location=medoid,
            radius=c['max_dist'] / 2,   # radius = half the max intra-cluster distance
            color=color,
            fill=True,
            fill_opacity=0.15,
            tooltip=f"Cluster {c['id']+1} spread ({c['max_dist']:.2f}m diameter)"
        ).add_to(m)

        # Medoid marker
        folium.Marker(
            location=medoid,
            icon=folium.Icon(color=color, icon='user', prefix='fa'),
            popup=(
                f"<b>Person cluster {c['id']+1}</b><br>"
                f"Detections: {c['n']}<br>"
                f"Medoid: {medoid[0]:.7f}, {medoid[1]:.7f}<br>"
                f"Mean:   {c['mean'][0]:.7f}, {c['mean'][1]:.7f}<br>"
                f"Max spread: {c['max_dist']:.3f} m"
            )
        ).add_to(m)

    m.fit_bounds([[min(p[0] for p in all_latlons), min(p[1] for p in all_latlons)],
                  [max(p[0] for p in all_latlons), max(p[1] for p in all_latlons)]])
    return m


def main():
    print(f"Reading from: {CSV_PATH}")
    detections, drone_pts = load_csv(CSV_PATH)
    print(f"Loaded {len(detections)} detections, {len(drone_pts)} drone positions.")

    filtered, n_removed = filter_outliers_iqr(detections)
    print(f"  IQR pre-filter removed {n_removed} outlier point(s) → {len(filtered)} remain.")

    clusters, noise = run_dbscan(filtered)

    print(f"\nDBSCAN results (eps={EPS_METERS}m, min_samples={MIN_SAMPLES}):")
    print(f"  Clusters found : {len(clusters)}")
    print(f"  Noise points   : {noise}")

    for c in clusters:
        print(f"\n  Cluster {c['id']+1}  ({c['n']} detections)")
        print(f"    Medoid : Lat={c['medoid'][0]:.8f}, Lon={c['medoid'][1]:.8f}")
        print(f"    Mean   : Lat={c['mean'][0]:.8f},  Lon={c['mean'][1]:.8f}")
        print(f"    Spread : {c['max_dist']:.3f} m (max intra-cluster distance)")

    m = build_map(clusters, drone_pts)
    m.save(OUTPUT_HTML)
    print(f"\nMap saved → {OUTPUT_HTML}")


if __name__ == '__main__':
    main()