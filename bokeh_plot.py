import folium
import csv
import sys
import os

# ── Config ─────────────────────────────────────────────────────────────────────
# Pass CSV path as argument, or set default here
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "detections.csv"
OUTPUT_HTML = "map.html"
# ───────────────────────────────────────────────────────────────────────────────


def load_detections(csv_path):
    """
    Load detection and drone GPS points from the CSV produced by geolocalization.py.
    Returns two dicts keyed by person_idx, plus a flat drone list.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # person_idx → list of [lat, lon]
    person_tracks = {}
    drone_points = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_idx = int(row['person_idx'])
            p_lat = float(row['person_lat'])
            p_lon = float(row['person_lon'])
            d_lat = float(row['drone_lat'])
            d_lon = float(row['drone_lon'])
            ts    = float(row['timestamp_sec'])
            conf  = float(row['confidence'])

            if p_idx not in person_tracks:
                person_tracks[p_idx] = []
            person_tracks[p_idx].append({
                'lat': p_lat, 'lon': p_lon,
                'ts': ts, 'conf': conf
            })

            drone_points.append({'lat': d_lat, 'lon': d_lon, 'ts': ts})

    return person_tracks, drone_points


def build_map(person_tracks, drone_points):
    """Render a Folium map with per-person colour-coded tracks and drone path."""

    # Colour palette for up to 8 distinct persons
    PERSON_COLORS = ['red', 'blue', 'orange', 'purple',
                     'darkred', 'cadetblue', 'darkblue', 'darkgreen']

    all_latlons = (
        [[p['lat'], p['lon']] for pts in person_tracks.values() for p in pts]
        + [[d['lat'], d['lon']] for d in drone_points]
    )

    if not all_latlons:
        print("No data to plot.")
        return

    m = folium.Map(location=all_latlons[0], zoom_start=20, max_zoom=22)

    # ── Per-person tracks ──────────────────────────────────────────────────────
    for p_idx, pts in sorted(person_tracks.items()):
        color = PERSON_COLORS[(p_idx - 1) % len(PERSON_COLORS)]
        coords = [[p['lat'], p['lon']] for p in pts]

        folium.PolyLine(
            coords, color=color, weight=2, opacity=0.8,
            tooltip=f"Person {p_idx} path"
        ).add_to(m)

        for i, p in enumerate(pts):
            folium.CircleMarker(
                location=[p['lat'], p['lon']],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=(
                    f"<b>Person {p_idx}</b> (det #{i+1})<br>"
                    f"Lat: {p['lat']:.7f}<br>"
                    f"Lon: {p['lon']:.7f}<br>"
                    f"Conf: {p['conf']:.2f}<br>"
                    f"Time: {p['ts']:.2f}s"
                )
            ).add_to(m)

    # ── Drone path ─────────────────────────────────────────────────────────────
    if drone_points:
        drone_coords = [[d['lat'], d['lon']] for d in drone_points]
        folium.PolyLine(
            drone_coords, color='green', weight=3, opacity=1,
            tooltip="Drone path"
        ).add_to(m)

        for i, d in enumerate(drone_points):
            folium.CircleMarker(
                location=[d['lat'], d['lon']],
                radius=3,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.7,
                popup=(
                    f"<b>Drone</b> pos #{i+1}<br>"
                    f"Lat: {d['lat']:.7f}<br>"
                    f"Lon: {d['lon']:.7f}<br>"
                    f"Time: {d['ts']:.2f}s"
                )
            ).add_to(m)

    m.fit_bounds(all_latlons)
    return m


def main():
    print(f"Reading detections from: {CSV_PATH}")
    person_tracks, drone_points = load_detections(CSV_PATH)

    total_detections = sum(len(v) for v in person_tracks.values())
    print(f"Loaded {total_detections} detections across {len(person_tracks)} person(s).")
    print(f"Loaded {len(drone_points)} drone positions.")

    m = build_map(person_tracks, drone_points)
    if m:
        m.save(OUTPUT_HTML)
        print(f"Map saved → {OUTPUT_HTML}")


if __name__ == '__main__':
    main()