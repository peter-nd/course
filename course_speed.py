import gpxpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from matplotlib import cm
import pandas as pd
from io import StringIO
import math


import gpxpy
import pandas as pd
import numpy as np
import math
from io import StringIO


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def calculate_initial_bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(delta_lon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def parse_gpx(file_path):
    with open(file_path, 'r') as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.time, point.latitude, point.longitude, point.elevation))

    df = pd.DataFrame(points, columns=["time", "lat", "lon", "ele"])
    df['distance'] = [0] + [haversine(df['lat'][i-1], df['lon'][i-1], df['lat'][i], df['lon'][i]) for i in range(1, len(df))]
    df['cum_dist'] = df['distance'].cumsum()
    df['ele_diff'] = df['ele'].diff().fillna(0)
    df['gradient'] = (df['ele_diff'] / df['distance'].replace(0, np.nan)).clip(-1, 1).fillna(0)
    df['bearing'] = [0] + [calculate_initial_bearing(df['lat'][i-1], df['lon'][i-1], df['lat'][i], df['lon'][i]) for i in range(1, len(df))]
    df['bearing'] = df['bearing'].ffill()  # ✅ modern equivalent

    return df



def segment_route_slope_then_wind(df, wind_direction_deg, grad_thresh=0.01, min_segment_length=10, wind_speed=None):
    df = df.copy()

    df['slope_type'] = pd.cut(df['gradient'],
                          bins=[-np.inf, -grad_thresh, grad_thresh, np.inf],
                          labels=['downhill', 'flat', 'uphill'])

    df['slope_type'] = df['slope_type'].astype('category')

    codes = df['slope_type'].cat.codes

    def rolling_mode(series):
        modes = series.mode()
        return modes.iloc[0] if not modes.empty else series.iloc[0]

    codes_smooth = codes.rolling(window=5, center=True, min_periods=1).apply(rolling_mode, raw=False)

    df['slope_type_smooth'] = pd.Categorical.from_codes(
        codes_smooth.round().astype(int),
        categories=df['slope_type'].cat.categories
    )

    df['slope_segment_id'] = (df['slope_type_smooth'] != df['slope_type_smooth'].shift()).cumsum()

    if wind_speed == 0:
        df['wind_type'] = 'no wind'  # changed label here
    else:
        df['wind_rel_angle'] = (df['bearing'] - wind_direction_deg + 360) % 360

        def wind_type_from_angle(angle):
            if (angle <= 60) or (angle >= 300):
                return 'headwind'
            elif (120 <= angle <= 240):
                return 'tailwind'
            else:
                return 'crosswind'

        df['wind_type'] = df['wind_rel_angle'].apply(wind_type_from_angle)

    segments = []
    segment_id = 0

    for slope_id, group in df.groupby('slope_segment_id'):
        major_wind_type = group['wind_type'].mode()[0]

        group = group.copy()
        if wind_speed == 0:
            # combined label is just slope type with 'no wind' (no arrow)
            group['combined_label'] = group['slope_type_smooth'].iloc[0]
        else:
            group['combined_label'] = major_wind_type + " | " + group['slope_type_smooth'].iloc[0]

        group['wind_segment_id'] = (group['combined_label'] != group['combined_label'].shift()).cumsum()

        for wind_seg_id, wind_group in group.groupby('wind_segment_id'):
            segment_length = wind_group['cum_dist'].iloc[-1] - wind_group['cum_dist'].iloc[0]

            segments.append({
                'segment_id': segment_id,
                'segment_label': wind_group['combined_label'].iloc[0],
                'start_idx': wind_group.index[0],
                'end_idx': wind_group.index[-1],
                'start_dist': wind_group['cum_dist'].iloc[0],
                'end_dist': wind_group['cum_dist'].iloc[-1],
                'segment_length': segment_length
            })
            segment_id += 1

    segment_summary = pd.DataFrame(segments)

    df['final_segment_id'] = -1
    df['final_segment_label'] = ''

    for _, row in segment_summary.iterrows():
        df.loc[row['start_idx']:row['end_idx'], 'final_segment_id'] = row['segment_id']
        df.loc[row['start_idx']:row['end_idx'], 'final_segment_label'] = row['segment_label']

    return df, segment_summary


def plot_route_2d_with_segments(lat, lon, segments, wind_dir_deg=None, wind_speed=None, arrow_length=200):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('2D Route Colored by Segment')

    x, y = latlon_to_xy_meters(lat, lon)  # convert lat/lon to meters (East, North)

    points = np.array([x, y]).T
    line_segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

    unique_segments = np.unique(segments)
    segment_to_color = {seg: i for i, seg in enumerate(unique_segments)}
    segment_indices = np.array([segment_to_color[s] for s in segments[:-1]])

    cmap = cm.get_cmap('tab20', len(unique_segments))
    colors = cmap(segment_indices)

    for i in range(len(line_segments)):
        seg = line_segments[i]
        ax.plot(seg[:, 0], seg[:, 1], color=colors[i], linewidth=2)

    handles = [plt.Line2D([0], [0], color=cmap(i), lw=3) for i in range(len(unique_segments))]
    ax.legend(handles, unique_segments, title='Segments', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Only draw arrow if wind_dir_deg is set and wind_speed is not zero or None
    if wind_dir_deg is not None and wind_speed not in [0, 0.0]:
        wind_rad = np.deg2rad(wind_dir_deg)  # 0 = N, 90 = E, etc.

        margin = 100  # meters offset from min x/y
        start_x = x.min() + margin
        start_y = y.max() - margin

        dx = -arrow_length * np.sin(wind_rad)
        dy = -arrow_length * np.cos(wind_rad)

        ax.arrow(start_x, start_y, dx, dy, head_width=30, head_length=40, fc='red', ec='red', linewidth=2)
        ax.text(start_x, start_y - 30, f'Wind from {wind_dir_deg}°', color='red', fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()



def latlon_to_xy_meters(lat, lon):
    """Convert lat/lon to x/y meters using equirectangular projection centered at mean latitude."""
    R = 6371000  # Earth radius in meters
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    mean_lat_rad = np.mean(lat_rad)
    x = R * (lon_rad - lon_rad[0]) * np.cos(mean_lat_rad)  # meters east-west
    y = R * (lat_rad - lat_rad[0])  # meters north-south
    return x, y

def plot_route_3d_with_speed(lat, lon, ele, speeds):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Route Colored by Speed')

    x, y = latlon_to_xy_meters(lat, lon)
    points = np.array([x, y, ele]).T   # X=East, Y=North
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

    speeds_norm = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-9)
    cmap = cm.get_cmap('viridis')
    colors = cmap(speeds_norm)

    for i in range(len(segments)):
        seg = segments[i]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colors[i])

    # Equal scaling for East and North axes
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    max_range = max(x_range, y_range)
    x_mid = np.mean(xlim)
    y_mid = np.mean(ylim)
    ax.set_xlim3d(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim3d(y_mid - max_range / 2, y_mid + max_range / 2)

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(speeds)
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label('Speed (m/s)')

    plt.show()

def plot_route_2d_with_speed(lat, lon, speeds):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('2D Route Colored by Speed')

    x, y = latlon_to_xy_meters(lat, lon)
    points = np.array([x, y]).T   # X=East, Y=North
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

    speeds_norm = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-9)
    cmap = cm.get_cmap('viridis')
    colors = cmap(speeds_norm)

    for i in range(len(segments)):
        seg = segments[i]
        ax.plot(seg[:, 0], seg[:, 1], color=colors[i])

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(speeds)
    cbar = plt.colorbar(mappable, ax=ax, pad=0.02)
    cbar.set_label('Speed (m/s)')

    ax.set_aspect('equal')  # equal scaling on x and y axes (meters)
    plt.show()



def analyze_gpx_power_with_energy(
    gpx_path,
    power_by_gradient,
    mass_kg=88,
    cda=0.36,
    c_rr=0.003,
    wind_speed=0.0,
    wind_direction=0.0,   # degrees, 0 = North, clockwise
    air_density=1.225,
    g=9.81
):
    """Uses constant power on different grades"""
    # Load GPX data
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.latitude, point.longitude, point.elevation))

    points = np.array(points)
    lat, lon, ele_raw = points[:, 0], points[:, 1], points[:, 2]
    ele = gaussian_filter1d(ele_raw, sigma=3)
    elevation_gain = np.sum(np.maximum(np.diff(ele), 0))

    def latlon_to_xy(lat, lon):
        R = 6371000  # Earth radius in meters
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = R * lon_rad * np.cos(np.mean(lat_rad))
        y = R * lat_rad
        return x, y

    x, y = latlon_to_xy(lat, lon)

    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(ele)

    segment_distances = np.sqrt(dx**2 + dy**2 + dz**2)
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    slope = np.arctan2(dz, horizontal_dist)

    # Rider heading in radians (0 = North)
    rider_heading_rad = (np.pi/2 - np.arctan2(dy, dx)) % (2*np.pi)
    wind_dir_rad = np.radians(wind_direction) % (2*np.pi)
    relative_angle = wind_dir_rad - rider_heading_rad
    effective_wind_speed = wind_speed * np.cos(relative_angle)

    def get_power_for_slope(s):
        for (low, high), pwr in power_by_gradient.items():
            if low <= s < high:
                return pwr
        return 0

    powers = np.array([get_power_for_slope(s) for s in slope])

    # Initialize values
    v_prev = 2.0  # m/s starting speed
    h_prev = ele[0]
    E_prev = 0.5 * mass_kg * v_prev**2 + mass_kg * g * h_prev

    speeds = []
    segment_times = []

    min_speed = 0.5  # Minimum speed to avoid instability
    max_dt = 10.0    # Max time step in seconds

    for i in range(len(segment_distances)):
        dist = segment_distances[i]
        slope_i = slope[i]
        h_new = ele[i + 1]
        power = powers[i]
        wind_i = effective_wind_speed[i]

        v_prev = max(v_prev, min_speed)  # clamp to minimum speed
        dt = dist / v_prev
        dt = min(dt, max_dt)  # cap time step

        # Energy added and lost
        E_add = power * dt
        F_aero = 0.5 * air_density * cda * (v_prev + wind_i)**2
        E_aero = F_aero * dist
        F_roll = mass_kg * g * np.cos(slope_i) * c_rr
        E_roll = F_roll * dist

        # New mechanical energy
        E_mech_new = E_prev + E_add - E_aero - E_roll
        E_pot_new = mass_kg * g * h_new
        E_kin_new = E_mech_new - E_pot_new

        if E_kin_new < 0:
            v_new = min_speed  # avoid zero speed
            E_kin_new = 0.0
            E_mech_new = E_pot_new
        else:
            v_new = np.sqrt(2 * E_kin_new / mass_kg)

        speeds.append(v_new)
        segment_times.append(dist / v_new)

        v_prev = v_new
        E_prev = E_mech_new

    speeds = np.array(speeds)
    segment_times = np.array(segment_times)

    cumulative_distance = np.insert(np.cumsum(segment_distances), 0, 0)
    cumulative_time = np.insert(np.cumsum(segment_times), 0, 0)

    total_distance = cumulative_distance[-1]
    total_time = cumulative_time[-1]
    average_speed_m_s = total_distance / total_time
    average_speed_kmh = average_speed_m_s * 3.6

    average_power = np.sum(powers * segment_times) / np.sum(segment_times)
    normalized_power = (np.sum((powers ** 4) * segment_times) / np.sum(segment_times)) ** 0.25

    # Plot elevation and speed
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Elevation (m)', color=color)
    ax1.plot(cumulative_distance / 1000, ele, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(cumulative_distance / 1000, ele_raw, color='gray', alpha=0.3, label='Raw Elevation')
    ax1.plot(cumulative_distance / 1000, ele, color='blue', label='Smoothed Elevation')
    ax1.legend(loc='upper left')
    ax1.text(0.01, 0.05,
            f'Total Elevation Gain: {elevation_gain:.0f} m',
            transform=ax1.transAxes,
            fontsize=9,
            horizontalalignment='left',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Speed (kph)', color=color)
    ax2.plot(cumulative_distance[1:] / 1000, speeds * 3.6, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Elevation and Estimated Speed vs Distance (Energy-based, No Smoothing)')
    plt.show()

    # Plot route functions assumed defined elsewhere
    plot_route_2d_with_speed(lat, lon, speeds)
    plot_route_3d_with_speed(lat, lon, ele, speeds)

    return {
        "average_speed_m_s": average_speed_m_s,
        "average_speed_kmh": average_speed_kmh,
        "average_power": average_power,
        "normalized_power": normalized_power,
        "cumulative_distance": cumulative_distance,
        "elevation": ele,
        "speeds": speeds,
    }

def analyze_gpx_power_with_energy_segments(
    gpx_path,
    power_per_segment_id=None,  # dict {segment_id: power_watts}
    segment_ids=None,           # array aligned with points (len N), segments = N-1
    default_power=250,          # fallback power if segment_id missing
    mass_kg=88,
    cda=0.36,
    c_rr=0.003,
    wind_speed=0.0,
    wind_direction=0.0,   # degrees, 0 = North, clockwise
    air_density=1.225,
    g=9.81
):
    """Applies a constant power on per segment basis"""


    # Load GPX data
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.latitude, point.longitude, point.elevation))

    points = np.array(points)
    lat, lon, ele_raw = points[:, 0], points[:, 1], points[:, 2]
    ele = gaussian_filter1d(ele_raw, sigma=3)
    elevation_gain = np.sum(np.maximum(np.diff(ele), 0))

    def latlon_to_xy(lat, lon):
        R = 6371000  # Earth radius in meters
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = R * lon_rad * np.cos(np.mean(lat_rad))
        y = R * lat_rad
        return x, y

    x, y = latlon_to_xy(lat, lon)

    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(ele)

    segment_distances = np.sqrt(dx**2 + dy**2 + dz**2)
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    slope = np.arctan2(dz, horizontal_dist)

    # Rider heading in radians (0 = North)
    rider_heading_rad = (np.pi/2 - np.arctan2(dy, dx)) % (2*np.pi)
    wind_dir_rad = np.radians(wind_direction) % (2*np.pi)
    relative_angle = wind_dir_rad - rider_heading_rad
    effective_wind_speed = wind_speed * np.cos(relative_angle)

    # Build powers array: one constant power per segment
    if power_per_segment_id is not None and segment_ids is not None:
        # segment_ids aligned with points (len N), so segments = N-1
        powers = np.array([
            power_per_segment_id.get(sid, default_power) for sid in segment_ids[:-1]
        ])
    else:
        # fallback constant power for all segments
        powers = np.full(len(segment_distances), default_power)
        print("Warning! Error in program falling back to default power.")

    # Initialize values
    v_prev = 2.0  # m/s starting speed
    h_prev = ele[0]
    E_prev = 0.5 * mass_kg * v_prev**2 + mass_kg * g * h_prev

    speeds = []
    segment_times = []

    min_speed = 0.5  # Minimum speed to avoid instability
    max_dt = 10.0    # Max time step in seconds

    for i in range(len(segment_distances)):
        dist = segment_distances[i]
        slope_i = slope[i]
        h_new = ele[i + 1]
        power = powers[i]
        wind_i = effective_wind_speed[i]

        v_prev = max(v_prev, min_speed)  # clamp to minimum speed
        dt = dist / v_prev
        dt = min(dt, max_dt)  # cap time step

        # Energy added and lost
        E_add = power * dt
        F_aero = 0.5 * air_density * cda * (v_prev + wind_i)**2
        E_aero = F_aero * dist
        F_roll = mass_kg * g * np.cos(slope_i) * c_rr
        E_roll = F_roll * dist

        # New mechanical energy
        E_mech_new = E_prev + E_add - E_aero - E_roll
        E_pot_new = mass_kg * g * h_new
        E_kin_new = E_mech_new - E_pot_new

        if E_kin_new < 0:
            v_new = min_speed  # avoid zero speed
            E_kin_new = 0.0
            E_mech_new = E_pot_new
        else:
            v_new = np.sqrt(2 * E_kin_new / mass_kg)

        speeds.append(v_new)
        segment_times.append(dist / v_new)

        v_prev = v_new
        E_prev = E_mech_new

    speeds = np.array(speeds)
    segment_times = np.array(segment_times)

    cumulative_distance = np.insert(np.cumsum(segment_distances), 0, 0)
    cumulative_time = np.insert(np.cumsum(segment_times), 0, 0)

    total_distance = cumulative_distance[-1]
    total_time = cumulative_time[-1]
    average_speed_m_s = total_distance / total_time
    average_speed_kmh = average_speed_m_s * 3.6

    average_power = np.sum(powers * segment_times) / np.sum(segment_times)
    normalized_power = (np.sum((powers ** 4) * segment_times) / np.sum(segment_times)) ** 0.25
    """
    # Plot elevation and speed
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Elevation (m)', color=color)
    ax1.plot(cumulative_distance / 1000, ele, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(cumulative_distance / 1000, ele_raw, color='gray', alpha=0.3, label='Raw Elevation')
    ax1.plot(cumulative_distance / 1000, ele, color='blue', label='Smoothed Elevation')
    ax1.legend(loc='upper left')
    ax1.text(0.01, 0.05,
            f'Total Elevation Gain: {elevation_gain:.0f} m',
            transform=ax1.transAxes,
            fontsize=9,
            horizontalalignment='left',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Speed (kph)', color=color)
    ax2.plot(cumulative_distance[1:] / 1000, speeds * 3.6, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Elevation and Estimated Speed vs Distance (Energy-based, No Smoothing)')
    plt.show()

    # Plot route functions assumed defined elsewhere
    plot_route_2d_with_speed(lat, lon, speeds)
    plot_route_3d_with_speed(lat, lon, ele, speeds)
    """
    return {
        "average_speed_m_s": average_speed_m_s,
        "average_speed_kmh": average_speed_kmh,
        "average_power": average_power,
        "normalized_power": normalized_power,
        "cumulative_distance": cumulative_distance,
        "elevation": ele,
        "speeds": speeds,
    }


def plot_power_and_elevation_by_segment(cum_dist, elevation, segments, powers, segment_labels_list):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.lines import Line2D

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Elevation (m)', color='tab:blue')

    seg_ids = np.array(segments)
    unique_seg_ids = np.unique(seg_ids)

    # Map segment ID to label
    seg_id_to_label = {seg_id: segment_labels_list[seg_id] for seg_id in unique_seg_ids}

    # Unique labels and color map for elevation segments
    unique_labels = list(sorted(set(seg_id_to_label.values())))
    cmap = colormaps['tab20']
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

    # Plot elevation colored by segment label
    for seg_id in unique_seg_ids:
        idx = np.where(seg_ids == seg_id)[0]
        start_idx, end_idx = idx[0], idx[-1] + 1 if idx[-1] + 1 < len(cum_dist) else idx[-1]
        x = cum_dist[start_idx:end_idx] / 1000
        y = elevation[start_idx:end_idx]
        label = seg_id_to_label[seg_id]
        color = label_to_color[label]
        ax1.plot(x, y, color=color)

    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot power on secondary y-axis, single color
    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (W)', color='tab:red')

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Enable grid on primary axis
    ax2.grid(True)

    prev_end = None
    prev_power = None
    power_color = 'tab:red'

    for seg_id in unique_seg_ids:
        idx = np.where(seg_ids == seg_id)[0]
        start = cum_dist[idx[0]]
        end = cum_dist[idx[-1] + 1] if idx[-1] + 1 < len(cum_dist) else cum_dist[-1]
        mid_power = powers[seg_id]

        ax2.hlines(mid_power, start / 1000, end / 1000, colors=power_color, linewidth=2)

        if prev_end is not None and prev_power != mid_power:
            ax2.vlines(start / 1000, prev_power, mid_power, colors=power_color, linewidth=1)

        prev_end = end
        prev_power = mid_power

    ax2.tick_params(axis='y', labelcolor=power_color)

    # Legend for elevation segment labels only
    legend_lines = [Line2D([0], [0], color=label_to_color[label], lw=4) for label in unique_labels]
    ax1.legend(
        legend_lines,
        unique_labels,
        title="Elevation Segment Labels",
        loc='center left',
        bbox_to_anchor=(1.1, 0.5),
        fontsize='small',
        title_fontsize='small',
        markerscale=0.7,
        handlelength=1.5,
        borderpad=0.3
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # leave space on right for legend



    plt.title("Elevation and Segment-Based Power vs Distance")
    fig.tight_layout()
    plt.show()



def optimize_segment_powers(
    gpx_path,
    segment_ids,
    fixed_avg_power,
    VI,
    max_power,
    mass_kg=88,
    cda=0.36,
    c_rr=0.003,
    wind_speed=0.0,
    wind_direction=0.0,
    air_density=1.225,
    g=9.81
):
  

    unique_segments = np.unique(segment_ids)
    # Initial guess: all powers = fixed_avg_power
    P0 = np.full(len(unique_segments), fixed_avg_power)# + np.random.uniform(-20, 20, len(unique_segments))
    

    def objective(P):
        # objective: minimize total time = sum segment_times
        # We use your existing function, modified to accept per segment powers as list/dict
        
        power_dict = {sid: power for sid, power in zip(unique_segments, P)}
        
        results = analyze_gpx_power_with_energy_segments(
            gpx_path,
            power_per_segment_id=power_dict,
            segment_ids=segment_ids,  # We'll modify analyze_gpx_power to accept segment lengths for timing
            default_power=fixed_avg_power,
            mass_kg=mass_kg,
            cda=cda,
            c_rr=c_rr,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            air_density=air_density,
            g=g
        )
        
        # total time can be extracted:
        total_time = results['cumulative_distance'][-1] / results['average_speed_m_s']
        
        
        # your analyze_gpx_power_with_energy_segments returns total_time? If not, you can add it.
        # If not available, compute total_time from speeds and distances here.
       
        
        print("Total time")
        print(total_time)
        #print(results['average_speed_m_s'])
        return total_time
        
    def avg_power_constraint(P):
        # Compute weighted average power over all segments = fixed_avg_power
        # Weighted by time spent per segment (segment_length / speed)
        
        power_dict = {sid: power for sid, power in zip(unique_segments, P)}
        
        results = analyze_gpx_power_with_energy_segments(
            gpx_path,
            power_per_segment_id=power_dict,
            segment_ids=segment_ids,  # We'll modify analyze_gpx_power to accept segment lengths for timing
            default_power=fixed_avg_power,
            mass_kg=mass_kg,
            cda=cda,
            c_rr=c_rr,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            air_density=air_density,
            g=g
        )
        avg_power = results['average_power']
        return avg_power - fixed_avg_power

    def norm_power_constraint(P):
        power_dict = {sid: power for sid, power in zip(unique_segments, P)}
        
        results = analyze_gpx_power_with_energy_segments(
            gpx_path,
            power_per_segment_id=power_dict,
            segment_ids=segment_ids,  # We'll modify analyze_gpx_power to accept segment lengths for timing
            default_power=fixed_avg_power,
            mass_kg=mass_kg,
            cda=cda,
            c_rr=c_rr,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            air_density=air_density,
            g=g
        )
        npow = results['normalized_power']
        avg_power = results['average_power']
        return VI * avg_power - npow  # inequality constraint: ≥ 0

    # Constraints definition for scipy
    constraints = [
        {'type': 'eq', 'fun': avg_power_constraint},  # avg power == fixed
        {'type': 'ineq', 'fun': norm_power_constraint}  # normalized_power ≤ VI * avg_power
    ]


    # Bounds for power per segment
    bounds = [(0, max_power) for _ in unique_segments]


    try:
        res = minimize(
            objective,
            P0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 30}  # increased maxiter
        )
    except Exception as e:
        print(f"Optimization raised an exception: {e}")
        res = None

    if res is not None:
        print(f"Optimizer result message: {res.message}")
        if res.success:
            optimized_powers = res.x
            print(f"Optimization successful. Avg speed: {-res.fun:.2f}")
        else:
            print("Optimization failed but best solution available.")
            optimized_powers = res.x  # still assign best guess
            print(f"Best found solution: {optimized_powers}")
    else:
        optimized_powers = None
        print("No optimization result available.")

    time_saved = objective(P0) - objective(optimized_powers)

    return optimized_powers, time_saved



# Example usage
if __name__ == "__main__":
    gpx_path = "ikke_projekt/krakaer_runden.gpx"  # replace with your GPX file path
    wind_dir = 90  # degrees, from NW
    wind_speed = 5
    df = parse_gpx(gpx_path)
    df, segments = segment_route_slope_then_wind(df, wind_dir, wind_speed=wind_speed)

    plot_route_2d_with_segments(df['lat'].values, df['lon'].values, df['final_segment_label'].values, wind_dir_deg=wind_dir, wind_speed=wind_speed)

    # Example segment IDs per point (length N)
    segment_ids = df['final_segment_id'].values  # or any array matching your GPX points

    unique_segments = np.unique(segment_ids)


    power_per_segment_id = {seg: np.random.uniform(200, 400) for seg in unique_segments}
    segment_labels_list = segments['segment_label'].tolist()

    results = analyze_gpx_power_with_energy_segments(
        gpx_path,
        power_per_segment_id=power_per_segment_id,
        segment_ids=segment_ids,
        default_power=220
    )

    print(f"Average speed: {results['average_speed_m_s']:.2f} m/s ({results['average_speed_kmh']:.2f} km/h)")
    print(f"Average power: {results['average_power']:.2f} W")
    print(f"Normalized power: {results['normalized_power']:.2f} W")


    plot_power_and_elevation_by_segment(
        cum_dist=results["cumulative_distance"],
        elevation=results["elevation"],
        segments=segment_ids,
        powers=power_per_segment_id,
        segment_labels_list=segment_labels_list
    )


    plt.switch_backend('Agg')

    # Make sure segments_df has 'segment_id' and 'segment_length' columns only (others ignored here)
    segments_df = segments[['segment_id', 'segment_length']]
    fixed_avg_power = 360  # your target average power
    VI = 1.1              # variability index (normalized power <= VI * avg power)
    max_power = 480        # max allowed power per segment (watts)
    cda = 0.24
    c_rr = 0.003
    wind_direction = 90
    wind_speed = 5
    mass_kg = 88

    optimized_powers, time_saved = optimize_segment_powers(
        gpx_path=gpx_path,
        segment_ids=segment_ids,
        fixed_avg_power=fixed_avg_power,
        VI=VI,
        max_power=max_power,
        mass_kg=88,
        cda=0.24,
        c_rr=0.003,
        wind_speed=5,
        wind_direction=90,
        air_density=1.225,
        g=9.81
    )


    power_per_segment_id = {seg: power for seg, power in zip(unique_segments, optimized_powers)}

    plt.switch_backend('TkAgg')
    print("Optimized powers per segment:", optimized_powers)

    plot_power_and_elevation_by_segment(
        cum_dist=results["cumulative_distance"],
        elevation=results["elevation"],
        segments=segment_ids,
        powers=power_per_segment_id,
        segment_labels_list=segment_labels_list
    )

    results = analyze_gpx_power_with_energy_segments(
        gpx_path,
        power_per_segment_id=power_per_segment_id,
        segment_ids=segment_ids,
        default_power=220,
        c_rr=c_rr,
        cda=cda,
        mass_kg=mass_kg,
        wind_speed=wind_speed,
        wind_direction=wind_direction

    )

    print(f"Average speed: {results['average_speed_m_s']:.2f} m/s ({results['average_speed_kmh']:.2f} km/h)")
    print(f"Average power: {results['average_power']:.2f} W")
    print(f"Normalized power: {results['normalized_power']:.2f} W")
    print(f"Time saved compared to even pacing: {time_saved:.2f}s")


    

    




"""
# Example usage
power_by_gradient_example = {
    (-np.inf, -0.02): 300,
    (-0.02, 0.0): 370,
    (0.0, 0.03): 380,
    (0.03, 0.06): 450,
    (0.06, np.inf): 500
}

result = analyze_gpx_power_with_energy(
    gpx_path,
    power_by_gradient_example,
    wind_speed=5,
    wind_direction=0,    #Measure clockwise from north. 0 degrees is wind from north, 180 is wind from south
    cda = 0.25
)


print(f"Average speed: {result['average_speed_m_s']:.2f} m/s ({result['average_speed_kmh']:.2f} km/h)")
print(f"Average power: {result['average_power']:.2f} W")
print(f"Normalized power: {result['normalized_power']:.2f} W")
"""