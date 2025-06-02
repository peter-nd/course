import streamlit as st
import gpxpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib import cm

# Your existing functions here (latlon_to_xy_meters, plot_route_2d_with_speed, plot_route_3d_with_speed)

def latlon_to_xy_meters(lat, lon):
    R = 6371000
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    mean_lat_rad = np.mean(lat_rad)
    x = R * (lon_rad - lon_rad[0]) * np.cos(mean_lat_rad)
    y = R * (lat_rad - lat_rad[0])
    return x, y

def plot_route_3d_with_speed(lat, lon, ele, speeds):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Route Colored by Speed')
    x, y = latlon_to_xy_meters(lat, lon)
    points = np.array([x, y, ele]).T
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    speeds_norm = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-9)
    cmap = cm.get_cmap('viridis')
    colors = cmap(speeds_norm)
    for i in range(len(segments)):
        seg = segments[i]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colors[i])
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
    st.pyplot(fig)

def plot_route_2d_with_speed(lat, lon, speeds):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('2D Route Colored by Speed')
    x, y = latlon_to_xy_meters(lat, lon)
    points = np.array([x, y]).T
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
    ax.set_aspect('equal')
    st.pyplot(fig)

def analyze_gpx_power_with_energy(
    gpx_file,
    power_by_gradient,
    mass_kg=88,
    cda=0.36,
    c_rr=0.003,
    wind_speed=0.0,
    wind_direction=0.0,
    air_density=1.225,
    g=9.81
):
    # Read GPX from file-like object
    gpx = gpxpy.parse(gpx_file)

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
        R = 6371000
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

    v_prev = 2.0
    h_prev = ele[0]
    E_prev = 0.5 * mass_kg * v_prev**2 + mass_kg * g * h_prev

    speeds = []
    segment_times = []

    min_speed = 0.5
    max_dt = 10.0

    for i in range(len(segment_distances)):
        dist = segment_distances[i]
        slope_i = slope[i]
        h_new = ele[i + 1]
        power = powers[i]
        wind_i = effective_wind_speed[i]

        v_prev = max(v_prev, min_speed)
        dt = dist / v_prev
        dt = min(dt, max_dt)

        E_add = power * dt
        F_aero = 0.5 * air_density * cda * (v_prev + wind_i)**2
        E_aero = F_aero * dist
        F_roll = mass_kg * g * np.cos(slope_i) * c_rr
        E_roll = F_roll * dist

        E_mech_new = E_prev + E_add - E_aero - E_roll
        E_pot_new = mass_kg * g * h_new
        E_kin_new = E_mech_new - E_pot_new

        if E_kin_new < 0:
            v_new = min_speed
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

    return {
        "average_speed_m_s": average_speed_m_s,
        "average_speed_kmh": average_speed_kmh,
        "average_power": average_power,
        "normalized_power": normalized_power,
        "cumulative_distance": cumulative_distance,
        "elevation": ele,
        "speeds": speeds,
        "lat": lat,
        "lon": lon,
    }

# === Streamlit app interface ===
st.title("GPX Route Power & Speed Analysis")

uploaded_file = st.file_uploader("Upload your GPX file", type=["gpx"])

if uploaded_file:
    st.sidebar.header("Power by Gradient Input")
    # Input power for slope intervals, default values provided
    power_neg = st.sidebar.number_input("Power for gradient < -2%", value=300)
    power_flat_neg = st.sidebar.number_input("Power for gradient -2% to 0%", value=370)
    power_flat_pos = st.sidebar.number_input("Power for gradient 0% to 3%", value=380)
    power_mid = st.sidebar.number_input("Power for gradient 3% to 6%", value=450)
    power_high = st.sidebar.number_input("Power for gradient > 6%", value=500)

    power_by_gradient = {
        (-np.inf, -0.02): power_neg,
        (-0.02, 0.0): power_flat_neg,
        (0.0, 0.03): power_flat_pos,
        (0.03, 0.06): power_mid,
        (0.06, np.inf): power_high,
    }

    mass = st.sidebar.number_input("Rider + Bike Mass (kg)", value=88)
    cda = st.sidebar.number_input("CdA (drag area)", value=0.36, format="%.3f")
    c_rr = st.sidebar.number_input("Rolling resistance coefficient", value=0.003, format="%.4f")
    wind_speed = st.sidebar.number_input("Wind speed (m/s)", value=0.0, format="%.2f")
    wind_dir = st.sidebar.number_input("Wind direction (degrees, 0 = from North)", value=0)

    # Run analysis button
    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):
            result = analyze_gpx_power_with_energy(
                uploaded_file,
                power_by_gradient,
                mass_kg=mass,
                cda=cda,
                c_rr=c_rr,
                wind_speed=wind_speed,
                wind_direction=wind_dir,
            )
        st.success("Analysis complete!")

        st.write(f"**Average speed:** {result['average_speed_m_s']:.2f} m/s ({result['average_speed_kmh']:.2f} km/h)")
        st.write(f"**Average power:** {result['average_power']:.2f} W")
        st.write(f"**Normalized power:** {result['normalized_power']:.2f} W")

        # Plot elevation and speed
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color = 'tab:blue'
        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Elevation (m)', color=color)
        ax1.plot(result['cumulative_distance'] / 1000, result['elevation'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title('Elevation vs Distance')
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Speed (kph)', color=color)
        ax2.plot(result['cumulative_distance'][1:] / 1000, result['speeds'] * 3.6, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        st.pyplot(fig)

        # Show 2D and 3D route plots
        plot_route_2d_with_speed(result['lat'], result['lon'], result['speeds'])
        plot_route_3d_with_speed(result['lat'], result['lon'], result['elevation'], result['speeds'])

else:
    st.info("Please upload a GPX file to start analysis.")
