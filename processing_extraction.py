import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.decomposition import PCA

# Global Configuration
FS = 30.0           # Sampling frequency (Hz)
CUTOFF = 4.0        # Butterworth low-pass cutoff frequency (Hz)
BUTTER_ORDER = 4    # Butterworth filter order

# Gait detection thresholds
FORCE_THRESHOLD = 500       # Minimum ADC force reading to consider foot grounded (a.u.)
ACTIVITY_THRESHOLD = 150    # Rolling std threshold to detect walking vs. idle
LOADING_WINDOW_SEC = 0.15   # Time window (s) used to compute heel strike loading rate
IDLE_MARGIN_SEC = 0.2       # Padding (s) added around detected walking window

# Stride validity bounds
MIN_STRIDE_SEC = 0.4        # Shortest plausible stride duration (s)
MAX_STRIDE_SEC = 2.5        # Longest plausible stride duration (s)
MIN_STRIDE_SAMPLES = 20     # Minimum samples required to attempt detection
MIN_ANALYSIS_SAMPLES = 50   # Minimum samples required for full gait analysis

# Peak detection
PEAK_DISTANCE_SEC = 0.4     # Minimum time between heel strikes (s)
COP_MASK_THRESHOLD = 100        # Below this force, COP is unreliable and masked to zero

# Calibration
GRAVITY_REST_FRAMES = 30    # Number of initial frames used to estimate gravity vector
PCA_FIT_FRAMES = 300        # Max frames used to fit PCA for forward direction
PCA_FIT_SKIP = 50           # Frames skipped at start before fitting PCA

def get_latest_files(target_names=["ESP32_GAIT", "ESP32_GAIT2"]):
    latest_files = {}
    for name in target_names:
        list_of_files = glob.glob(f"{name}_*.csv")
        if not list_of_files: continue
        latest_files[name] = max(list_of_files, key=os.path.getmtime)
    return latest_files

def sync_and_resample(df1, df2, target_fs=FS):
    """Aligns two dataframes to a common time grid using the overlapping PC time."""
    t_start = max(df1['pc_time'].min(), df2['pc_time'].min())
    t_end = min(df1['pc_time'].max(), df2['pc_time'].max())
    common_time = np.arange(t_start, t_end, 1.0 / target_fs)
    
    def resample(df, grid):
        new_df = pd.DataFrame({'pc_time': grid})
        for col in df.columns:
            if col == 'pc_time': continue
            new_df[col] = np.interp(grid, df['pc_time'], df[col])
        return new_df
        
    return resample(df1, common_time), resample(df2, common_time)

def detect_walking_window(df1, df2, threshold=ACTIVITY_THRESHOLD):
    """Detects when walking starts and ends to trim idle time."""
    total_force = (df1['fsr1'] + df1['fsr2']) + (df2['fsr1'] + df2['fsr2'])
    activity = total_force.rolling(window=int(FS/2), center=True).std().fillna(0)
    walk_mask = activity > threshold
    indices = np.where(walk_mask)[0]
    
    if len(indices) < MIN_STRIDE_SAMPLES: 
        return 0, len(df1) - 1
    
    start = max(0, indices[0] - int(IDLE_MARGIN_SEC * FS))
    end = min(len(df1) - 1, indices[-1] + int(IDLE_MARGIN_SEC * FS))
    return start, end

def calibrate_and_rotate(df):
    """Calibrates gravity and determines forward movement direction."""
    rest_period = df.iloc[:GRAVITY_REST_FRAMES]
    g_vec = rest_period[['ax', 'ay', 'az']].mean().values
    g_norm = np.linalg.norm(g_vec)
    z_axis = g_vec / g_norm 
    
    if abs(z_axis[0]) < 0.9: x_temp = np.array([1, 0, 0])
    else: x_temp = np.array([0, 1, 0])
    y_temp = np.cross(z_axis, x_temp)
    y_temp /= np.linalg.norm(y_temp)
    x_temp = np.cross(y_temp, z_axis)
    
    raw_acc = df[['ax', 'ay', 'az']].values
    acc_v = np.dot(raw_acc, z_axis) - g_norm  
    acc_h_x = np.dot(raw_acc, x_temp)
    acc_h_y = np.dot(raw_acc, y_temp)
    
    horizontal_data = np.column_stack((acc_h_x, acc_h_y))
    pca = PCA(n_components=2)
    # Fit PCA on a segment of data likely to contain movement
    fit_idx = min(PCA_FIT_FRAMES, len(horizontal_data))
    pca.fit(horizontal_data[min(PCA_FIT_SKIP, fit_idx-1):fit_idx]) 
    
    acc_fwd = pca.transform(horizontal_data)[:, 0]
    df['acc_vert'] = acc_v
    df['acc_horiz'] = np.abs(acc_fwd) 
    return df

def analyze_gait_data(df, sensor_name):
    """Analyzes a pre-loaded and resampled DataFrame with COP cleaning."""
    if len(df) < MIN_ANALYSIS_SAMPLES: return None, None, None
    dt = 1.0 / FS
    df = calibrate_and_rotate(df)

    def butter_lp(data, cutoff=CUTOFF):
        nyq = 0.5 * FS
        b, a = butter(BUTTER_ORDER, cutoff / nyq, btype='low')
        return filtfilt(b, a, data)
    
    df['f_toe'] = butter_lp(df['fsr1']).clip(min=0)
    df['f_heel'] = butter_lp(df['fsr2']).clip(min=0)
    df['av_s'] = butter_lp(df['acc_vert'])
    df['af_s'] = butter_lp(df['acc_horiz'])
    df['total_force'] = df['f_toe'] + df['f_heel']
    
    df['cop'] = df['f_toe'] / (df['total_force'] + 1e-6)
    df.loc[df['total_force'] < COP_MASK_THRESHOLD, 'cop'] = 0

    prominence = df['f_heel'].std() * PROMINENCE_SCALE
    hs_idx, _ = find_peaks(df['f_heel'], prominence=prominence, distance=int(PEAK_DISTANCE_SEC * FS))

    stride_lens, stride_times, loading_rates, swing_phases = [], [], [], []
    stance_phases = []

    for i in range(len(hs_idx)-1):
        s, e = hs_idx[i], hs_idx[i+1]
        dur = (e - s) * dt
        if not (MIN_STRIDE_SEC < dur < MAX_STRIDE_SEC): continue 
        
        stride_times.append(dur)
        
        loading_window = min(e, s + int(LOADING_WINDOW_SEC * FS))
        grad = np.max(np.gradient(df['f_heel'].iloc[s:loading_window], dt))
        loading_rates.append(abs(grad))
        
        # Calculate stance and swing phase percentages for this stride
        stride_force = df['total_force'].iloc[s:e]
        stance_samples = (stride_force > FORCE_THRESHOLD).sum()
        
        current_stance_pct = (stance_samples / len(stride_force)) * 100
        stance_phases.append(current_stance_pct)
        
        swing_pct = (1.0 - (stance_samples / len(stride_force))) * 100
        swing_phases.append(swing_pct)
        
        v_h_raw = np.cumsum(df['af_s'].iloc[s:e] * dt)
        v_h = v_h_raw - np.linspace(0, v_h_raw.iloc[-1], len(v_h_raw))
        s_len = np.trapezoid(np.abs(v_h), dx=dt)
        stride_lens.append(s_len)

    cv_stride = (np.std(stride_times) / np.mean(stride_times)) * 100 if stride_times else 0
    jerk_v = np.gradient(df['av_s'], dt)
    jerk_h = np.gradient(df['af_s'], dt)
    jerk_index = np.sqrt(np.mean(jerk_v**2 + jerk_h**2))

    stride_cadences = [60.0 / d for d in stride_times]

    stats = {
        "cadence": (len(stride_times) / (sum(stride_times)/60)) if stride_times else 0,
        "stance": np.mean(stance_phases) if stance_phases else 0,
        "steps": len(stride_times),
        "len": np.mean(stride_lens) * 100 if stride_lens else 0,
        "cv_stride": cv_stride,
        "swing": np.mean(swing_phases) if swing_phases else 0,
        "jerk": jerk_index,
        "loading_rate": np.mean(loading_rates) if loading_rates else 0,
        "cv_stance": (np.std(stance_phases) / np.mean(stance_phases)) * 100 if stance_phases else 0,
        "cv_cadence": (np.std(stride_cadences) / np.mean(stride_cadences)) * 100 if stride_cadences else 0
    }
    return df, hs_idx, stats

def main():
    targets = ["ESP32_GAIT", "ESP32_GAIT2"]
    files = get_latest_files(targets)
    
    if len(files) < 2:
        print("Error: Could not find both sensor files.")
        return

    # 1. LOAD RAW DATA
    print(f"Loading: {files['ESP32_GAIT']} and {files['ESP32_GAIT2']}")
    df1_raw = pd.read_csv(files["ESP32_GAIT"])
    df2_raw = pd.read_csv(files["ESP32_GAIT2"])

    # 2. SYNCHRONIZE & RESAMPLE
    df1_sync, df2_sync = sync_and_resample(df1_raw, df2_raw, FS)

    # 3. AUTOMATIC TRIMMING (Remove idle time at start/end)
    start_idx, end_idx = detect_walking_window(df1_sync, df2_sync)
    df1_trimmed = df1_sync.iloc[start_idx:end_idx].reset_index(drop=True)
    df2_trimmed = df2_sync.iloc[start_idx:end_idx].reset_index(drop=True)
    
    print(f"Processing walking window: {len(df1_trimmed)/FS:.1f} seconds.")

    # 4. ANALYZE EACH FOOT (Passing DataFrames, not paths)
    df1, hs1, s1 = analyze_gait_data(df1_trimmed, "ESP32_GAIT")
    df2, hs2, s2 = analyze_gait_data(df2_trimmed, "ESP32_GAIT2")

    if s1 is None or s2 is None:
        print("Error: Analysis failed. Segment too short.")
        return

    # 5. CALCULATE COMPARATIVE METRICS
    threshold = FORCE_THRESHOLD
    both_down = (df1['total_force'] > threshold) & (df2['total_force'] > threshold)
    double_support_pct = both_down.mean() * 100
    symmetry_val = abs(s1['len'] - s2['len']) / (0.5 * (s1['len'] + s2['len'])) * 100

    # 6. OUTPUT RESULTS
    print(f"\n{'Metric':<25} | {'Sensor 1 (L)':<22} | {'Sensor 2 (R)':<22}")
    print("-" * 75)
    print(f"{'Cadence (Steps/min)':<25} | {s1['cadence']:<22.1f} | {s2['cadence']:<22.1f}")
    print(f"{'Avg Stride Length':<25} | {str(round(s1['len'],1))+' cm':<22} | {str(round(s2['len'],1))+' cm':<22}")
    print(f"{'Stance Ratio (%)':<25} | {str(round(s1['stance'],1))+'%':<22} | {str(round(s2['stance'],1))+'%':<22}")
    print(f"{'Swing Phase (%)':<25} | {str(round(s1['swing'],1))+'%':<22} | {str(round(s2['swing'],1))+'%':<22}")
    print(f"{'Stride Variability (CV)':<25} | {str(round(s1['cv_stride'],1))+'%':<22} | {str(round(s2['cv_stride'],1))+'%':<22}")
    print(f"{'Stance Variability (CV)':<25} | {str(round(s1['cv_stance'],1))+'%':<22} | {str(round(s2['cv_stance'],1))+'%':<22}")
    print(f"{'Cadence Variability (CV)':<25} | {str(round(s1['cv_cadence'],1))+'%':<22} | {str(round(s2['cv_cadence'],1))+'%':<22}")
    print(f"{'Smoothness (Jerk)':<25} | {str(round(s1['jerk'],1)):<22} | {str(round(s2['jerk'],1)):<22}")
    print("-" * 75)
    print(f"Double Support Time: {double_support_pct:.1f}%")
    print(f"Gait Asymmetry Index: {symmetry_val:.2f}%")

    # 7. PLOTTING
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    t = df1['pc_time'] - df1['pc_time'].iloc[0]

    axes[0].plot(t, df1['total_force'], label="Left")
    axes[0].plot(t, df2['total_force'], label="Right")
    axes[0].set_title("Foot Pressure Rhythm")
    
    axes[1].plot(t, df1['cop'], label="Left COP")
    axes[1].plot(t, df2['cop'], label="Right COP")
    axes[1].set_title("Center of Pressure (0=Heel, 1=Toe)")
    
    axes[2].plot(t, df1['acc_horiz'], label="Left Forward Accel")
    axes[2].plot(t, df2['acc_horiz'], label="Right Forward Accel", alpha=0.7)
    axes[2].set_title("Forward Acceleration (m/s²)")

    for ax in axes: ax.legend(); ax.grid(True, alpha=0.3)
    plt.xlabel("Walking Time (s)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()