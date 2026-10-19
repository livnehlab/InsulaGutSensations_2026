import matplotlib
matplotlib.use('Qt5Agg')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector, EllipseSelector
from scipy.signal import butter, filtfilt

# ----------------------------------------
# 1) Utility: load raw as memmap
# ----------------------------------------
def load_raw_memmap(path, width, height, dtype=np.uint16):
    """
    Memory-map a raw binary file and reshape into (n_frames, height, width).
    """
    itemsize = np.dtype(dtype).itemsize
    file_size = os.path.getsize(path)
    num_pixels = file_size // itemsize
    total_frames = num_pixels // (width * height)
    if total_frames * width * height != num_pixels:
        raise ValueError("File size is not an integer multiple of width*height*dtype")
    return np.memmap(path, dtype=dtype, mode='r', shape=(total_frames, height, width))


# ----------------------------------------
# 2) Separate channels & planes (1 or 2 channels)
# ----------------------------------------
def separate_channels_and_planes(all_frames, n_planes, channels=2):
    """
    Split raw frames into per-channel stacks and then into planes.

    Returns a dict: {
        'green': [np.ndarray per plane, each (T, H, W)],
        'red':   [np.ndarray per plane, each (T, H, W)]  # only if channels == 2
    }
    """
    total_frames, H, W = all_frames.shape

    if channels == 2:
        if total_frames % (2 * n_planes) != 0:
            raise ValueError("Total frames must be divisible by 2 * n_planes for 2-channel data")
        n_cycles = total_frames // (2 * n_planes)

        green_frames = all_frames[0::2]
        red_frames   = all_frames[1::2]

        green_frames = green_frames.reshape((n_cycles, n_planes, H, W))
        red_frames   = red_frames.reshape((n_cycles, n_planes, H, W))

        green_vids = [green_frames[:, p, :, :] for p in range(n_planes)]
        red_vids   = [red_frames[:, p, :, :] for p in range(n_planes)]
        return {'green': green_vids, 'red': red_vids}

    elif channels == 1:
        if total_frames % n_planes != 0:
            raise ValueError("Total frames must be divisible by n_planes for 1-channel data")
        n_cycles = total_frames // n_planes

        green_frames = all_frames.reshape((n_cycles, n_planes, H, W))
        green_vids = [green_frames[:, p, :, :] for p in range(n_planes)]
        return {'green': green_vids}

    else:
        raise ValueError("channels must be 1 or 2")


# ----------------------------------------
# 3) Compute standard-deviation projection
# ----------------------------------------
def std_projection(vid_stack, n_sample=500):
    # Pick evenly spaced frames so that temporal structure is kept
    if n_sample is not None and vid_stack.shape[0] > n_sample:
        idx = np.linspace(0, vid_stack.shape[0] - 1, n_sample, dtype=int)
        return vid_stack[idx].std(axis=0)
    return vid_stack.std(axis=0)


# ----------------------------------------
# 4) Draw a single rectangular ROI
# ----------------------------------------
def select_rectangle_roi_on_image(image):
    """
    Display `image` (a single frame) and let the user draw exactly ONE rectangle.
    Returns (xmin, xmax, ymin, ymax) in integer pixel coords.
    """
    roi_coords = {}

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(
        "Draw ONE rectangular ROI on this frame:\n"
        "- Click and drag to form the rectangle\n"
        "- Release to finalize\n"
        "- Close window when done"
    )
    plt.axis('off')

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        xmin, xmax = int(round(min(x1, x2))), int(round(max(x1, x2)))
        ymin, ymax = int(round(min(y1, y2))), int(round(max(y1, y2)))
        roi_coords['xmin'] = max(0, xmin)
        roi_coords['xmax'] = min(image.shape[1] - 1, xmax)
        roi_coords['ymin'] = max(0, ymin)
        roi_coords['ymax'] = min(image.shape[0] - 1, ymax)

        # Draw a rectangle patch for feedback
        rect = patches.Rectangle(
            (roi_coords['xmin'], roi_coords['ymin']),
            roi_coords['xmax'] - roi_coords['xmin'],
            roi_coords['ymax'] - roi_coords['ymin'],
            linewidth=1.5, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],            # left click only
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=False
    )

    plt.show()
    if not roi_coords:
        raise ValueError("No rectangle drawn")
    return roi_coords['xmin'], roi_coords['xmax'], roi_coords['ymin'], roi_coords['ymax']


# ----------------------------------------
# 5) Draw any number of elliptical ROIs
# ----------------------------------------
def select_ellipse_rois_on_image(image):
    """
    Display a single-image projection and let the user draw any number
    of axis-aligned elliptical ROIs.

    Returns:
      masks  : list of 2-D boolean ndarrays
      params : list of dicts with keys cx, cy, rx, ry
    """
    masks, params = [], []
    ny, nx = image.shape

    def _enhance(img, p_low=1, p_high=99):
        v_min, v_max = np.percentile(img, (p_low, p_high))
        img = np.clip(img, v_min, v_max)
        return (img - v_min) / (v_max - v_min + 1e-6)

    fig, ax = plt.subplots()
    ax.imshow(_enhance(image), cmap='gray')
    ax.set_title(
        "Draw ellipse ROIs:\n"
        "- Click and drag to place an ellipse\n"
        "- Release to add it\n"
        "- Close the window when finished"
    )
    plt.axis('off')

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx = abs(x2 - x1) / 2.0
        ry = abs(y2 - y1) / 2.0
        if rx < 1 or ry < 1:
            return

        yy, xx = np.ogrid[:ny, :nx]
        mask = ((xx - cx)**2) / (rx**2) + ((yy - cy)**2) / (ry**2) <= 1.0
        masks.append(mask)
        params.append({'cx': float(cx), 'cy': float(cy),
                       'rx': float(rx), 'ry': float(ry)})

        patch = patches.Ellipse((cx, cy), 2*rx, 2*ry,
                                edgecolor='red', facecolor='none',
                                linewidth=1.5)
        ax.add_patch(patch)
        fig.canvas.draw_idle()

    selector = EllipseSelector(
        ax, onselect,
        useblit=True,
        button=[1],            # left click only
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=False
    )

    plt.show()
    return masks, params


# ----------------------------------------
# 6) Extract ROI traces from a stack
# ----------------------------------------
def extract_traces(vid_stack, masks):
    """
    Given a video stack (n_time, H, W) and a list of ROI masks (H x W),
    return an array of shape (n_rois, n_time) with mean pixel value inside each ROI.
    """
    n_time = vid_stack.shape[0]
    n_rois = len(masks)
    traces = np.zeros((n_rois, n_time), dtype=np.float32)

    mask_indices = [np.nonzero(m) for m in masks]

    for i, inds in enumerate(mask_indices):
        for t in range(n_time):
            traces[i, t] = vid_stack[t][inds].mean()

    return traces


# ----------------------------------------
# 7) Simple moving averages and filters
# ----------------------------------------
def moving_average_same(x, window):
    """Centered moving average that keeps length len(x)."""
    if window < 2:
        return x.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def butter_highpass(data, cutoff_hz, fs_hz, order=3):
    """Zero-phase Butterworth high-pass filter."""
    nyquist = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)


def butter_lowpass(data, cutoff_hz, fs_hz=1.0, order=3):
    """Zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * fs_hz
    norm_cut = cutoff_hz / nyq
    b, a = butter(order, norm_cut, btype="low", analog=False)
    return filtfilt(b, a, data)


# ----------------------------------------
# 8) Find raw-frame cut points by detecting big steps
# ----------------------------------------
def find_cut_points_by_rectangle_roi(
    mm_all,
    xmin, xmax, ymin, ymax,
    ds=8,
    lp_cutoff=0.03,
    lp_order=3,
    diff_win=30,
    jump_abs=30,
    channels=2,
    motion_channel="green"   # "green" or "red" (if available)
):
    """
    Detect step-like jumps in a small ROI on the chosen motion_channel.
    Returns:
        raw_cut_idxs, diff_signal, jump_abs, ROI_signal, ROI_lp
    """

    # Choose the view
    if channels == 2:
        if motion_channel == "green":
            offset = 0
        elif motion_channel == "red":
            offset = 1
        else:
            raise ValueError("motion_channel must be 'green' or 'red' for 2 channels")
        stride = 2
    elif channels == 1:
        if motion_channel != "green":
            raise ValueError("Only 'green' channel exists in 1-channel data")
        offset = 0
        stride = 1
    else:
        raise ValueError("channels must be 1 or 2")

    # Crop and downsample frames of the chosen channel
    ch_frames = mm_all[offset::stride, ymin:ymax + 1:ds, xmin:xmax + 1:ds]
    n_ch = ch_frames.shape[0]
    if n_ch <= diff_win:
        return [], np.array([]), jump_abs, np.zeros(0), np.zeros(0)

    # Mean intensity per channel frame
    ROI_signal = ch_frames.reshape(n_ch, -1).mean(axis=1)

    # Low-pass filter
    ROI_lp = butter_lowpass(ROI_signal, lp_cutoff, fs_hz=1.0, order=lp_order)

    # Wide diff over diff_win
    diff_signal = np.zeros_like(ROI_lp)
    diff_signal[diff_win:] = ROI_lp[diff_win:] - ROI_lp[:-diff_win]

    # Threshold
    idxs = np.where(np.abs(diff_signal) >= jump_abs)[0]

    # Group contiguous and keep first of each group
    spike_idxs = []
    if idxs.size:
        breaks = np.where(np.diff(idxs) != 1)[0]
        for g in np.split(idxs, breaks + 1):
            onset_idx = int(g[0]) + diff_win
            if onset_idx < len(ROI_lp):
                spike_idxs.append(onset_idx)
    spike_idxs = np.asarray(spike_idxs, dtype=int)

    # Map back to raw indices
    raw_cut_idxs = []
    for i in spike_idxs:
        if channels == 2 and motion_channel == "red":
            # Preserve original mapping used in the old code path
            raw_idx = 2 * i + 3
        else:
            raw_idx = offset + stride * i
        raw_cut_idxs.append(raw_idx)
    raw_cut_idxs = sorted(set(int(x) for x in raw_cut_idxs))

    # Diagnostic plot
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(n_ch), ROI_lp, label=f"{motion_channel} ROI low-pass")
    if spike_idxs.size:
        plt.scatter(spike_idxs, ROI_lp[spike_idxs], color="red", s=40, label="detected jump")
    plt.axhline(ROI_lp.mean(), color="gray", linestyle=":")
    plt.xlabel(f"{motion_channel}-frame index")
    plt.ylabel("intensity (a.u.)")
    plt.title(f"Low-pass ROI, jumps where |delta over {diff_win} frames| >= {jump_abs}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Detected {len(raw_cut_idxs)} cut points:", raw_cut_idxs)
    return raw_cut_idxs, diff_signal, jump_abs, ROI_signal, ROI_lp


# ----------------------------------------
# 9) Main script
# ----------------------------------------
def main():
    # Option: process all planes or specific planes
    planes_to_use = None  # set like [0, 1] to process only specific planes

    parser = argparse.ArgumentParser(
        description="Load .raw 2P data, draw a rectangular ROI on the first motion channel frame, "
                    "detect steps, split on those, then for each segment draw ellipse ROIs and extract traces."
    )
    parser.add_argument('raw_path', type=str,
                        nargs='?',
                        default=r"Z:\2p_data\Omer_gastric_balloons\OI348\OI348int_balloon1\Image_001_001.raw",
                        help="Path to the .raw file")
    parser.add_argument('width', type=int, nargs='?', default=512, help="Image width in pixels")
    parser.add_argument('height', type=int, nargs='?', default=512, help="Image height in pixels")
    parser.add_argument('n_planes', type=int, nargs='?', default=1, help="Number of z-planes per channel")

    parser.add_argument('--threshold_factor', type=float, default=3.0,
                        help="Std-multiplier for motion threshold on ROI-trace diff (kept for compatibility)")
    parser.add_argument('--ds', type=int, default=8, help="Spatial downsample factor for motion detection")
    parser.add_argument('--smooth_win', type=int, default=5,
                        help="Window length for moving-average smoothing of the ROI trace (compatibility)")

    parser.add_argument('--channels', default=2, help='Number of channels: "auto", 1, or 2')
    parser.add_argument('--motion_channel', default="red",
                        help='Channel for motion detection: "green" or "red" (if red exists)')

    args = parser.parse_args()

    raw_path = args.raw_path
    width = args.width
    height = args.height
    n_planes = args.n_planes
    ds = args.ds

    # sanitize channels flag
    if isinstance(args.channels, str) and args.channels.lower() == "auto":
        channels_flag = "auto"
    else:
        try:
            channels_flag = int(args.channels)
            if channels_flag not in (1, 2):
                raise ValueError
        except Exception:
            raise ValueError('--channels must be "auto", 1, or 2')

    motion_channel = args.motion_channel.lower()
    if motion_channel not in ("green", "red"):
        raise ValueError('--motion_channel must be "green" or "red"')

    # 1) Load full video as a memmap
    print("Loading raw data as memmap ...")
    mm_all = load_raw_memmap(raw_path, width, height, dtype=np.uint16)

    # 1b) auto-detect channels if requested
    total_frames = mm_all.shape[0]
    if channels_flag == "auto":
        if total_frames % (2 * n_planes) == 0:
            channels = 2
        elif total_frames % n_planes == 0:
            channels = 1
        else:
            raise ValueError("Could not auto-detect channels. Total frames not divisible by n_planes or 2*n_planes.")
    else:
        channels = channels_flag

    if channels == 1 and motion_channel == "red":
        print("No red channel in 1-channel data. Using green for motion detection.")
        motion_channel = "green"

    # 2) Show first frame of the selected motion channel and draw rectangle ROI
    if channels == 2:
        channel_frames = mm_all[0::2] if motion_channel == "green" else mm_all[1::2]
    else:
        channel_frames = mm_all  # single channel is green

    if channel_frames.shape[0] == 0:
        raise ValueError(f"No frames found for motion channel: {motion_channel}")
    example_motion_frame = channel_frames[0]

    print(f"Displaying first {motion_channel} frame. Draw ONE rectangular ROI for motion detection.")
    xmin, xmax, ymin, ymax = select_rectangle_roi_on_image(example_motion_frame)
    print(f"Chosen rectangle: x=[{xmin}:{xmax}], y=[{ymin}:{ymax}]")

    # 3) Detect big steps based on ROI trace
    print("Detecting big movements inside chosen rectangle ROI ...")
    cuts, diff_sig, jump_abs, roi, roi_lp = find_cut_points_by_rectangle_roi(
        mm_all,
        xmin, xmax, ymin, ymax,
        ds=ds,
        lp_cutoff=0.03,
        jump_abs=1000,
        channels=channels,
        motion_channel=motion_channel
    )
    print("Detected raw-frame cut points:", cuts)

    # 4) Build segment ranges snapped to cycle boundaries
    n_frames = mm_all.shape[0]
    cycle_len = channels * n_planes

    aligned_cuts = []
    for c in cuts:
        boundary_exclusive = ((c + 1) // cycle_len) * cycle_len
        aligned_inclusive = boundary_exclusive - 1
        if aligned_inclusive > (aligned_cuts[-1] if aligned_cuts else -1):
            aligned_cuts.append(aligned_inclusive)

    segment_ranges = []
    prev = -1
    for c in aligned_cuts:
        segment_ranges.append((prev + 1, c + 1))  # end is exclusive
        prev = c

    last_exclusive = (n_frames // cycle_len) * cycle_len
    if last_exclusive > prev + 1:
        segment_ranges.append((prev + 1, last_exclusive))

    print("Aligned cut indices:", aligned_cuts)
    print("Segment ranges:", segment_ranges)

    # 5) For each segment: separate channels & planes, std projections, ROI selection, extract traces
    all_masks = {'segment': [], 'plane': [], 'masks': []}
    all_traces = {'green': [], 'red': [], 'red_ctrl': []}
    all_roi_meta = []

    if not segment_ranges:
        # If no cuts were found, process the whole movie as a single segment
        segment_ranges = [(0, last_exclusive if last_exclusive > 0 else n_frames)]

    for seg_idx, (start, end) in enumerate(segment_ranges):
        print(f"\nProcessing segment {seg_idx + 1}/{len(segment_ranges)}: frames [{start}:{end}]")
        segment_stack = mm_all[start:end]  # shape = (n_seg_frames, H, W)

        vids = separate_channels_and_planes(segment_stack, n_planes, channels=channels)

        # Determine which planes to process
        if planes_to_use is None:
            plane_indices = list(range(n_planes))
        else:
            plane_indices = [p for p in planes_to_use if 0 <= p < n_planes]

        for p_idx in plane_indices:
            print(f"  Segment {seg_idx + 1}, Plane {p_idx + 1}/{n_planes}")

            # Green channel is always present
            green_stack = vids['green'][p_idx]
            std_img_green = std_projection(green_stack)
            print("    Displaying std projection of GREEN channel. Draw elliptical ROIs now.")
            masks_green, params_green = select_ellipse_rois_on_image(std_img_green)
            if masks_green:
                green_traces = extract_traces(green_stack, masks_green)
            else:
                green_traces = np.zeros((0, green_stack.shape[0]), dtype=np.float32)

            # Red channel only if present
            if 'red' in vids:
                red_stack = vids['red'][p_idx]

                std_img_red = std_projection(red_stack)
                print("    Displaying std projection of RED channel. Draw elliptical ROIs now.")
                masks_red, params_red = select_ellipse_rois_on_image(std_img_red)
                if masks_red:
                    red_traces = extract_traces(red_stack, masks_red)
                else:
                    red_traces = np.zeros((0, red_stack.shape[0]), dtype=np.float32)

                if masks_green:
                    red_ctrl_traces = extract_traces(red_stack, masks_green)
                else:
                    red_ctrl_traces = np.zeros((0, red_stack.shape[0]), dtype=np.float32)
            else:
                masks_red, params_red = [], []
                red_traces = np.zeros((0, green_stack.shape[0]), dtype=np.float32)
                red_ctrl_traces = np.zeros((0, green_stack.shape[0]), dtype=np.float32)

            # ROI metadata
            for g_idx, p in enumerate(params_green, start=1):
                all_roi_meta.append({
                    'segment': seg_idx,
                    'plane': p_idx,
                    'channel': 'green',
                    'roi_id': g_idx,
                    'kind': 'ellipse',
                    **p
                })
            for r_idx, p in enumerate(params_red, start=1):
                all_roi_meta.append({
                    'segment': seg_idx,
                    'plane': p_idx,
                    'channel': 'red',
                    'roi_id': r_idx,
                    'kind': 'ellipse',
                    **p
                })

            # Store masks and traces
            all_masks['segment'].append(seg_idx)
            all_masks['plane'].append(p_idx)
            all_masks['masks'].append({
                'green': masks_green,
                'red': masks_red if 'red' in vids else []
            })

            all_traces['green'].append({
                'segment': seg_idx,
                'plane': p_idx,
                'traces': green_traces
            })
            all_traces['red'].append({
                'segment': seg_idx,
                'plane': p_idx,
                'traces': red_traces
            })
            all_traces['red_ctrl'].append({
                'segment': seg_idx,
                'plane': p_idx,
                'traces': red_ctrl_traces
            })

    # 6) Save all traces and metadata to a single .npz
    save_dict = {}
    for channel_name, traces_list in all_traces.items():
        for entry in traces_list:
            seg = entry['segment']
            plane = entry['plane']
            traces = entry['traces']
            key = f"{channel_name}_seg{seg + 1}_plane{plane + 1}_traces"
            save_dict[key] = traces

    save_dict['motion_rectangle'] = np.array([xmin, xmax, ymin, ymax], dtype=int)
    save_dict['roi_table'] = np.array(all_roi_meta, dtype=object)

    save_path = os.path.splitext(raw_path)[0] + '_roi_traces.npz'
    print(f"\nSaving ROI traces to {save_path}")
    np.savez_compressed(save_path, **save_dict)
    print("Done.")


if __name__ == '__main__':
    main()
