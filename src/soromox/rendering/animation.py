import cv2  # importing cv2
from jax import Array
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as onp
import os
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Optional, Union


def animate_cv2(
    rendering_fn: Callable[[Array], onp.ndarray],
    t_ts: onp.ndarray,
    q_ts: onp.ndarray,
    filepath: os.PathLike,
    width: int,
    height: int,
    speed_up: Union[float, Array] = 1,
    skip_step: int = 1,
    rgb_to_bgr: bool = True,
    **kwargs,
):
    """
    Animate using OpenCV
    Args:
        rendering_fn: A function that takes in the time and the configuration and returns an image.
        t_ts: time steps of the data
        q_ts: configurations at each time step
        filepath: path to the output video
        speed_up: The speed up factor of the video.
        skip_step: The number of time steps to skip between animation frames.
        rgb_to_bgr: whether to convert the images from RGB to BGR
        **kwargs: Additional keyword arguments for the rendering function.

    Returns:

    """
    # extract parameters
    dt = onp.mean(onp.diff(t_ts)).item()
    fps = float(speed_up / (skip_step * dt))

    # create video with fallback codecs
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure filepath has .mp4 extension
    filepath_str = str(filepath)
    if not filepath_str.endswith(('.mp4', '.avi', '.mov')):
        filepath_str += '.mp4'
    
    video = None
    success_count = 0
    failed_frames = []
    
    # First try to create the video writer using system's default codec
    try:
        # Use -1 to let OpenCV choose the codec via system dialog, or use a more reliable codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            filepath_str,
            fourcc,
            fps,
            (width, height),
            True
        )
        
        # Check if video writer was created successfully
        if not video.isOpened():
            print("Failed to open video writer - trying alternative approach")
            if video:
                video.release()
            video = None
    except Exception as e:
        print(f"Failed to create VideoWriter: {e}")
        if video:
            video.release()
        video = None

    # skip frames
    t_ts = t_ts[::skip_step]
    q_ts = q_ts[::skip_step]

    # Collect all frames first
    frames = []
    print(f"Generating {len(t_ts)} frames...")
    
    for time_idx, t in enumerate(tqdm(t_ts)):
        img = rendering_fn(q_ts[time_idx], **kwargs)

        # Ensure image is in correct format
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 1:  # single channel
            img = onp.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif rgb_to_bgr and img.shape[-1] == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Ensure correct data type and dimensions
        img = onp.asarray(img, dtype=onp.uint8)
        
        # Resize image if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        # Validate image before adding to frames
        if img.shape != (height, width, 3):
            print(f"Warning: Frame {time_idx} has unexpected shape {img.shape}, expected ({height}, {width}, 3)")
            continue
            
        if not onp.isfinite(img).all():
            print(f"Warning: Frame {time_idx} contains non-finite values")
            img = onp.nan_to_num(img, nan=0, posinf=255, neginf=0).astype(onp.uint8)
        
        # Ensure values are in valid range [0, 255]
        img = onp.clip(img, 0, 255).astype(onp.uint8)
        frames.append(img)

    # Try to write video if VideoWriter was successfully created
    if video is not None:
        print("Attempting to write video...")
        for time_idx, img in enumerate(frames):
            success = video.write(img)
            if success:
                success_count += 1
            else:
                failed_frames.append(time_idx)
                
        video.release()
        
        # Check if video writing was successful
        if success_count > 0:
            print(f"Successfully wrote {success_count}/{len(frames)} frames to video")
            if failed_frames:
                print(f"Failed to write {len(failed_frames)} frames: {failed_frames[:10]}...")
        else:
            print("All frame writes failed - falling back to image sequence")
            video = None
    
    # Fallback: Save as image sequence if video writing failed
    if video is None or success_count == 0:
        print("Saving as image sequence instead...")
        # Create directory for image sequence
        img_dir = Path(filepath_str).with_suffix('')
        img_dir.mkdir(exist_ok=True)
        
        for time_idx, img in enumerate(frames):
            img_path = img_dir / f"frame_{time_idx:06d}.png"
            cv2.imwrite(str(img_path), img)
            
        print(f"Image sequence saved to {img_dir}")
        
        # Try to create video using ffmpeg if available
        try:
            import subprocess
            output_path = str(img_dir.parent / f"{img_dir.name}.mp4")
            cmd = [
                "ffmpeg", "-y",  # overwrite output
                "-framerate", str(fps),
                "-i", str(img_dir / "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully created video using ffmpeg: {output_path}")
                # Clean up image sequence
                import shutil
                shutil.rmtree(img_dir)
                filepath_str = output_path
            else:
                print(f"ffmpeg failed: {result.stderr}")
        except (ImportError, FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"ffmpeg not available or failed: {e}")
            print(f"Image sequence preserved at: {img_dir}")

    print(f"Animation completed. Output: {filepath_str}")
