"""
Video to Frames Converter
Splits a video file into individual frames and saves them to a directory.
"""

import os
import sys
import argparse
import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, frame_skip=1, frame_format='jpg', prefix='frame'):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_skip: Extract every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)
        frame_format: Output image format (jpg, png, bmp)
        prefix: Prefix for output filenames
    """
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print("\n" + "="*60)
    print("VIDEO INFORMATION")
    print("="*60)
    print(f"File: {os.path.basename(video_path)}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Frame skip: Every {frame_skip} frame(s)")
    print(f"Expected output: ~{total_frames // frame_skip} frames")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    print("Extracting frames...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame if it matches the skip pattern
        if frame_count % frame_skip == 0:
            # Create filename with zero-padded numbers
            filename = f"{prefix}_{saved_count:06d}.{frame_format}"
            output_path = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            # Show progress every 100 frames
            if saved_count % 100 == 0:
                print(f"  Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("="*60 + "\n")
    
    return True


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract frames from a video file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all frames from video
  python video_to_frames.py --video my_video.mp4
  
  # Extract every 10th frame
  python video_to_frames.py --video my_video.mp4 --skip 10
  
  # Specify custom output directory
  python video_to_frames.py --video my_video.mp4 --output frames_output
  
  # Save as PNG format
  python video_to_frames.py --video my_video.mp4 --format png
        """
    )
    
    parser.add_argument('--video', '-v',
                        required=True,
                        help='Path to input video file')
    
    parser.add_argument('--output', '-o',
                        default=None,
                        help='Output directory for frames (default: video_name_frames)')
    
    parser.add_argument('--skip', '-s',
                        type=int,
                        default=1,
                        help='Extract every Nth frame (default: 1 = all frames)')
    
    parser.add_argument('--format', '-f',
                        choices=['jpg', 'png', 'bmp'],
                        default='jpg',
                        help='Output image format (default: jpg)')
    
    parser.add_argument('--prefix', '-p',
                        default='frame',
                        help='Prefix for output filenames (default: frame)')
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)
    
    # Check if it's a valid video extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_ext = os.path.splitext(args.video)[1].lower()
    if video_ext not in valid_extensions:
        print(f"WARNING: '{video_ext}' may not be a valid video format")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Create default output directory based on video filename
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_dir = f"{video_name}_frames"
    
    # Extract frames
    success = extract_frames(
        video_path=args.video,
        output_dir=output_dir,
        frame_skip=args.skip,
        frame_format=args.format,
        prefix=args.prefix
    )
    
    if success:
        print("✓ Frame extraction completed successfully!")
    else:
        print("✗ Frame extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

