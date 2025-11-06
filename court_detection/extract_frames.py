import cv2
import os
import numpy as np
from pathlib import Path


def extract_frames_uniform(video_path, output_dir, num_frames=50):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {video_path.name}")
    print(f"    Total frames: {total_frames}")
    print(f"    FPS: {fps}")
    print(f"    Duration: {total_frames/fps:.1f}s")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem
    extracted_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print(f"    Warning: Could not read frame {idx}")
            continue

        frame = cv2.resize(frame, (960, 540))

        frame_filename = f"{video_name}_frame_{idx:06d}.jpg"
        frame_path = output_path / frame_filename
        cv2.imwrite(str(frame_path), frame)
        extracted_count += 1

        if extracted_count % 10 == 0:
            print(f"    Extracted {extracted_count}/{num_frames} frames")

    cap.release()
    print(f"    ✓ Extracted {extracted_count} frames to {output_dir}")
    return extracted_count


def extract_frames_diverse(video_path, output_dir, num_frames=50, skip_similar=True):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {video_path.name}")
    print(f"    Total frames: {total_frames}")
    print(f"    FPS: {fps}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem
    extracted_frames = []
    extracted_hists = []
    extracted_count = 0


    sample_size = num_frames * 3 if skip_similar else num_frames
    frame_indices = np.linspace(0, total_frames - 1, sample_size, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        
        frame = cv2.resize(frame, (960, 540))

        if skip_similar and extracted_count > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            is_similar = False
            for prev_hist in extracted_hists:
                similarity = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                if similarity > 0.95:
                    is_similar = True
                    break

            if is_similar:
                continue

            extracted_hists.append(hist)

        frame_filename = f"{video_name}_frame_{idx:06d}.jpg"
        frame_path = output_path / frame_filename
        cv2.imwrite(str(frame_path), frame)
        extracted_frames.append((idx, frame_path))
        extracted_count += 1

        if extracted_count % 10 == 0:
            print(f"    Extracted {extracted_count}/{num_frames} diverse frames")

        if extracted_count >= num_frames:
            break

    cap.release()
    print(f"    ✓ Extracted {extracted_count} diverse frames to {output_dir}")
    return extracted_count


def main():
    video_dir = Path("VideoInput")
    output_dir = Path("training_data/frames")

    if not video_dir.exists():
        print(f"Error: {video_dir} not found")
        return

    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video(s)")
    print("\nExtracting frames (diverse sampling)...\n")

    total_extracted = 0
    frames_per_video = 30

    for video_file in video_files:
        count = extract_frames_diverse(
            video_file,
            output_dir,
            num_frames=frames_per_video,
            skip_similar=True
        )
        total_extracted += count
        print()

    print(f"="*60)
    print(f"Total frames extracted: {total_extracted}")
    print(f"Saved to: {output_dir}")
    print(f"\nNext step: Annotate these frames with LabelMe")
    print(f"  Run: labelme {output_dir}")


if __name__ == "__main__":
    main()
