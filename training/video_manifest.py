import json
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


def _ffprobe_bin(ffmpeg_path: Optional[str]) -> str:
    """
    Resolve ffprobe executable based on configured ffmpeg path.
    """
    if ffmpeg_path:
        ffmpeg_path = str(ffmpeg_path)
        p = Path(ffmpeg_path)
        if p.name.lower().startswith("ffmpeg"):
            probe_candidate = p.with_name("ffprobe" + p.suffix)
            if probe_candidate.exists():
                return str(probe_candidate)
    return "ffprobe"


def _probe_video(path: Path, ffmpeg_path: Optional[str]) -> Dict[str, Any]:
    """
    Probe a video file using ffprobe to extract duration, fps, and resolution.
    """
    probe_bin = _ffprobe_bin(ffmpeg_path)
    cmd = [
        probe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        data = json.loads(result.stdout)
        stream = (data.get("streams") or [{}])[0]
        width = stream.get("width")
        height = stream.get("height")
        fps_str = stream.get("r_frame_rate")
        duration = stream.get("duration") or stream.get("tags", {}).get("DURATION")
        fps_val: Optional[float] = None
        if fps_str and isinstance(fps_str, str) and "/" in fps_str:
            num, den = fps_str.split("/", 1)
            try:
                fps_val = float(num) / float(den)
            except Exception:
                fps_val = None
        try:
            duration_val = float(duration) if duration is not None else None
        except Exception:
            duration_val = None
        return {
            "duration_sec": duration_val,
            "fps": fps_val,
            "resolution": f"{width}x{height}" if width and height else None,
        }
    except Exception:
        return {
            "duration_sec": None,
            "fps": None,
            "resolution": None,
        }


def manifest_path(training_dir: Path, run_id: str) -> Path:
    return training_dir / run_id / "manifest.json"


def generate_manifest(run_id: str, training_dir: Path, ffmpeg_path: Optional[str]) -> Dict[str, Any]:
    """
    Generate a manifest by scanning training video segments.
    """
    run_dir = training_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    segment_files = sorted(run_dir.glob("*.mp4"))
    segments: List[Dict[str, Any]] = []
    for idx, path in enumerate(segment_files):
        meta = _probe_video(path, ffmpeg_path)
        env_index = None
        name = path.name
        if "env_" in name:
            try:
                env_index = int(name.split("env_")[1].split("_")[0])
            except Exception:
                env_index = None
        segment = {
            "path": path.name,
            "index": idx,
            "duration_sec": meta.get("duration_sec"),
            "fps": meta.get("fps"),
            "resolution": meta.get("resolution"),
            "env_index": env_index,
            "created_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        segments.append(segment)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "segments": segments,
    }

    mp = manifest_path(training_dir, run_id)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(run_id: str, training_dir: Path) -> Optional[Dict[str, Any]]:
    mp = manifest_path(training_dir, run_id)
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def ensure_manifest(run_id: str, training_dir: Path, ffmpeg_path: Optional[str]) -> Dict[str, Any]:
    manifest = load_manifest(run_id, training_dir)
    if manifest:
        return manifest
    return generate_manifest(run_id, training_dir, ffmpeg_path)


def hash_manifest(manifest: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
