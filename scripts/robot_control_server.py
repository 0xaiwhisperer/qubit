#!/usr/bin/env python3
"""
SO-101 Animation Editor
=======================
- Dual arm keyframe animation
- Recording mode (with selective arm recording)
- Sequence chaining  
"""

import threading, time, json, os, glob, argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
from collections import deque
from flask import Flask, request, jsonify, render_template_string, Response
import logging

# Robot imports
try:
    from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
    from lerobot.robots.so100_follower.so100_follower import SO100Follower
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("⚠️ Robot libraries not available")

# Camera imports
try:
    import cv2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

logging.getLogger("lerobot").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# ============================================================
# CONFIG
# ============================================================

ARM_CONFIGS = {
    "arm1": {"port": "/dev/ttyACM2", "id": "my_awesome_follower_arm1", "name": "Arm 1"},
    "arm2": {"port": "/dev/ttyACM3", "id": "my_awesome_follower_arm2", "name": "Arm 2"},
}

ANIMATIONS_DIR, POSES_DIR, SEQUENCES_DIR = Path("animations"), Path("poses"), Path("sequences")
for d in [ANIMATIONS_DIR, POSES_DIR, SEQUENCES_DIR]: d.mkdir(exist_ok=True)

USE_DEGREES, CONTROL_HZ, MAX_UNDO = True, 50.0, 50
ALL_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_LIMITS = {j: (-150, 150) for j in ALL_JOINTS[:5]}
JOINT_LIMITS["gripper"] = (0, 100)
JOINT_DEFAULTS = {"shoulder_pan": 0, "shoulder_lift": -90, "elbow_flex": 90, "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Keyframe:
    time: float
    joints: Dict[str, float]
    easing: str = "linear"

@dataclass
class Animation:
    name: str
    duration: float
    arm1_keyframes: List[Keyframe] = field(default_factory=list)
    arm2_keyframes: List[Keyframe] = field(default_factory=list)

@dataclass
class SequenceStep:
    animation_name: str
    loops: int = 1

@dataclass
class Sequence:
    name: str
    steps: List[SequenceStep] = field(default_factory=list)
    loop_entire: int = 1

@dataclass
class ArmState:
    enabled: bool = False
    targets: Dict[str, float] = field(default_factory=lambda: dict(JOINT_DEFAULTS))
    current: Dict[str, float] = field(default_factory=dict)
    connected: bool = False

@dataclass 
class State:
    arms: Dict[str, ArmState] = field(default_factory=lambda: {"arm1": ArmState(), "arm2": ArmState()})
    animation: Animation = field(default_factory=lambda: Animation(name="Untitled", duration=5.0))
    playing: bool = False
    current_time: float = 0.0
    loop: bool = False
    speed: float = 1.0
    recording_active: bool = False
    recording_start: float = 0.0
    recording_rate: int = 15
    recording_arms: Dict[str, bool] = field(default_factory=lambda: {"arm1": True, "arm2": True})
    arm1_samples: List[Dict] = field(default_factory=list)
    arm2_samples: List[Dict] = field(default_factory=list)
    undo_stack: deque = field(default_factory=lambda: deque(maxlen=MAX_UNDO))
    redo_stack: deque = field(default_factory=lambda: deque(maxlen=MAX_UNDO))
    current_sequence: Sequence = field(default_factory=lambda: Sequence(name="Untitled"))
    seq_playing: bool = False
    seq_step_idx: int = 0
    seq_loop: int = 0
    seq_total_loop: int = 0
    seq_time: float = 0.0
    seq_animations: Dict[str, Animation] = field(default_factory=dict)

app = Flask(__name__)
robots: Dict[str, Optional[Any]] = {"arm1": None, "arm2": None}
state = State()
state_lock = threading.Lock()
threads_stop = threading.Event()
camera = None

# ============================================================
# EASING & INTERPOLATION
# ============================================================

def ease_linear(t): return t
def ease_in(t): return t * t
def ease_out(t): return 1 - (1 - t) ** 2
def ease_in_out(t): return 3 * t * t - 2 * t * t * t
EASING = {"linear": ease_linear, "ease-in": ease_in, "ease-out": ease_out, "ease-in-out": ease_in_out}

def interpolate(keyframes, t):
    if not keyframes: return dict(JOINT_DEFAULTS)
    kfs = sorted(keyframes, key=lambda k: k.time)
    if t <= kfs[0].time: return dict(kfs[0].joints)
    if t >= kfs[-1].time: return dict(kfs[-1].joints)
    for i in range(len(kfs) - 1):
        if kfs[i].time <= t <= kfs[i + 1].time:
            k1, k2 = kfs[i], kfs[i + 1]
            dur = k2.time - k1.time
            if dur < 0.001: return dict(k1.joints)
            p = EASING.get(k2.easing, ease_linear)((t - k1.time) / dur)
            return {j: k1.joints.get(j, 0) + (k2.joints.get(j, 0) - k1.joints.get(j, 0)) * p for j in ALL_JOINTS}
    return dict(JOINT_DEFAULTS)

# ============================================================
# PERSISTENCE
# ============================================================

def save_undo():
    with state_lock:
        state.undo_stack.append({
            "arm1": [{"time": k.time, "joints": dict(k.joints), "easing": k.easing} for k in state.animation.arm1_keyframes],
            "arm2": [{"time": k.time, "joints": dict(k.joints), "easing": k.easing} for k in state.animation.arm2_keyframes],
        })
        state.redo_stack.clear()

def save_animation(anim):
    fn = anim.name.lower().replace(" ", "_") + ".json"
    with open(ANIMATIONS_DIR / fn, 'w') as f:
        json.dump({
            "name": anim.name, "duration": anim.duration,
            "arm1_keyframes": [{"time": k.time, "joints": k.joints, "easing": k.easing} for k in anim.arm1_keyframes],
            "arm2_keyframes": [{"time": k.time, "joints": k.joints, "easing": k.easing} for k in anim.arm2_keyframes],
        }, f)
    return fn

def load_animation(fn):
    with open(ANIMATIONS_DIR / fn) as f: d = json.load(f)
    return Animation(
        name=d["name"], duration=d["duration"],
        arm1_keyframes=[Keyframe(**k) for k in d.get("arm1_keyframes", [])],
        arm2_keyframes=[Keyframe(**k) for k in d.get("arm2_keyframes", [])],
    )

def save_sequence(seq):
    fn = seq.name.lower().replace(" ", "_") + ".json"
    with open(SEQUENCES_DIR / fn, 'w') as f:
        json.dump({"name": seq.name, "loop_entire": seq.loop_entire,
                   "steps": [{"animation_name": s.animation_name, "loops": s.loops} for s in seq.steps]}, f)
    return fn

def load_sequence(fn):
    with open(SEQUENCES_DIR / fn) as f: d = json.load(f)
    return Sequence(name=d["name"], loop_entire=d.get("loop_entire", 1),
                    steps=[SequenceStep(**s) for s in d.get("steps", [])])

def simplify_samples(samples, tol=2.0):
    if len(samples) < 2: return [Keyframe(time=s["time"], joints=s["joints"], easing="linear") for s in samples]
    result = [samples[0]]
    for s in samples[1:-1]:
        if any(abs(s["joints"].get(j, 0) - result[-1]["joints"].get(j, 0)) > tol for j in ALL_JOINTS):
            result.append(s)
    result.append(samples[-1])
    return [Keyframe(time=s["time"], joints=s["joints"], easing="linear") for s in result]

# ============================================================
# CONTROL LOOPS
# ============================================================

def control_loop():
    dt = 1.0 / CONTROL_HZ
    while not threads_stop.is_set():
        t0 = time.perf_counter()
        for arm_id, robot in robots.items():
            if robot is None: continue
            try:
                obs = robot.get_observation()
                with state_lock:
                    for j in ALL_JOINTS:
                        if f"{j}.pos" in obs: state.arms[arm_id].current[j] = float(obs[f"{j}.pos"])
                    if state.recording_active or not state.arms[arm_id].enabled: continue
                    robot.send_action({f"{j}.pos": float(state.arms[arm_id].targets.get(j, 0)) for j in ALL_JOINTS})
            except Exception as e: print(f"Control error: {e}")
        time.sleep(max(0, dt - (time.perf_counter() - t0)))

def playback_loop():
    last = time.perf_counter()
    while not threads_stop.is_set():
        now = time.perf_counter()
        dt = now - last
        last = now
        with state_lock:
            if state.playing and not state.seq_playing:
                state.current_time += dt * state.speed
                if state.current_time >= state.animation.duration:
                    state.current_time = 0.0 if state.loop else state.animation.duration
                    if not state.loop: state.playing = False
                if state.playing:
                    for j, v in interpolate(state.animation.arm1_keyframes, state.current_time).items():
                        state.arms["arm1"].targets[j] = v
                    for j, v in interpolate(state.animation.arm2_keyframes, state.current_time).items():
                        state.arms["arm2"].targets[j] = v
        time.sleep(1/60)

def sequence_loop():
    last = time.perf_counter()
    while not threads_stop.is_set():
        now = time.perf_counter()
        dt = now - last
        last = now
        with state_lock:
            if not state.seq_playing or not state.current_sequence.steps:
                time.sleep(0.01); continue
            step = state.current_sequence.steps[state.seq_step_idx]
            if step.animation_name not in state.seq_animations:
                try: state.seq_animations[step.animation_name] = load_animation(step.animation_name + ".json")
                except: state.seq_step_idx += 1; continue
            anim = state.seq_animations[step.animation_name]
            state.seq_time += dt * state.speed
            if state.seq_time >= anim.duration:
                state.seq_time = 0.0
                state.seq_loop += 1
                if state.seq_loop >= step.loops:
                    state.seq_loop = 0
                    state.seq_step_idx += 1
                    if state.seq_step_idx >= len(state.current_sequence.steps):
                        state.seq_step_idx = 0
                        state.seq_total_loop += 1
                        if state.seq_total_loop >= state.current_sequence.loop_entire:
                            state.seq_playing = False; continue
            for j, v in interpolate(anim.arm1_keyframes, state.seq_time).items(): state.arms["arm1"].targets[j] = v
            for j, v in interpolate(anim.arm2_keyframes, state.seq_time).items(): state.arms["arm2"].targets[j] = v
        time.sleep(1/60)

def recording_loop():
    while not threads_stop.is_set():
        with state_lock:
            if not state.recording_active: time.sleep(0.01); continue
            t = time.perf_counter() - state.recording_start
            if state.recording_arms.get("arm1", False):
                state.arm1_samples.append({"time": t, "joints": dict(state.arms["arm1"].current)})
            if state.recording_arms.get("arm2", False):
                state.arm2_samples.append({"time": t, "joints": dict(state.arms["arm2"].current)})
        time.sleep(1.0 / state.recording_rate)

def init_robots():
    if not ROBOT_AVAILABLE: return False
    success = False
    for arm_id, cfg in ARM_CONFIGS.items():
        try:
            robot = SO100Follower(SO100FollowerConfig(port=cfg["port"], id=cfg["id"], use_degrees=USE_DEGREES))
            robot.connect()
            robots[arm_id] = robot
            obs = robot.get_observation()
            with state_lock:
                state.arms[arm_id].connected = True
                for j in ALL_JOINTS:
                    if f"{j}.pos" in obs:
                        state.arms[arm_id].current[j] = state.arms[arm_id].targets[j] = float(obs[f"{j}.pos"])
            print(f"✅ {cfg['name']} connected")
            success = True
        except Exception as e: print(f"⚠️ {cfg['name']} failed: {e}")
    return success

def get_anim_resp():
    return {
        "name": state.animation.name, "duration": state.animation.duration,
        "arm1_keyframes": [{"time": k.time, "joints": k.joints, "easing": k.easing} for k in state.animation.arm1_keyframes],
        "arm2_keyframes": [{"time": k.time, "joints": k.joints, "easing": k.easing} for k in state.animation.arm2_keyframes],
    }

# Camera
def get_camera():
    global camera
    if not CAMERA_AVAILABLE: return None
    if camera is None or not camera.isOpened():
        for dev in sorted(glob.glob("/dev/video*")):
            cap = cv2.VideoCapture(dev)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: camera = cap; break
                cap.release()
    return camera

def generate_frames():
    cam = get_camera()
    if not cam: return
    while True:
        ok, frame = cam.read()
        if not ok: continue
        frame = cv2.flip(frame, 1)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

# ============================================================
# HTML TEMPLATE
# ============================================================

HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>SO-101 Animation Editor</title>
<style>
:root{--bg:#08080c;--surface:#101018;--surface2:#181822;--border:#282838;--text:#e8e8f0;--text-dim:#606070;--arm1:#58a6ff;--arm2:#f0883e;--green:#3fb950;--red:#f85149;--purple:#a371f7;--yellow:#ffd93d;--cyan:#00d4ff}
*{margin:0;padding:0;box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;overflow:hidden}
.layout{display:grid;grid-template-columns:1fr 380px;grid-template-rows:auto 1fr auto;height:100vh}
header{grid-column:1/-1;display:flex;justify-content:space-between;align-items:center;padding:8px 16px;background:var(--surface);border-bottom:1px solid var(--border)}
h1{font-size:14px;background:linear-gradient(135deg,var(--arm1),var(--arm2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.tabs{display:flex;gap:2px;background:var(--surface2);padding:2px;border-radius:6px}
.tab{padding:5px 12px;border:none;background:transparent;color:var(--text-dim);font-size:10px;cursor:pointer;border-radius:4px}
.tab.active{background:var(--purple);color:#fff}
.main{background:var(--surface);overflow-y:auto;padding:10px}
.section{margin-bottom:12px}
.section-title{font-size:10px;color:var(--text-dim);margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
.joint-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.joint{background:var(--surface2);padding:6px;border-radius:4px}
.joint-name{font-size:8px;color:var(--text-dim)}
.joint-val{font-size:10px;color:var(--purple)}
.slider{width:100%;margin-top:2px;height:4px}
.side{background:var(--surface);border-left:1px solid var(--border);overflow-y:auto;padding:8px;font-size:10px}
.btn{padding:5px 8px;border:1px solid var(--border);border-radius:4px;background:var(--surface2);color:var(--text);font-size:9px;cursor:pointer}
.btn:hover{background:var(--border)}.btn.primary{background:var(--green);border-color:var(--green);color:#000}
.btn.danger{background:var(--red);border-color:var(--red);color:#fff}.btn.accent{background:var(--purple);color:#fff}
.btn.cyan{background:var(--cyan);color:#000}
.list{background:var(--bg);border-radius:4px;padding:4px;max-height:100px;overflow-y:auto}
.list-item{padding:5px 6px;border-radius:3px;cursor:pointer;font-size:9px;margin-bottom:2px;display:flex;justify-content:space-between}
.list-item:hover{background:var(--surface2)}
.timeline{grid-column:1/-1;background:#0a0a10;border-top:1px solid var(--border);padding:10px}
.tl-controls{display:flex;justify-content:space-between;margin-bottom:8px;align-items:center}
.play-btn{width:32px;height:32px;border-radius:50%;border:2px solid var(--green);background:transparent;color:var(--green);font-size:12px;cursor:pointer}
.play-btn.playing{background:var(--yellow);border-color:var(--yellow);color:#000}
.time-display{background:var(--surface);padding:4px 10px;border-radius:4px;font-size:11px}
.tracks{position:relative;background:var(--surface);border-radius:6px}
.track{height:28px;border-bottom:1px solid var(--border);position:relative}
.track:last-child{border-bottom:none}
.track-label{position:absolute;left:0;width:50px;height:100%;background:var(--surface2);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:600}
.track-label.arm1{color:var(--arm1)}.track-label.arm2{color:var(--arm2)}
.track-area{position:absolute;left:50px;right:0;height:100%;background:var(--bg);cursor:crosshair}
.kf-marker{position:absolute;top:50%;width:10px;height:10px;transform:translate(-50%,-50%) rotate(45deg);border-radius:2px;cursor:pointer}
.kf-marker.arm1{background:var(--arm1)}.kf-marker.arm2{background:var(--arm2)}
.kf-marker.selected{box-shadow:0 0 6px var(--yellow)}
.playhead{position:absolute;top:0;bottom:0;width:2px;background:var(--red);z-index:10}
.hidden{display:none!important}
.camera-feed{width:100%;max-width:200px;aspect-ratio:4/3;background:var(--bg);border-radius:6px;overflow:hidden;margin:8px auto}
.camera-feed img{width:100%;height:100%;object-fit:cover}
.rec-arm-select{display:flex;gap:12px;margin-bottom:8px;align-items:center}
.rec-arm-select label{display:flex;align-items:center;gap:4px;cursor:pointer;font-size:10px}
.rec-arm-select input[type="checkbox"]{accent-color:var(--purple)}
.rec-arm-cb.arm1:checked{accent-color:var(--arm1)}
.rec-arm-cb.arm2:checked{accent-color:var(--arm2)}
.rec-status{font-size:9px;color:var(--text-dim);margin-top:4px;padding:4px 6px;background:var(--bg);border-radius:4px}
.rec-status.active{color:var(--red);background:rgba(248,81,73,0.15)}
.expr-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:4px}
.expr-btn{display:flex;flex-direction:column;align-items:center;padding:6px 4px;background:var(--surface2);border:2px solid var(--border);border-radius:6px;cursor:pointer;transition:all 0.15s}
.expr-btn:hover{border-color:var(--cyan);transform:scale(1.05)}

/* Cinematic Countdown Overlay */
.countdown-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:radial-gradient(ellipse at center,rgba(8,8,12,0.95) 0%,rgba(0,0,0,0.98) 100%);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:9999;opacity:0;pointer-events:none;transition:opacity 0.3s ease}
.countdown-overlay.active{opacity:1;pointer-events:all}
.countdown-container{position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center}
.countdown-number{font-size:220px;font-weight:900;font-family:'SF Pro Display',-apple-system,system-ui,sans-serif;line-height:1;opacity:0;transform:scale(2);color:#fff;text-shadow:0 0 60px var(--cyan),0 0 120px var(--purple);position:relative;z-index:2}
.countdown-number.show{animation:countIn 0.9s cubic-bezier(0.16,1,0.3,1) forwards}
.countdown-go{font-size:140px;font-weight:900;font-family:'SF Pro Display',-apple-system,system-ui,sans-serif;background:linear-gradient(135deg,var(--green) 0%,var(--cyan) 50%,var(--green) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;opacity:0;transform:scale(0.5);text-shadow:0 0 80px var(--green)}
.countdown-go.show{animation:goIn 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards}
.countdown-label{font-size:12px;text-transform:uppercase;letter-spacing:12px;color:var(--text-dim);margin-top:30px;opacity:0.7}
.countdown-rings{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none}
.countdown-ring{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);border-radius:50%;border:2px solid;opacity:0}
.countdown-ring.r1{width:280px;height:280px;border-color:var(--cyan)}
.countdown-ring.r2{width:340px;height:340px;border-color:var(--purple)}
.countdown-ring.r3{width:400px;height:400px;border-color:var(--arm1)}
.countdown-ring.pulse{animation:ringPulse 0.9s cubic-bezier(0.16,1,0.3,1) forwards}
.countdown-glow{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:200px;height:200px;border-radius:50%;background:radial-gradient(circle,rgba(0,212,255,0.15) 0%,transparent 70%);opacity:0}
.countdown-glow.pulse{animation:glowPulse 0.9s ease-out forwards}
.countdown-particles{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:400px;height:400px}
.particle{position:absolute;width:4px;height:4px;background:var(--cyan);border-radius:50%;opacity:0}
.particle.burst{animation:particleBurst 0.8s ease-out forwards}

@keyframes countIn{
  0%{opacity:0;transform:scale(2.5);filter:blur(20px)}
  30%{opacity:1;filter:blur(0)}
  70%{opacity:1;transform:scale(1)}
  100%{opacity:0;transform:scale(0.8);filter:blur(10px)}
}
@keyframes goIn{
  0%{opacity:0;transform:scale(0.3)}
  60%{opacity:1;transform:scale(1.1)}
  100%{opacity:1;transform:scale(1)}
}
@keyframes ringPulse{
  0%{opacity:0.6;transform:translate(-50%,-50%) scale(0.5)}
  100%{opacity:0;transform:translate(-50%,-50%) scale(1.3)}
}
@keyframes glowPulse{
  0%{opacity:0.8;transform:translate(-50%,-50%) scale(0.5)}
  100%{opacity:0;transform:translate(-50%,-50%) scale(2)}
}
@keyframes particleBurst{
  0%{opacity:1;transform:translate(0,0) scale(1)}
  100%{opacity:0;transform:translate(var(--tx),var(--ty)) scale(0)}
}
</style></head><body>
<div class="layout">
<header>
<h1>🤖 SO-101 Animation Editor</h1>
<div class="tabs">
<button class="tab active" onclick="setMode('edit')">Edit</button>
<button class="tab" onclick="setMode('seq')">Sequence</button>
</div>
<div style="display:flex;gap:8px;align-items:center;font-size:10px">
<span id="arm-status">Arms: --</span>
</div>
</header>

<main class="main">
<div class="section"><div class="section-title"><span style="color:var(--arm1)">Arm 1</span><div><button class="btn" onclick="readJoints('arm1')">📖</button> <button class="btn" style="background:var(--arm1);color:#000" onclick="addKf('arm1')">+K</button></div></div>
<div class="joint-grid" id="joints-arm1"></div></div>
<div class="section"><div class="section-title"><span style="color:var(--arm2)">Arm 2</span><div><button class="btn" onclick="readJoints('arm2')">📖</button> <button class="btn" style="background:var(--arm2);color:#000" onclick="addKf('arm2')">+K</button></div></div>
<div class="joint-grid" id="joints-arm2"></div></div>
</main>

<aside class="side">
<div class="section"><div class="section-title">Control</div>
<div style="display:flex;gap:4px;flex-wrap:wrap"><button class="btn primary" onclick="api('/api/enable',{})">▶ Enable</button><button class="btn danger" onclick="stop()">■ Stop</button><button class="btn" onclick="api('/api/torque_off',{})">🔓</button></div></div>

<!-- Edit Panel -->
<div id="edit-panel">
<div class="section"><div class="section-title">Recording</div>
<div class="rec-arm-select">
<label><input type="checkbox" id="rec-arm1" class="rec-arm-cb arm1" checked><span style="color:var(--arm1)">Arm 1</span></label>
<label><input type="checkbox" id="rec-arm2" class="rec-arm-cb arm2" checked><span style="color:var(--arm2)">Arm 2</span></label>
</div>
<div style="display:flex;gap:4px"><button class="btn danger" id="rec-start" onclick="startRec()">⏺ Rec</button><button class="btn" id="rec-stop" onclick="stopRec()" disabled>⏹</button><button class="btn accent" id="rec-apply" onclick="applyRec()" disabled>✓ Apply</button></div>
<div class="rec-status" id="rec-status">Ready to record</div>
</div>
<div class="section"><div class="section-title">Keyframes</div>
<select id="easing" style="width:100%;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:4px;border-radius:4px;margin-bottom:4px;font-size:9px"><option>linear</option><option>ease-in</option><option>ease-out</option><option>ease-in-out</option></select>
<div class="list" id="kf-list"></div></div>
<div class="section"><div class="section-title">Animations <div><button class="btn" onclick="newAnim()">New</button><button class="btn accent" onclick="saveAnim()">Save</button></div></div>
<input id="anim-name" value="Untitled" style="width:100%;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:4px;border-radius:4px;margin-bottom:4px;font-size:10px">
<div class="list" id="anim-list"></div></div>
<div class="section"><div class="section-title">Camera</div>
<div class="camera-feed"><img id="camera-img" src="/api/video" onerror="this.style.display='none'" alt="Camera"></div></div>
</div>

<!-- Sequence Panel -->
<div id="seq-panel" class="hidden">
<div class="section"><div class="section-title">Sequence</div>
<input id="seq-name" value="Untitled" style="width:100%;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:4px;border-radius:4px;margin-bottom:4px;font-size:10px">
<div style="display:flex;gap:6px;align-items:center;margin-bottom:6px"><span>Loop:</span><input type="number" id="seq-loops" value="1" min="1" style="width:40px;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:2px;border-radius:4px;font-size:10px"></div>
<div style="display:flex;gap:4px"><button class="btn cyan" onclick="playSeq()">▶ Play</button><button class="btn danger" onclick="stopSeq()">■</button></div></div>
<div class="section"><div class="section-title">Steps</div>
<div class="list" id="step-list"></div>
<div style="display:flex;gap:4px;margin-top:4px"><button class="btn" onclick="moveStep(-1)">↑</button><button class="btn" onclick="moveStep(1)">↓</button><button class="btn danger" onclick="clearSteps()">Clear</button></div></div>
<div class="section"><div class="section-title">Add Animation</div>
<div class="expr-grid" id="anim-pick"></div></div>
<div class="section"><div class="section-title">Saved Sequences</div>
<div class="list" id="seq-list"></div>
<div style="display:flex;gap:4px;margin-top:4px"><button class="btn" onclick="newSeq()">New</button><button class="btn accent" onclick="saveSeq()">Save</button></div></div>
</div>
</aside>

<div class="timeline">
<div class="tl-controls">
<div style="display:flex;gap:4px;align-items:center">
<button class="btn" onclick="seek(0)">⏮</button>
<button class="play-btn" id="play-btn" onclick="togglePlay()">▶</button>
<button class="btn" onclick="seek(S.animation.duration)">⏭</button>
<div class="time-display"><span id="time-cur">0.00</span>/<span id="time-dur">5.00</span>s</div>
</div>
<div style="display:flex;gap:6px;align-items:center;font-size:9px">
<label><input type="checkbox" id="loop-cb" onchange="setLoop(this.checked)">Loop</label>
<select id="speed-sel" onchange="setSpeed(this.value)" style="background:var(--surface);border:1px solid var(--border);color:var(--text);padding:2px;border-radius:4px;font-size:9px">
<option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option>
</select>
<input type="number" id="dur-input" value="5" min="1" max="60" style="width:40px;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:2px;border-radius:4px;font-size:9px" onchange="setDuration(this.value)">s
</div>
</div>
<div class="tracks" id="tracks">
<div class="track"><div class="track-label arm1">ARM1</div><div class="track-area" id="track-arm1" onclick="clickTrack(event,'arm1')"></div></div>
<div class="track"><div class="track-label arm2">ARM2</div><div class="track-area" id="track-arm2" onclick="clickTrack(event,'arm2')"></div></div>
<div class="playhead" id="playhead"></div>
</div>
</div>
</div>

<!-- Cinematic Countdown Overlay -->
<div class="countdown-overlay" id="countdown-overlay">
  <div class="countdown-container">
    <div class="countdown-rings">
      <div class="countdown-ring r1"></div>
      <div class="countdown-ring r2"></div>
      <div class="countdown-ring r3"></div>
    </div>
    <div class="countdown-glow"></div>
    <div class="countdown-particles" id="countdown-particles"></div>
    <div class="countdown-number" id="countdown-number">3</div>
    <div class="countdown-go" id="countdown-go">GO</div>
    <div class="countdown-label">ANIMATION STARTING</div>
  </div>
</div>

<script>
const JOINTS=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];
const LIMITS={shoulder_pan:[-150,150],shoulder_lift:[-150,150],elbow_flex:[-150,150],wrist_flex:[-150,150],wrist_roll:[-150,150],gripper:[0,100]};
const DEFAULTS={shoulder_pan:0,shoulder_lift:-90,elbow_flex:90,wrist_flex:0,wrist_roll:0,gripper:50};

let S={mode:'edit',animation:{name:'Untitled',duration:5,arm1_keyframes:[],arm2_keyframes:[]},
playing:false,current_time:0,arms:{arm1:{current:{}},arm2:{current:{}}},sliders:{arm1:{...DEFAULTS},arm2:{...DEFAULTS}},
selectedKf:null,library:[],
sequence:{name:'Untitled',loop_entire:1,steps:[]},selectedStep:null,
recording:false,recordingArms:{arm1:true,arm2:true},recordedSamples:{arm1:0,arm2:0},
countdownActive:false};

// ============================================================
// CINEMATIC COUNTDOWN
// ============================================================

function createParticles() {
  const container = document.getElementById('countdown-particles');
  container.innerHTML = '';
  for (let i = 0; i < 20; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    const angle = (i / 20) * Math.PI * 2;
    const distance = 150 + Math.random() * 100;
    particle.style.setProperty('--tx', `${Math.cos(angle) * distance}px`);
    particle.style.setProperty('--ty', `${Math.sin(angle) * distance}px`);
    particle.style.left = '50%';
    particle.style.top = '50%';
    particle.style.background = i % 2 === 0 ? 'var(--cyan)' : 'var(--purple)';
    container.appendChild(particle);
  }
}

function triggerEffects() {
  // Rings
  document.querySelectorAll('.countdown-ring').forEach((ring, i) => {
    ring.classList.remove('pulse');
    void ring.offsetWidth;
    setTimeout(() => ring.classList.add('pulse'), i * 50);
  });
  
  // Glow
  const glow = document.querySelector('.countdown-glow');
  glow.classList.remove('pulse');
  void glow.offsetWidth;
  glow.classList.add('pulse');
  
  // Particles
  createParticles();
  document.querySelectorAll('.particle').forEach((p, i) => {
    setTimeout(() => p.classList.add('burst'), i * 20);
  });
}

function showCountdown(onComplete) {
  if (S.countdownActive) return;
  S.countdownActive = true;
  
  const overlay = document.getElementById('countdown-overlay');
  const numEl = document.getElementById('countdown-number');
  const goEl = document.getElementById('countdown-go');
  
  overlay.classList.add('active');
  goEl.classList.remove('show');
  numEl.classList.remove('show');
  
  const counts = [3, 2, 1];
  let idx = 0;
  
  function showNext() {
    if (idx < counts.length) {
      numEl.textContent = counts[idx];
      numEl.classList.remove('show');
      void numEl.offsetWidth;
      numEl.classList.add('show');
      triggerEffects();
      idx++;
      setTimeout(showNext, 900);
    } else {
      // Show GO!
      numEl.style.display = 'none';
      goEl.classList.add('show');
      triggerEffects();
      
      setTimeout(() => {
        overlay.classList.remove('active');
        numEl.style.display = '';
        S.countdownActive = false;
        if (onComplete) onComplete();
      }, 600);
    }
  }
  
  setTimeout(showNext, 100);
}

function setMode(m){S.mode=m;document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',['edit','seq'].indexOf(m)===i));
document.getElementById('edit-panel').classList.toggle('hidden',m!=='edit');
document.getElementById('seq-panel').classList.toggle('hidden',m!=='seq');
if(m==='seq')loadAnimPick()}

function buildSliders(){['arm1','arm2'].forEach(arm=>{const c=document.getElementById('joints-'+arm);c.innerHTML=JOINTS.map(j=>
`<div class="joint"><div style="display:flex;justify-content:space-between"><span class="joint-name">${j.split('_')[0]}</span><span class="joint-val" id="v-${arm}-${j}">${DEFAULTS[j]}</span></div>
<input type="range" class="slider" id="s-${arm}-${j}" min="${LIMITS[j][0]}" max="${LIMITS[j][1]}" value="${DEFAULTS[j]}" oninput="slide('${arm}','${j}',this.value)"></div>`).join('')})}

function slide(arm,j,v){S.sliders[arm][j]=+v;document.getElementById('v-'+arm+'-'+j).textContent=(+v).toFixed(0);if(!S.playing)api('/api/targets',{arm,targets:S.sliders[arm]})}
function readJoints(arm){JOINTS.forEach(j=>{const v=S.arms[arm]?.current[j];if(v!==undefined){S.sliders[arm][j]=v;document.getElementById('s-'+arm+'-'+j).value=v;document.getElementById('v-'+arm+'-'+j).textContent=v.toFixed(0)}})}

async function api(url,data){return(await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})).json()}

async function addKf(arm,t){if(t===undefined)t=S.current_time;const d=await api('/api/keyframe/add',{arm,time:t,joints:S.sliders[arm],easing:document.getElementById('easing').value});if(d.ok){S.animation=d.animation;render()}}
async function deleteKf(arm,i){const d=await api('/api/keyframe/delete',{arm,index:i});if(d.ok){S.animation=d.animation;render()}}

function selectKf(arm,i){S.selectedKf={arm,idx:i};const kf=(arm==='arm1'?S.animation.arm1_keyframes:S.animation.arm2_keyframes)[i];
if(kf){JOINTS.forEach(j=>{S.sliders[arm][j]=kf.joints[j]||0;document.getElementById('s-'+arm+'-'+j).value=kf.joints[j]||0;document.getElementById('v-'+arm+'-'+j).textContent=(kf.joints[j]||0).toFixed(0)});S.current_time=kf.time;updatePlayhead()}render()}

function render(){renderKfList();renderTimeline()}
function renderKfList(){const l=document.getElementById('kf-list');const all=[...S.animation.arm1_keyframes.map((k,i)=>({...k,arm:'arm1',idx:i})),...S.animation.arm2_keyframes.map((k,i)=>({...k,arm:'arm2',idx:i}))].sort((a,b)=>a.time-b.time);
l.innerHTML=all.length?all.map(k=>`<div class="list-item" style="border-left:3px solid var(--${k.arm})" onclick="selectKf('${k.arm}',${k.idx})"><span>${k.time.toFixed(2)}s</span><button onclick="event.stopPropagation();deleteKf('${k.arm}',${k.idx})" style="background:none;border:none;color:var(--red);cursor:pointer">×</button></div>`).join(''):'<div style="color:var(--text-dim);text-align:center;padding:8px">No keyframes</div>'}

function renderTimeline(){['arm1','arm2'].forEach(arm=>{const t=document.getElementById('track-'+arm);t.querySelectorAll('.kf-marker').forEach(m=>m.remove());
const kfs=arm==='arm1'?S.animation.arm1_keyframes:S.animation.arm2_keyframes;kfs.forEach((k,i)=>{const m=document.createElement('div');m.className='kf-marker '+arm+(S.selectedKf?.arm===arm&&S.selectedKf?.idx===i?' selected':'');m.style.left=(k.time/S.animation.duration*100)+'%';m.onclick=e=>{e.stopPropagation();selectKf(arm,i)};t.appendChild(m)})});
document.getElementById('time-dur').textContent=S.animation.duration.toFixed(2);updatePlayhead()}

function updatePlayhead(){const p=(S.current_time/S.animation.duration)*100;document.getElementById('playhead').style.left=`calc(50px + ${p}% * (100% - 50px) / 100)`;document.getElementById('time-cur').textContent=S.current_time.toFixed(2)}
function clickTrack(e,type){const r=e.target.getBoundingClientRect();const t=(e.clientX-r.left)/r.width*S.animation.duration;const time=Math.max(0,Math.min(t,S.animation.duration));addKf(type,time)}

async function togglePlay(){
  if(S.playing){
    // Stop immediately
    S.playing=false;
    document.getElementById('play-btn').textContent='▶';
    document.getElementById('play-btn').classList.remove('playing');
    await api('/api/playback/toggle',{playing:false});
  } else {
    // Show countdown then play
    showCountdown(async () => {
      S.playing=true;
      document.getElementById('play-btn').textContent='⏸';
      document.getElementById('play-btn').classList.add('playing');
      await api('/api/playback/toggle',{playing:true});
      await api('/api/enable',{});
    });
  }
}

async function seek(t){S.current_time=t;updatePlayhead();await api('/api/playback/seek',{time:t})}
async function setLoop(v){await api('/api/playback/loop',{loop:v})}
async function setSpeed(v){await api('/api/playback/speed',{speed:+v})}
async function setDuration(v){S.animation.duration=+v;await api('/api/animation/duration',{duration:+v});render()}
async function stop(){S.playing=false;document.getElementById('play-btn').textContent='▶';document.getElementById('play-btn').classList.remove('playing');await api('/api/disable',{})}

function getRecordingArms(){return {arm1:document.getElementById('rec-arm1').checked,arm2:document.getElementById('rec-arm2').checked}}
function updateRecStatus(text,active=false){const el=document.getElementById('rec-status');el.textContent=text;el.classList.toggle('active',active)}

async function startRec(){
  const arms=getRecordingArms();
  if(!arms.arm1&&!arms.arm2){alert('Select at least one arm to record');return}
  S.recordingArms=arms;
  await api('/api/torque_off',{});
  await api('/api/recording/start',{arms});
  S.recording=true;
  document.getElementById('rec-start').disabled=true;
  document.getElementById('rec-stop').disabled=false;
  document.getElementById('rec-arm1').disabled=true;
  document.getElementById('rec-arm2').disabled=true;
  const armList=[];if(arms.arm1)armList.push('Arm 1');if(arms.arm2)armList.push('Arm 2');
  updateRecStatus('⏺ Recording '+armList.join(' + ')+'...',true);
}

async function stopRec(){
  const d=await api('/api/recording/stop',{});
  S.recording=false;
  S.recordedSamples=d.samples||{arm1:0,arm2:0};
  document.getElementById('rec-start').disabled=false;
  document.getElementById('rec-stop').disabled=true;
  document.getElementById('rec-apply').disabled=false;
  document.getElementById('rec-arm1').disabled=false;
  document.getElementById('rec-arm2').disabled=false;
  const parts=[];
  if(S.recordingArms.arm1&&S.recordedSamples.arm1>0)parts.push(`Arm1: ${S.recordedSamples.arm1}`);
  if(S.recordingArms.arm2&&S.recordedSamples.arm2>0)parts.push(`Arm2: ${S.recordedSamples.arm2}`);
  updateRecStatus('Recorded: '+(parts.length?parts.join(', ')+' samples':'no samples'));
}

async function applyRec(){
  const d=await api('/api/recording/apply',{simplify:true,arms:S.recordingArms});
  if(d.ok){S.animation=d.animation;document.getElementById('rec-apply').disabled=true;render();
  const parts=[];if(S.recordingArms.arm1)parts.push('Arm 1');if(S.recordingArms.arm2)parts.push('Arm 2');
  updateRecStatus('✓ Applied to '+parts.join(' + '))}
}

async function loadAnims(){const d=await fetch('/api/animations/list').then(r=>r.json());S.library=d.animations||[];document.getElementById('anim-list').innerHTML=S.library.length?S.library.map(n=>`<div class="list-item" onclick="loadAnim('${n}')">${n.replace('.json','')}</div>`).join(''):'<div style="color:var(--text-dim);text-align:center;padding:8px">No animations</div>'}
async function loadAnim(fn){const d=await api('/api/animation/load',{filename:fn});if(d.ok){S.animation=d.animation;document.getElementById('anim-name').value=S.animation.name;render()}}
async function saveAnim(){S.animation.name=document.getElementById('anim-name').value;await api('/api/animation/save',{animation:S.animation});loadAnims()}
async function newAnim(){S.animation={name:'Untitled',duration:5,arm1_keyframes:[],arm2_keyframes:[]};S.current_time=0;document.getElementById('anim-name').value='Untitled';await api('/api/animation/new',{});render()}

// Sequences
function loadAnimPick(){const g=document.getElementById('anim-pick');g.innerHTML=S.library?.length?S.library.map(n=>`<div class="expr-btn" onclick="addStep('${n.replace('.json','')}')" style="padding:6px"><span style="font-size:12px">📁</span><span style="font-size:8px;color:var(--text-dim);margin-top:2px">${n.replace('.json','')}</span></div>`).join(''):'<div style="grid-column:1/-1;color:var(--text-dim);text-align:center;padding:8px">No animations</div>'}
function addStep(name){S.sequence.steps.push({animation_name:name,loops:1});renderSteps()}
function renderSteps(){const l=document.getElementById('step-list');
l.innerHTML=S.sequence.steps.length?S.sequence.steps.map((s,i)=>`<div class="list-item" onclick="S.selectedStep=${i};renderSteps()" style="border-left:3px solid var(--cyan)"><span>${i+1}. ${s.animation_name}</span><input type="number" value="${s.loops}" min="1" style="width:30px;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:2px;margin:0 4px" onchange="S.sequence.steps[${i}].loops=+this.value" onclick="event.stopPropagation()"><button onclick="event.stopPropagation();S.sequence.steps.splice(${i},1);renderSteps()" style="background:none;border:none;color:var(--red);cursor:pointer">×</button></div>`).join(''):'<div style="color:var(--text-dim);text-align:center;padding:8px">Add animations</div>'}
function moveStep(dir){if(S.selectedStep===null)return;const i=S.selectedStep,j=i+dir;if(j<0||j>=S.sequence.steps.length)return;[S.sequence.steps[i],S.sequence.steps[j]]=[S.sequence.steps[j],S.sequence.steps[i]];S.selectedStep=j;renderSteps()}
function clearSteps(){S.sequence.steps=[];S.selectedStep=null;renderSteps()}

async function playSeq(){
  S.sequence.name=document.getElementById('seq-name').value;
  S.sequence.loop_entire=+document.getElementById('seq-loops').value;
  if(!S.sequence.steps.length)return alert('Add animations');
  
  // Show countdown then play sequence
  showCountdown(async () => {
    await api('/api/sequence/play',{sequence:S.sequence});
  });
}

async function stopSeq(){await api('/api/sequence/stop',{})}
async function loadSeqs(){const d=await fetch('/api/sequences/list').then(r=>r.json());document.getElementById('seq-list').innerHTML=d.sequences?.length?d.sequences.map(n=>`<div class="list-item" onclick="loadSeq('${n}')">${n.replace('.json','')}</div>`).join(''):'<div style="color:var(--text-dim);text-align:center;padding:8px">No sequences</div>'}
async function loadSeq(fn){const d=await api('/api/sequence/load',{filename:fn});if(d.ok){S.sequence=d.sequence;document.getElementById('seq-name').value=S.sequence.name;document.getElementById('seq-loops').value=S.sequence.loop_entire;renderSteps()}}
async function saveSeq(){S.sequence.name=document.getElementById('seq-name').value;S.sequence.loop_entire=+document.getElementById('seq-loops').value;await api('/api/sequence/save',{sequence:S.sequence});loadSeqs()}
function newSeq(){S.sequence={name:'Untitled',loop_entire:1,steps:[]};document.getElementById('seq-name').value='Untitled';document.getElementById('seq-loops').value=1;S.selectedStep=null;renderSteps()}

async function poll(){const d=await fetch('/api/status').then(r=>r.json()).catch(()=>null);if(!d)return;
S.arms=d.arms||{};S.library=d.library||[];if(d.playing)S.current_time=d.current_time;if(d.playing||S.playing)updatePlayhead();
const ac=[d.arms?.arm1?.connected,d.arms?.arm2?.connected].filter(Boolean).length;
document.getElementById('arm-status').textContent='Arms:'+ac+'/2'}

buildSliders();render();loadAnims();loadSeqs();setInterval(poll,150);
</script></body></html>'''

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/video")
def video():
    if CAMERA_AVAILABLE:
        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    return "Camera not available", 503

@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({
            "arms": {a: {"current": s.current, "enabled": s.enabled, "connected": s.connected} for a, s in state.arms.items()},
            "playing": state.playing, "current_time": state.current_time,
            "library": [str(f.name) for f in ANIMATIONS_DIR.glob("*.json")],
            "seq_playing": state.seq_playing,
            "recording_active": state.recording_active,
            "recording_arms": state.recording_arms
        })

@app.route("/api/enable", methods=["POST"])
def api_enable():
    with state_lock: state.arms["arm1"].enabled = state.arms["arm2"].enabled = True
    return jsonify({"ok": True})

@app.route("/api/disable", methods=["POST"])
def api_disable():
    with state_lock: 
        state.arms["arm1"].enabled = state.arms["arm2"].enabled = False
        state.playing = False; state.seq_playing = False
    return jsonify({"ok": True})

@app.route("/api/torque_off", methods=["POST"])
def api_torque_off():
    with state_lock: 
        state.arms["arm1"].enabled = state.arms["arm2"].enabled = False
        state.playing = False; state.seq_playing = False
    for r in robots.values():
        if r:
            try: r.bus.disable_torque()
            except: pass
    return jsonify({"ok": True})

@app.route("/api/targets", methods=["POST"])
def api_targets():
    d = request.get_json() or {}
    with state_lock:
        for j, v in d.get("targets", {}).items():
            if j in ALL_JOINTS: state.arms[d.get("arm", "arm1")].targets[j] = float(v)
    return jsonify({"ok": True})

@app.route("/api/keyframe/add", methods=["POST"])
def api_kf_add():
    d = request.get_json() or {}
    save_undo()
    arm = d.get("arm", "arm1")
    kf = Keyframe(time=float(d.get("time", 0)), joints=d.get("joints", {}), easing=d.get("easing", "linear"))
    with state_lock:
        (state.animation.arm1_keyframes if arm == "arm1" else state.animation.arm2_keyframes).append(kf)
        return jsonify({"ok": True, "animation": get_anim_resp()})

@app.route("/api/keyframe/delete", methods=["POST"])
def api_kf_del():
    d = request.get_json() or {}
    save_undo()
    arm, idx = d.get("arm", "arm1"), int(d.get("index", 0))
    with state_lock:
        kfs = state.animation.arm1_keyframes if arm == "arm1" else state.animation.arm2_keyframes
        if 0 <= idx < len(kfs): kfs.pop(idx)
        return jsonify({"ok": True, "animation": get_anim_resp()})

@app.route("/api/playback/toggle", methods=["POST"])
def api_play_toggle():
    with state_lock: state.playing = request.get_json().get("playing", False)
    return jsonify({"ok": True})

@app.route("/api/playback/seek", methods=["POST"])
def api_seek():
    with state_lock:
        state.current_time = float(request.get_json().get("time", 0))
        for j, v in interpolate(state.animation.arm1_keyframes, state.current_time).items(): 
            state.arms["arm1"].targets[j] = v
        for j, v in interpolate(state.animation.arm2_keyframes, state.current_time).items(): 
            state.arms["arm2"].targets[j] = v
    return jsonify({"ok": True})

@app.route("/api/playback/loop", methods=["POST"])
def api_loop():
    with state_lock: state.loop = request.get_json().get("loop", False)
    return jsonify({"ok": True})

@app.route("/api/playback/speed", methods=["POST"])
def api_speed():
    with state_lock: state.speed = float(request.get_json().get("speed", 1))
    return jsonify({"ok": True})

@app.route("/api/animation/new", methods=["POST"])
def api_anim_new():
    with state_lock: 
        state.animation = Animation(name="Untitled", duration=5.0)
        state.current_time = 0
    return jsonify({"ok": True})

@app.route("/api/animation/save", methods=["POST"])
def api_anim_save():
    d = request.get_json().get("animation", {})
    with state_lock:
        state.animation.name = d.get("name", "Untitled")
        return jsonify({"ok": True, "filename": save_animation(state.animation)})

@app.route("/api/animation/load", methods=["POST"])
def api_anim_load():
    try:
        anim = load_animation(request.get_json().get("filename", ""))
        with state_lock: state.animation = anim; state.current_time = 0
        return jsonify({"ok": True, "animation": get_anim_resp()})
    except Exception as e: return jsonify({"ok": False, "error": str(e)})

@app.route("/api/animation/duration", methods=["POST"])
def api_dur():
    with state_lock: state.animation.duration = float(request.get_json().get("duration", 5))
    return jsonify({"ok": True})

@app.route("/api/animations/list")
def api_anims_list():
    return jsonify({"animations": [f.name for f in ANIMATIONS_DIR.glob("*.json")]})

@app.route("/api/recording/start", methods=["POST"])
def api_rec_start():
    d = request.get_json() or {}
    arms = d.get("arms", {"arm1": True, "arm2": True})
    with state_lock: 
        state.recording_active = True
        state.recording_start = time.perf_counter()
        state.recording_arms = {"arm1": arms.get("arm1", True), "arm2": arms.get("arm2", True)}
        state.arm1_samples = []
        state.arm2_samples = []
    return jsonify({"ok": True, "arms": state.recording_arms})

@app.route("/api/recording/stop", methods=["POST"])
def api_rec_stop():
    with state_lock: 
        state.recording_active = False
        samples = {"arm1": len(state.arm1_samples), "arm2": len(state.arm2_samples)}
    return jsonify({"ok": True, "samples": samples})

@app.route("/api/recording/apply", methods=["POST"])
def api_rec_apply():
    d = request.get_json() or {}
    arms = d.get("arms", {"arm1": True, "arm2": True})
    save_undo()
    with state_lock:
        if arms.get("arm1", True) and state.arm1_samples:
            state.animation.arm1_keyframes = simplify_samples(state.arm1_samples)
        if arms.get("arm2", True) and state.arm2_samples:
            state.animation.arm2_keyframes = simplify_samples(state.arm2_samples)
        
        max_time = 0
        if arms.get("arm1", True) and state.arm1_samples:
            max_time = max(max_time, state.arm1_samples[-1]["time"])
        if arms.get("arm2", True) and state.arm2_samples:
            max_time = max(max_time, state.arm2_samples[-1]["time"])
        if max_time > 0:
            state.animation.duration = max(state.animation.duration, max_time + 0.5)
        
        return jsonify({"ok": True, "animation": get_anim_resp()})

@app.route("/api/sequences/list")
def api_seqs_list():
    return jsonify({"sequences": [f.name for f in SEQUENCES_DIR.glob("*.json")]})

@app.route("/api/sequence/save", methods=["POST"])
def api_seq_save():
    d = request.get_json().get("sequence", {})
    seq = Sequence(name=d.get("name", "Untitled"), loop_entire=d.get("loop_entire", 1),
                   steps=[SequenceStep(**s) for s in d.get("steps", [])])
    return jsonify({"ok": True, "filename": save_sequence(seq)})

@app.route("/api/sequence/load", methods=["POST"])
def api_seq_load():
    try:
        seq = load_sequence(request.get_json().get("filename", ""))
        return jsonify({"ok": True, "sequence": {"name": seq.name, "loop_entire": seq.loop_entire,
                        "steps": [{"animation_name": s.animation_name, "loops": s.loops} for s in seq.steps]}})
    except Exception as e: return jsonify({"ok": False, "error": str(e)})

@app.route("/api/sequence/play", methods=["POST"])
def api_seq_play():
    d = request.get_json().get("sequence", {})
    with state_lock:
        state.current_sequence = Sequence(name=d.get("name", ""), loop_entire=d.get("loop_entire", 1),
                                          steps=[SequenceStep(**s) for s in d.get("steps", [])])
        state.seq_playing = True; state.seq_step_idx = 0; state.seq_loop = 0
        state.seq_total_loop = 0; state.seq_time = 0; state.seq_animations = {}
        state.arms["arm1"].enabled = state.arms["arm2"].enabled = True
    return jsonify({"ok": True})

@app.route("/api/sequence/stop", methods=["POST"])
def api_seq_stop():
    with state_lock: state.seq_playing = False
    return jsonify({"ok": True})

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SO-101 Animation Editor")
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("🤖 SO-101 Animation Editor")
    print("="*50)
    
    robot_ok = init_robots()
    if robot_ok:
        for t in [control_loop, playback_loop, sequence_loop, recording_loop]:
            threading.Thread(target=t, daemon=True).start()
    
    print(f"\n🌐 Dashboard: http://0.0.0.0:{args.port}")
    print("\n📋 Features:")
    print("   • Edit: Arm keyframes + Recording (selective arm)")
    print("   • Sequence: Chain animations with loops")
    print("   • Cinematic 3-2-1 countdown before playback")
    print()
    
    app.run(host="0.0.0.0", port=args.port, threaded=True)

if __name__ == "__main__":
    main()