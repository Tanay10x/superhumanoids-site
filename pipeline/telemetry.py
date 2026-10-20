import json, sys, mujoco
sys.path.insert(0, "/home/tanay/sh-build/tools")
import motion
m, frames, _ = motion.humanoid(90)
names = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint"]
out = {"fps": 30, "joints": {n: [round(float(q[motion.jadr(m, n)]), 3) for q in frames] for n in names}}
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
