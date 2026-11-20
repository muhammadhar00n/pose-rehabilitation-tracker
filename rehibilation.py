import cv2
import time
import json
import math
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, asdict

# --------------------------
# Utilities: geometry
# --------------------------
def to_xy(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)

def angle_3pt(a, b, c):
    """
    Angle at point b formed by points a-b-c, in degrees.
    """
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def exp_smooth(prev, cur, alpha=0.25):
    return cur if prev is None else (alpha * cur + (1 - alpha) * prev)

# --------------------------
# Rehab logic: exercise FSMs
# --------------------------
@dataclass
class RepMetrics:
    reps: int = 0
    best_rom_deg: float = 0.0
    last_rep_time_s: float = 0.0
    quality_scores: list = None

    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = []

class ExerciseBase:
    name = "base"

    def reset(self):
        self.metrics = RepMetrics()
        self.state = "idle"
        self.last_phase_change = time.time()

    def update(self, landmarks, frame_w, frame_h):
        """
        Returns: (overlay_lines: list[str], debug: dict)
        """
        raise NotImplementedError

class KneeExtensionExercise(ExerciseBase):
    """
    Seated knee extension tracking (one leg):
    - Track knee angle: hip-knee-ankle.
    - Rep: from bent -> extended -> bent.
    """
    name = "knee_extension"

    def __init__(self, side="right"):
        self.side = side
        self.reset()
        self.smoothed_angle = None

        # Thresholds (tune per patient)
        self.bent_thresh = 120.0      # more bent (smaller extension)
        self.extended_thresh = 165.0  # near full extension
        self.min_rep_interval = 0.7   # seconds

    def update(self, lm, w, h):
        mp_pose = mp.solutions.pose.PoseLandmark
        if self.side == "right":
            hip_i, knee_i, ankle_i = mp_pose.RIGHT_HIP, mp_pose.RIGHT_KNEE, mp_pose.RIGHT_ANKLE
        else:
            hip_i, knee_i, ankle_i = mp_pose.LEFT_HIP, mp_pose.LEFT_KNEE, mp_pose.LEFT_ANKLE

        hip = to_xy(lm[hip_i], w, h)
        knee = to_xy(lm[knee_i], w, h)
        ankle = to_xy(lm[ankle_i], w, h)

        ang = angle_3pt(hip, knee, ankle)
        self.smoothed_angle = exp_smooth(self.smoothed_angle, ang, alpha=0.30)

        overlay = [f"Exercise: Knee Extension ({self.side})",
                   f"Knee angle: {self.smoothed_angle:.1f} deg",
                   f"Reps: {self.metrics.reps}",
                   f"Best ROM: {self.metrics.best_rom_deg:.1f} deg"]

        # ROM tracking (max extension angle achieved)
        self.metrics.best_rom_deg = max(self.metrics.best_rom_deg, self.smoothed_angle)

        now = time.time()
        # Simple FSM: bent -> extended -> bent counts rep at return to bent
        if self.state in ("idle", "bent"):
            if self.smoothed_angle >= self.extended_thresh:
                self.state = "extended"
                self.last_phase_change = now
        elif self.state == "extended":
            if self.smoothed_angle <= self.bent_thresh:
                # Rep complete
                if now - self.metrics.last_rep_time_s >= self.min_rep_interval:
                    self.metrics.reps += 1
                    self.metrics.last_rep_time_s = now

                    # Simple quality score: did they reach extension threshold?
                    # Higher = better. Clip 0..1.
                    quality = np.clip((self.metrics.best_rom_deg - self.bent_thresh) /
                                      (180.0 - self.bent_thresh), 0.0, 1.0)
                    self.metrics.quality_scores.append(float(quality))

                self.state = "bent"
                self.last_phase_change = now

        # Feedback cues
        cues = []
        if self.smoothed_angle < self.extended_thresh - 5:
            cues.append("Try extending a bit more (increase ROM).")
        if (now - self.last_phase_change) < 0.25 and self.state == "extended":
            cues.append("Hold briefly at the top for control.")
        if not cues:
            cues.append("Good control. Keep steady tempo.")

        overlay.extend([f"Feedback: {cues[0]}"] + [f"- {c}" for c in cues[1:]])
        debug = {"knee_angle": self.smoothed_angle, "state": self.state}
        return overlay, debug

class SquatExercise(ExerciseBase):
    """
    Basic squat tracking:
    - Knee angle (hip-knee-ankle)
    - Torso lean (shoulder-hip-knee alignment proxy)
    - Knee alignment (knee relative to ankle in x direction)
    Rep: standing -> down -> standing
    """
    name = "squat"

    def __init__(self, side="right"):
        self.side = side
        self.reset()
        self.sm_knee = None
        self.sm_hip = None

        self.stand_thresh = 165.0
        self.down_thresh = 110.0
        self.min_rep_interval = 0.9

    def update(self, lm, w, h):
        mp_pose = mp.solutions.pose.PoseLandmark
        if self.side == "right":
            hip_i, knee_i, ankle_i = mp_pose.RIGHT_HIP, mp_pose.RIGHT_KNEE, mp_pose.RIGHT_ANKLE
            sh_i = mp_pose.RIGHT_SHOULDER
        else:
            hip_i, knee_i, ankle_i = mp_pose.LEFT_HIP, mp_pose.LEFT_KNEE, mp_pose.LEFT_ANKLE
            sh_i = mp_pose.LEFT_SHOULDER

        hip = to_xy(lm[hip_i], w, h)
        knee = to_xy(lm[knee_i], w, h)
        ankle = to_xy(lm[ankle_i], w, h)
        shoulder = to_xy(lm[sh_i], w, h)

        knee_ang = angle_3pt(hip, knee, ankle)  # knee flexion proxy
        hip_ang = angle_3pt(shoulder, hip, knee)  # hip angle proxy for torso/hip hinge

        self.sm_knee = exp_smooth(self.sm_knee, knee_ang, alpha=0.25)
        self.sm_hip = exp_smooth(self.sm_hip, hip_ang, alpha=0.25)

        overlay = [f"Exercise: Squat ({self.side} side view-ish)",
                   f"Knee angle: {self.sm_knee:.1f} deg",
                   f"Hip angle: {self.sm_hip:.1f} deg",
                   f"Reps: {self.metrics.reps}"]

        # Knee alignment cue: knee x should not drift too far inward vs ankle x (rough 2D heuristic)
        knee_to_ankle_x = float(knee[0] - ankle[0])

        now = time.time()
        # FSM
        if self.state in ("idle", "standing"):
            if self.sm_knee <= self.down_thresh:
                self.state = "down"
                self.last_phase_change = now
        elif self.state == "down":
            if self.sm_knee >= self.stand_thresh:
                if now - self.metrics.last_rep_time_s >= self.min_rep_interval:
                    self.metrics.reps += 1
                    self.metrics.last_rep_time_s = now
                self.state = "standing"
                self.last_phase_change = now

        # Feedback
        cues = []
        if self.sm_knee > self.down_thresh + 10 and self.state == "down":
            cues.append("Go a bit deeper if comfortable (more knee bend).")

        # Torso lean heuristic: very small hip angle can indicate excessive forward lean in some camera setups
        if self.sm_hip < 70:
            cues.append("Keep chest up (reduce forward lean).")

        # Knee tracking heuristic (camera dependent)
        if abs(knee_to_ankle_x) > 60:
            cues.append("Keep knee stacked over ankle (avoid drifting).")

        if not cues:
            cues.append("Nice form. Smooth and controlled.")

        overlay.append(f"Feedback: {cues[0]}")
        debug = {"knee_angle": self.sm_knee, "hip_angle": self.sm_hip, "state": self.state}
        return overlay, debug

# --------------------------
# Session logger
# --------------------------
class SessionLogger:
    def __init__(self):
        self.start_ts = time.time()
        self.events = []

    def log_frame(self, exercise_name, debug):
        self.events.append({
            "t": time.time() - self.start_ts,
            "exercise": exercise_name,
            **debug
        })

    def export_json(self, path="rehab_session.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"start": self.start_ts, "events": self.events}, f, indent=2)

# --------------------------
# Main app
# --------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    drawer = mp.solutions.drawing_utils
    style = mp.solutions.drawing_styles

    exercises = [
        KneeExtensionExercise(side="right"),
        SquatExercise(side="right"),
    ]
    ex_idx = 0
    current_ex = exercises[ex_idx]

    logger = SessionLogger()

    print("Controls:")
    print("  [1] Knee Extension  [2] Squat")
    print("  [r] Reset counters  [s] Save session JSON  [q] Quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            # Draw skeleton
            drawer.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=style.get_default_pose_landmarks_style()
            )

            overlay_lines, debug = current_ex.update(lm, w, h)
            logger.log_frame(current_ex.name, debug)

            # Overlay text
            y = 25
            for line in overlay_lines:
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y += 22
        else:
            cv2.putText(frame, "No pose detected - ensure full body is visible",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Rehab Monitor (MediaPipe)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            ex_idx = 0
            current_ex = exercises[ex_idx]
        elif key == ord('2'):
            ex_idx = 1
            current_ex = exercises[ex_idx]
        elif key == ord('r'):
            current_ex.reset()
        elif key == ord('s'):
            logger.export_json("rehab_session.json")
            print("Saved rehab_session.json")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
