// Rep counting and yoga timing, ported from workout_func.py and yoga_function.py.
// Thresholds are unchanged so the web app counts the same way as the desktop app.
// This file has no browser dependencies, so it can also be tested in Node.

export const EXERCISES = ["Push Up", "Curl Up", "Jumping Jack", "Squat"];
export const YOGA_POSES = ["Goddess", "Plank", "Tree", "Warrior II"]; // index = model class id
export const YOGA_CONFIDENCE = 0.8; // a pose's timer only starts above this confidence

// MediaPipe Pose landmark indices
const NOSE = 0, LEFT_MOUTH = 9, RIGHT_MOUTH = 10;
const LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12, LEFT_ELBOW = 13, RIGHT_ELBOW = 14;
const LEFT_WRIST = 15, RIGHT_WRIST = 16, LEFT_HIP = 23, RIGHT_HIP = 24;
const LEFT_KNEE = 25, RIGHT_KNEE = 26, LEFT_ANKLE = 27, RIGHT_ANKLE = 28;

export function calculateAngle(a, b, c) {
  const radians = Math.atan2(c[1] - b[1], c[0] - b[0]) - Math.atan2(a[1] - b[1], a[0] - b[0]);
  const angle = Math.abs((radians * 180) / Math.PI);
  return angle > 180 ? 360 - angle : angle;
}

export class WorkoutCounter {
  constructor() { this.reset(); }

  reset() {
    this.counts = Object.fromEntries(EXERCISES.map((e) => [e, 0]));
    this.stages = Object.fromEntries(EXERCISES.map((e) => [e, ""]));
  }

  /** landmarks: array of 33 {x, y} in normalized image coordinates. */
  update(landmarks, exercise) {
    const pt = (i) => [landmarks[i].x, landmarks[i].y];
    const nose = pt(NOSE), mouthLeft = pt(LEFT_MOUTH), mouthRight = pt(RIGHT_MOUTH);
    const wristLeft = pt(LEFT_WRIST), wristRight = pt(RIGHT_WRIST);
    const shoulderLeft = pt(LEFT_SHOULDER), shoulderRight = pt(RIGHT_SHOULDER);
    const elbowLeft = pt(LEFT_ELBOW), elbowRight = pt(RIGHT_ELBOW);
    const hipLeft = pt(LEFT_HIP), hipRight = pt(RIGHT_HIP);
    const kneeLeft = pt(LEFT_KNEE), kneeRight = pt(RIGHT_KNEE);
    const ankleLeft = pt(LEFT_ANKLE), ankleRight = pt(RIGHT_ANKLE);

    const hipKneeAnkle = (calculateAngle(hipLeft, kneeLeft, ankleLeft) + calculateAngle(hipRight, kneeRight, ankleRight)) / 2;
    const shoulderElbowWrist = (calculateAngle(shoulderLeft, elbowLeft, wristLeft) + calculateAngle(shoulderRight, elbowRight, wristRight)) / 2;
    const elbowY = (elbowLeft[1] + elbowRight[1]) / 2;
    const elbowShoulderHip = (calculateAngle(elbowLeft, shoulderLeft, hipLeft) + calculateAngle(elbowRight, shoulderRight, hipRight)) / 2;
    const shoulderHipKnee = (calculateAngle(shoulderLeft, hipLeft, kneeLeft) + calculateAngle(shoulderRight, hipRight, kneeRight)) / 2;
    const anklesApart = Math.abs(ankleRight[0] - ankleLeft[0]) > 0.1;

    let stage = this.stages[exercise];

    if (exercise === "Squat") {
      if (hipKneeAnkle > 150 && anklesApart) stage = "Up";
      if (hipKneeAnkle > 45 && hipKneeAnkle < 150 && anklesApart && stage === "Up") {
        this.counts[exercise] += 1;
        stage = "Down";
      }
    } else if (exercise === "Curl Up") {
      if (hipKneeAnkle > 30 && hipKneeAnkle < 100) {
        if (shoulderHipKnee > 100) stage = "Down";
        if (shoulderHipKnee < 50 && stage === "Down") {
          this.counts[exercise] += 1;
          stage = "Up";
        }
      }
    } else if (exercise === "Push Up") {
      if (shoulderElbowWrist < 70 && (nose[1] > elbowY || nose[1] > mouthLeft[1])) stage = "Down";
      if (shoulderElbowWrist > 160 && (nose[1] < elbowY || nose[1] < mouthRight[1]) && stage === "Down") {
        this.counts[exercise] += 1;
        stage = "Up";
      }
    } else if (exercise === "Jumping Jack") {
      if (elbowShoulderHip < 90 && shoulderHipKnee > 170) stage = "Down";
      if (elbowShoulderHip > 90 && shoulderHipKnee < 170 &&
          (nose[1] > wristLeft[1] || nose[1] > wristRight[1]) && stage === "Down") {
        this.counts[exercise] += 1;
        stage = "Up";
      }
    }
    this.stages[exercise] = stage;
  }
}

/** Accumulates how long each yoga pose is held. Times are in seconds, supplied by the caller. */
export class YogaTimer {
  constructor() { this.reset(); }

  reset() {
    this.durations = Object.fromEntries(YOGA_POSES.map((p) => [p, 0]));
    this.started = {};
  }

  /** detections: array of {pose, confidence} seen in this frame. */
  update(detections, now) {
    const seen = new Set(detections.map((d) => d.pose));
    for (const { pose, confidence } of detections) {
      if (confidence > YOGA_CONFIDENCE && !(pose in this.started)) this.started[pose] = now;
    }
    for (const pose of Object.keys(this.started)) {
      if (!seen.has(pose)) {
        this.durations[pose] += Math.max(0, now - this.started[pose]);
        delete this.started[pose];
      }
    }
  }

  totals(now) {
    return Object.fromEntries(YOGA_POSES.map((p) => [
      p, this.durations[p] + (p in this.started ? Math.max(0, now - this.started[p]) : 0),
    ]));
  }
}

export function formatSeconds(s) {
  const m = Math.floor(s / 60);
  return `${m}:${(s % 60).toFixed(2).padStart(5, "0")}`;
}
