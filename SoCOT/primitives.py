import cv2
import logging
import os
import re
import time

import numpy as np
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from faster_whisper import WhisperModel

from collections import Counter

import json
import parselmouth
import librosa
from pathlib import Path

import logging
import pandas as pd 

from moviepy.editor import VideoFileClip 

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


logging_format = "%(asctime)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO, datefmt="'%H:%M:%S")
logging.getLogger().setLevel(logging.DEBUG)

#also use vlm to encode the body language
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
class Primitives:

    def __init__(self):
        #want to define input stream if using cv2 input stream
        self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        pass

    def pick_largest_face(faces):
        """
        returns largest face from list of dictionaries from MTCNN
        faces: list of dicts from MTCNN in format [x,y,w,h]
        """
        if not faces:
            logging.DEBUG("No Faces detected")
            return None
        return max(faces, key=lambda f: f["box"][2] * f["box"][3])
    
    def safe_crop(img, box, pad=0.15):
        #getting last two items from img shape
        h,w = img.shape[:2]
        x,y,bw,bh = box

        #expanding box by pad fract - idk if this is necessary... or can be down using matrix multi... need to look at math more
        px = int(bw * pad)
        py = int(bh * pad)
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(w, x+ bw + px)
        y1 = min(h, y + bh + py)

        if x1 <= x0 or y1 <= y0:
            return None
        return img[y0:y1, x0:x1]


    def extract_audio_primitive(self, file_path):
        
        video = VideoFileClip(file_path)
        video.audio.write_audiofile("audio.wav")

        #using parselmouth
        sound = parselmouth.Sound("audio.wav")
        print(sound.get_sampling_frequency())
    
    def extract_transcription_primitive(self, file_path):
        segments, _ = self.whisper_model.transcribe(file_path, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments)
        # logging.DEBUG("Transcription result: %s", text)
        # with open("transcription.txt", "w", encoding="utf-8") as txt:
        #     txt.write(text["text"])

        return text

    def video_capture(self):
        """
        This function will be called to record until the person says so
        """
        cap = cv2.VideoCapture(0)

        #define codec (4 byte code to specify depends on operating systsem, OSX = MJPG (.MJPG) for bigger size or X264 (.mkv))
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

        if not cap.isOpened():
            print("")
            exit()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.DEBUG("Can't recieve frame, stream end. Exiting...")
                break
            frame = cv2.flip(frame, 0)
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        
        #loop finished, release capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    def emotion_primitive(self, img_path):
        result = DeepFace.analyze(img_path, actions = ['emotion'], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        emo = result.get("emotion", {})

        dom_emo = result.get("dominant_emotion", max(emo, key=emo.get))
        
        return emo, dom_emo

    def load_audio_files(self, dir):
        audio_files = []
        print(dir)
        for filename in os.listdir(dir):
            if filename.endswith(".wav"):
                audio_files.append(os.path.join(dir, filename))
            logging.DEBUG("file: %s", filename)
            print(filename)
        return audio_files

    
    def extract_features(self, audio_file):
        #making sound object from audio file
        sound = parselmouth.Sound(audio_file)

        #extracting pitch and intensity
        pitch = sound.to_pitch()
        intensity = sound.to_intensity()
        harmonicity = sound.to_harmonicity()

        pitch_mean = parselmouth.praat.call(pitch, "Get mean", 0,0, "Hertz")
        pitch_min = sound.xmin
        pitch_max = sound.xmax
        intensity_mean = parselmouth.praat.call(intensity, "Get mean", 0,0, "DB")

        #get point process - series of time points, from pitch and sound obj, for timing events (jitter, shimmer, num pulses)
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0,0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0,0, 0.0001, 0.02, 1.3, 1.6)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

        return {
        "pitch_mean": pitch_mean,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "intensity_mean": intensity_mean,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": hnr
    }

    def process_audio_files(self, dir):
        features_list = []
        for filename in os.listdir(dir):
            if filename.endswith(".wav"):
                print(filename)
                audio_file = os.path.join(dir, filename)
                features = self.extract_features(audio_file)
                features["file"] = filename
                features_list.append(features)

        return pd.DataFrame(features_list)

    def crop_from_relative_bbox(self, frame_bgr, rbb, pad=0.15):
        """
        rbb is mediapipe's relative_bounding_box with xmin,ymin,width,height in [0,1].
        Returns cropped BGR face image or None.
        """
        h, w = frame_bgr.shape[:2]
        x0 = rbb.origin_x
        y0 = rbb.origin_y
        bw = rbb.width
        bh = rbb.height

        # Convert to pixel coords
        x1 = int((x0 + bw))
        y1 = int((y0 + bh))

        # Pad
        px = int((x1 - x0) * pad)
        py = int((y1 - y0) * pad)
        x0 = max(0, x0 - px)
        y0 = max(0, y0 - py)
        x1 = min(w, x1 + px)
        y1 = min(h, y1 + py)

        if x1 <= x0 or y1 <= y0:
            return None
        return frame_bgr[y0:y1, x0:x1]

    def analyze_emotion_on_face_crop(self, face_bgr):
        """
        Uses DeepFace emotion head without re-detecting faces.
        Returns (probs_dict, dominant, confidence)
        """
        result = DeepFace.analyze(
            img_path=face_bgr,
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]

        emo = result.get("emotion", {})
        probs = {e: float(emo.get(e, 0.0)) for e in EMOTIONS}

        # Normalize if needed
        s = sum(probs.values())
        if s > 0:
            probs = {k: v / s for k, v in probs.items()}

        dominant = result.get("dominant_emotion", max(probs, key=probs.get))
        conf = probs.get(dominant, 0.0)
        return probs, dominant, conf

    def aggregate_per_second(self, rows):
        """
        rows: list of (t_sec, probs_dict or None, face_found bool)
        returns list of dict per second
        """
        buckets = {}
        for t, probs, ok in rows:
            if not ok or probs is None:
                continue
            sec = int(np.floor(t))
            buckets.setdefault(sec, []).append(probs)

        timeline = []
        for sec in sorted(buckets.keys()):
            mean = {e: 0.0 for e in EMOTIONS}
            plist = buckets[sec]
            for p in plist:
                for e in EMOTIONS:
                    mean[e] += p[e]
            n = len(plist)
            mean = {e: mean[e] / n for e in EMOTIONS}
            dom = max(mean, key=mean.get)
            timeline.append({
                "time_sec": sec,
                "dominant": dom,
                "confidence": float(mean[dom]),
                **{f"p_{e}": float(mean[e]) for e in EMOTIONS},
                "n_samples": n
            })
        return timeline

    def ema_smooth_probs(self, timeline, alpha=0.5):
        """
        Smooths the probability columns across time using EMA.
        """
        smoothed = []
        prev = None
        for row in timeline:
            cur = {e: row[f"p_{e}"] for e in EMOTIONS}
            if prev is None:
                s = cur
            else:
                s = {e: alpha * cur[e] + (1 - alpha) * prev[e] for e in EMOTIONS}
            prev = s
            dom = max(s, key=s.get)
            out = dict(row)
            for e in EMOTIONS:
                out[f"p_{e}"] = float(s[e])
            out["dominant"] = dom
            out["confidence"] = float(s[dom])
            smoothed.append(out)
        return smoothed

    def video_emotion_timeline_mediapipe(self, mp4_path, sample_fps=5, min_det_conf=0.6, ema_alpha=0.5):
        print
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            print("could not open video")
            raise RuntimeError(f"Could not open video: {mp4_path}")
            

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / native_fps if frame_count else None
        print("Duration:", duration)
        # have to use pip install mediapipe==0.10.14 vs 0.10.32 with naive pip install?
        # jk dont do that bc then u have to  downgrade protbuf but tensor flow only work with the upgraded version sooo
        # mp_face = mp.solutions.face_detection

        # the issue was that that is a legacy version of mp, the following is based on face_detector.ipynb on the mediapipe API page
        base_options = python.BaseOptions(model_asset_path='detector.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        detector = vision.FaceDetector.create_from_options(options)

        rows = []
        step_ms = 1000.0 / sample_fps
        t_ms = 0.0

        print("enterring while loop")
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
            ok, frame_bgr = cap.read()
            # print(ok)
            if not ok:
                break

            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)

            if len(result.detections) == 0:
                # print("detections none")
                rows.append((t_ms / 1000.0, None, False))
                t_ms += step_ms
                continue

            # Choose the best detection (highest score)
            # det = max(result.detections, key=lambda d: d.score[0] if d.score else 0.0)
            det = max(result.detections, key=lambda d: d.confidence if hasattr(d, 'confidence') else 0.0)
            rbb = det.bounding_box
            face_crop = self.crop_from_relative_bbox(frame_bgr, rbb, pad=0.15)
            if face_crop is None:
                # print("face crop none")
                rows.append((t_ms / 1000.0, None, False))
                t_ms += step_ms
                continue

            probs, dom, conf = self.analyze_emotion_on_face_crop(face_crop)
            # print("probs: ", probs)
            rows.append((t_ms / 1000.0, probs, True))

            t_ms += step_ms
            if duration is not None and (t_ms / 1000.0) > duration + 0.5:
                break

        cap.release()

        print("exited while loop")

        per_sec = self.aggregate_per_second(rows)
        per_sec_smooth = self.ema_smooth_probs(per_sec, alpha=ema_alpha)
        return per_sec_smooth, rows
    
    
    
    def timeline_to_json(self,
        timeline,
        output_path,
        video_path,
        sample_fps,
        ema_alpha,
        detector="MediaPipe FaceDetection",
        emotion_model="DeepFace (emotion)"
    ):
        payload = {
            "video": {
                "path": str(video_path),
                "sample_fps": sample_fps,
                "aggregation": "per_second",
                "ema_alpha": ema_alpha,
                "detector": detector,
                "emotion_model": emotion_model
            },
            "emotions": []
        }

        for row in timeline:
            payload["emotions"].append({
                "time_sec": int(row["time_sec"]),
                "dominant": row["dominant"],
                "confidence": float(row["confidence"]),
                "probabilities": {
                    e: float(row[f"p_{e}"]) for e in EMOTIONS
                },
                "n_samples": int(row.get("n_samples", 0))
            })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

        return output_path, payload
    
    def video_description_primitives(self, mp4_path):
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-Instruct",
                                                torch_dtype=torch.bfloat16,
                                                _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager").to(DEVICE)
        

    def main(self):
        print("starting main")
        audio_dir = "audio_files"
        features_df = self.process_audio_files(audio_dir)
        features_df.to_json("audio_primitives.json", index=False)

        with open ('audio_primitives.json') as audio_json:
            audio_primitive = json.load(audio_json)
        # audio_primtives = json.load("audio_primitives.json")

        print("audio primitves done")

        timeline, frame_rows = self.video_emotion_timeline_mediapipe("video_files/50_50_35.mp4", sample_fps=5, min_det_conf=0.7, ema_alpha=0.5)

        outpath, payload = self.timeline_to_json(timeline=timeline, output_path="output/emotion_timeline.json", video_path="input.mp4", sample_fps=5, ema_alpha=0.5)
        self
        print("out done")

        text = self.extract_transcription_primitive("audio_files/50_50_35.wav")
        print(text)

        text_primitive = {"transcription": text}

        emotion_primitive = {"emotion primitive": payload}

        # audio_primitive = {"audio primitve": features_df}

        combined_primitive = {**text_primitive, **emotion_primitive, **audio_primitive}

        with open("output/combined_primitive_50_50_35.json", "w") as f:
            json.dump(combined_primitive, f, indent=2)


    def capture_affect_window(
    self,
    duration_s=2.0,
    sample_fps=3,
    camera_index=0,
    width=424,
    height=240,
    min_det_conf=0.6,
    pad=0.15,
    ema_alpha=None,          # optional: smooth within-window
    return_rows=False,
    detector=None,           # optionally pass a pre-created detector to avoid re-init overhead
    detector_model_path="detector.tflite",
):
        """
        Capture a short live window from the camera and estimate affect.

        Returns:
            summary (dict), and optionally rows (list)
        """
        # --- Setup detector (prefer reusing across calls; init can be costly) ---
        owns_detector = False
        if detector is None:
            base_options = python.BaseOptions(model_asset_path=detector_model_path)
            options = vision.FaceDetectorOptions(base_options=base_options)
            detector = vision.FaceDetector.create_from_options(options)
            owns_detector = True

        # --- Open camera ---
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            if owns_detector:
                detector.close()
            raise RuntimeError(f"Could not open camera index={camera_index}")

        # Try to reduce compute
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        # --- Sampling loop ---
        rows = []
        probs_list = []
        dom_list = []
        face_hits = 0

        start_t = time.time()
        next_sample_t = start_t
        sample_period = 1.0 / float(max(1, sample_fps))

        # Optional EMA smoothing across samples in this window
        ema_prev = None

        while True:
            now = time.time()
            if now - start_t >= duration_s:
                break

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                # brief camera hiccup; continue until duration
                continue

            # Only run inference on sample ticks
            if now < next_sample_t:
                continue
            next_sample_t = now + sample_period

            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            det_result = detector.detect(mp_image)

            # Handle no detections
            detections = getattr(det_result, "detections", None) or []
            if len(detections) == 0:
                rows.append((now - start_t, None, False))
                continue

            # Choose best detection robustly across API shapes
            def det_score(d):
                # Tasks API usually: d.categories[0].score
                if hasattr(d, "categories") and d.categories:
                    return float(getattr(d.categories[0], "score", 0.0) or 0.0)
                # Some wrappers expose score list
                if hasattr(d, "score") and d.score:
                    try:
                        return float(d.score[0])
                    except Exception:
                        pass
                # Fallback
                if hasattr(d, "confidence"):
                    try:
                        return float(d.confidence)
                    except Exception:
                        pass
                return 0.0

            best = max(detections, key=det_score)
            score = det_score(best)
            if score < float(min_det_conf):
                rows.append((now - start_t, None, False))
                continue

            rbb = getattr(best, "bounding_box", None)
            if rbb is None:
                rows.append((now - start_t, None, False))
                continue

            face_crop = self.crop_from_relative_bbox(frame_bgr, rbb, pad=pad)
            if face_crop is None or face_crop.size == 0:
                rows.append((now - start_t, None, False))
                continue

            # DeepFace emotion on crop (skip detector)
            probs, dom, conf = self.analyze_emotion_on_face_crop(face_crop)

            # Optional EMA smoothing within this short window
            if ema_alpha is not None:
                if ema_prev is None:
                    ema_prev = probs
                else:
                    probs = {e: float(ema_alpha) * probs[e] + (1.0 - float(ema_alpha)) * ema_prev[e] for e in EMOTIONS}
                    # renorm (numerical safety)
                    s = sum(probs.values())
                    if s > 0:
                        probs = {k: v / s for k, v in probs.items()}
                    ema_prev = probs
                dom = max(probs, key=probs.get)
                conf = probs[dom]

            face_hits += 1
            probs_list.append(probs)
            dom_list.append(dom)
            rows.append((now - start_t, probs, True))

        cap.release()
        if owns_detector:
            detector.close()

        # --- Aggregate into a compact summary ---
        n_samples = len(rows)
        n_face = face_hits

        if n_face == 0:
            summary = {
                "duration_s": float(duration_s),
                "sample_fps": int(sample_fps),
                "n_samples": int(n_samples),
                "n_face": 0,
                "face_hit_rate": 0.0,
                "dominant": None,
                "confidence": 0.0,
                **{f"p_{e}": 0.0 for e in EMOTIONS},
            }
            return (summary, rows) if return_rows else summary

        # Mean probabilities across face samples
        mean = {e: 0.0 for e in EMOTIONS}
        for p in probs_list:
            for e in EMOTIONS:
                mean[e] += float(p.get(e, 0.0))
        mean = {e: mean[e] / float(n_face) for e in EMOTIONS}

        dom = max(mean, key=mean.get)
        conf = float(mean[dom])

        # Also useful: how stable was it in the window?
        dom_mode = Counter(dom_list).most_common(1)[0][0]
        dom_mode_frac = Counter(dom_list)[dom_mode] / float(len(dom_list))

        summary = {
            "duration_s": float(duration_s),
            "sample_fps": int(sample_fps),
            "n_samples": int(n_samples),
            "n_face": int(n_face),
            "face_hit_rate": float(n_face) / float(max(1, n_samples)),
            "dominant": dom,
            "confidence": conf,
            "dominant_mode": dom_mode,
            "dominant_mode_frac": float(dom_mode_frac),
            **{f"p_{e}": float(mean[e]) for e in EMOTIONS},
        }

        return (summary, rows) if return_rows else summary


        
if __name__ == "__main__":
    primitives = Primitives()
    primitives.main()


        


