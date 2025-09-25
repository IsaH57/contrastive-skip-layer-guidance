import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional

class HandValidator:
    def __init__(self, 
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe hand detection pipeline

        Args:
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, image_path: str) -> Dict:
        """
        Detect hands in image and extract validation scores

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with hand detection results and scores
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image", "valid_hands": 0}

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.hands.process(rgb_image)

        # Initialize results dictionary
        hand_data = {
            "num_hands_detected": 0,
            "valid_hands": 0,
            "hands": [],
            "overall_valid": False
        }

        if results.multi_hand_landmarks:
            hand_data["num_hands_detected"] = len(results.multi_hand_landmarks)

            for idx, (hand_landmarks, hand_info) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Extract hand information
                hand_score = hand_info.classification[0].score
                hand_label = hand_info.classification[0].label

                # Landmark confidences
                landmark_scores = [lm.visibility for lm in hand_landmarks.landmark if hasattr(lm, 'visibility')]
                if not landmark_scores:  # If visibility not available, assume presence=1
                    landmark_scores = [1.0] * len(hand_landmarks.landmark)

                avg_landmark_confidence = np.mean(landmark_scores)
                min_landmark_confidence = np.min(landmark_scores) if landmark_scores else 0.0

                # --- NEW: visible-only confidence ---
                visible_scores = [s for s in landmark_scores if s > 0.2]
                if visible_scores:
                    visible_avg_conf = float(np.mean(visible_scores))
                else:
                    visible_avg_conf = 0.0
                # ------------------------------------

                # Basic geometric validation
                geometric_score = self._validate_hand_geometry(hand_landmarks)

                # Overall hand validity score
                validity_score = (hand_score * 0.4 + 
                                  visible_avg_conf * 0.3 +   # updated to use visible confidence
                                  geometric_score * 0.3)

                # Determine if hand is valid
                is_valid = validity_score > 0.5

                if is_valid:
                    hand_data["valid_hands"] += 1

                # Store hand data
                hand_info_dict = {
                    "hand_index": idx,
                    "hand_type": hand_label,
                    "detection_confidence": hand_score,
                    "avg_landmark_confidence": avg_landmark_confidence,
                    "min_landmark_confidence": min_landmark_confidence,
                    "visible_avg_landmark_confidence": visible_avg_conf,
                    "geometric_score": geometric_score,
                    "validity_score": validity_score,
                    "is_valid": is_valid,
                    "landmarks": [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                }

                hand_data["hands"].append(hand_info_dict)

        # Overall validity: at least one valid hand detected
        hand_data["overall_valid"] = hand_data["valid_hands"] > 0

        return hand_data

    def _validate_hand_geometry(self, hand_landmarks) -> float:
        """
        Perform basic geometric validation of hand landmarks
        """
        landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

        if len(landmarks) != 21:
            return 0.0

        score = 1.0
        try:
            hand_width = max([lm[0] for lm in landmarks]) - min([lm[0] for lm in landmarks])
            hand_height = max([lm[1] for lm in landmarks]) - min([lm[1] for lm in landmarks])

            if hand_width > 0 and hand_height > 0:
                aspect_ratio = max(hand_width, hand_height) / min(hand_width, hand_height)
                if aspect_ratio > 3.0:
                    score *= 0.5

            distances = []
            for i in range(len(landmarks)):
                for j in range(i+1, len(landmarks)):
                    dist = np.sqrt((landmarks[i][0] - landmarks[j][0])**2 + 
                                   (landmarks[i][1] - landmarks[j][1])**2)
                    distances.append(dist)

            avg_distance = np.mean(distances)
            if avg_distance < 0.01:
                score *= 0.3

        except Exception:
            score *= 0.7

        return score

    def batch_validate(self, image_paths: List[str]) -> List[Dict]:
        results = []
        for image_path in image_paths:
            result = self.detect_hands(image_path)
            result["image_path"] = image_path
            results.append(result)
        return results

    def get_simple_score(self, image_path: str) -> Tuple[bool, float]:
        result = self.detect_hands(image_path)
        if result.get("error"):
            return False, 0.0
        if result["valid_hands"] == 0:
            return False, 0.0

        best_score = max([hand["validity_score"] for hand in result["hands"]])
        is_valid = result["overall_valid"]
        return is_valid, best_score

# Example usage function
def example_usage():
    validator = HandValidator(detection_confidence=0.7)
    image_path = "/export/home/ru63zus/repos/contrastive-skip-layer-guidance/experiments/flux_results_20250924_074213/A scientist working /seed_3/slg_4.0_cfg_5.0.png"

    detailed_results = validator.detect_hands(image_path)
    print("Detailed Results:")
    print(f"Number of hands detected: {detailed_results['num_hands_detected']}")
    print(f"Valid hands: {detailed_results['valid_hands']}")
    print(f"Overall valid: {detailed_results['overall_valid']}")

    for hand in detailed_results["hands"]:
        print(f"\nHand {hand['hand_index']} ({hand['hand_type']}):")
        print(f"  Hand detection: {hand['detection_confidence']:.3f}")
        print(f"  Avg landmark confidence: {hand['avg_landmark_confidence']:.3f}")
        print(f"  Min landmark confidence: {hand['min_landmark_confidence']:.3f}")
        print(f"  Visible avg landmark confidence: {hand['visible_avg_landmark_confidence']:.3f}")
        print(f"  Geometric score: {hand['geometric_score']:.3f}")
        print(f"  Validity score: {hand['validity_score']:.3f}")
        print(f"  Is valid: {hand['is_valid']}")

    is_valid, confidence = validator.get_simple_score(image_path)
    print(f"\nSimple result: Valid={is_valid}, Confidence={confidence:.3f}")

if __name__ == "__main__":
    example_usage()
