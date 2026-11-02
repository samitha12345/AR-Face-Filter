import cv2
import mediapipe as mp
import numpy as np
import time

# --- 1. Helper Functions ---
def overlay_png(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] + (1 - alpha) * background[y:y+h, x:x+w, c]
        )
    return background

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW, nH = int(h*sin + w*cos), int(h*cos + w*sin)
    M[0,2] += nW/2 - w/2
    M[1,2] += nH/2 - h/2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

# --- 2. Load Filters ---
filters = {
    "1": cv2.imread("filters/sunglasses.png", cv2.IMREAD_UNCHANGED),
    "2": cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED),
    "3": cv2.imread("filters/hat.png", cv2.IMREAD_UNCHANGED),
    "4": cv2.imread("filters/dog_ears.png", cv2.IMREAD_UNCHANGED)
}
current_filter = "1" # Default is sunglasses

# --- 3. Setup MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("AR Filter Started!")
print("Press 1: Sunglasses | 2: Mustache | 3: Hat | 4: Dog Ears")
print("Press 'S' to Save Snapshot | 'Q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        
        # Get specific landmarks for positioning
        left_eye = np.array([lms[33].x * w, lms[33].y * h])
        right_eye = np.array([lms[263].x * w, lms[263].y * h])
        nose_tip = np.array([lms[1].x * w, lms[1].y * h])
        forehead = np.array([lms[10].x * w, lms[10].y * h])
        chin = np.array([lms[152].x * w, lms[152].y * h])
        mouth_center = np.array([lms[13].x * w, lms[13].y * h])
        
        # Calculate face metrics
        cx, cy = (left_eye + right_eye) / 2
        eye_dist = np.linalg.norm(right_eye - left_eye)
        face_height = np.linalg.norm(chin - forehead)
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        img = filters[current_filter]
        
        if img is not None:
            # --- 4. Filter Specific Logic ---
            if current_filter == "1": # Sunglasses
                scale = eye_dist / (img.shape[1] * 0.6)
                new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
                resized = cv2.resize(img, (new_w, new_h))
                rotated = rotate_image(resized, angle)
                tx = int(cx - rotated.shape[1] / 2)
                ty = int(cy - rotated.shape[0] / 2)
                
            elif current_filter == "2": # Mustache
                scale = eye_dist / (img.shape[1] * 0.4)
                new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
                resized = cv2.resize(img, (new_w, new_h))
                rotated = rotate_image(resized, angle)
                tx = int(mouth_center[0] - rotated.shape[1] / 2)
                ty = int(mouth_center[1] - rotated.shape[0] / 2 + (eye_dist * 0.15)) # Slightly down
                
            elif current_filter == "3": # Hat
                scale = face_height / (img.shape[1] * 0.8)
                new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
                resized = cv2.resize(img, (new_w, new_h))
                rotated = rotate_image(resized, angle)
                tx = int(forehead[0] - rotated.shape[1] / 2)
                ty = int(forehead[1] - rotated.shape[0] * 0.8) # Sits on top of forehead
                
            elif current_filter == "4": # Dog Ears
                scale = face_height / (img.shape[1] * 0.6)
                new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
                resized = cv2.resize(img, (new_w, new_h))
                rotated = rotate_image(resized, angle)
                tx = int(forehead[0] - rotated.shape[1] / 2)
                ty = int(forehead[1] - rotated.shape[0] * 0.8)

            # Apply overlay
            frame = overlay_png(frame, rotated, tx, ty)

    # Display active filter name
    filter_names = {"1": "Sunglasses", "2": "Mustache", "3": "Hat", "4": "Dog Ears"}
    cv2.putText(frame, f"Filter: {filter_names[current_filter]}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AR Face Filter", frame)
    
    # --- 5. Keyboard Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key) in ["1", "2", "3", "4"]:
        current_filter = chr(key)
    elif key == ord('s'):
        filename = f"snapshots/AR_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Snapshot saved: {filename}")

cap.release()
cv2.destroyAllWindows()