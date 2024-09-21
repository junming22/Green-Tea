import cv2
import numpy as np
from openvino.runtime import Core
import os
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av

# OpenVINO Core 초기화
ie = Core()

# 모델 경로 설정
face_detection_model_xml = r"/workspaces/blank-app/model/face-detection-adas-0001.xml"
face_reid_model_xml = r"/workspaces/blank-app/model/face-reidentification-retail-0095.xml"

# 모델 로드 및 컴파일
face_det_model = ie.read_model(model=face_detection_model_xml)
face_det_compiled = ie.compile_model(model=face_det_model, device_name="CPU")
face_reid_model = ie.read_model(model=face_reid_model_xml)
face_reid_compiled = ie.compile_model(model=face_reid_model, device_name="CPU")

# 입력과 출력 레이어 명칭 추출
face_det_input_layer = face_det_compiled.input(0)
face_det_output_layer = face_det_compiled.output(0)
face_reid_input_layer = face_reid_compiled.input(0)
face_reid_output_layer = face_reid_compiled.output(0)

# 얼굴 검출 함수
def detect_faces(frame):
    if frame is None:
        print("Error: 이미지가 제대로 로드되지 않았습니다.")
        return []

    h, w = frame.shape[:2]
    input_shape = face_det_input_layer.shape
    resized_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
    transposed_frame = resized_frame.transpose((2, 0, 1))
    input_image = np.expand_dims(transposed_frame, axis=0)

    results = face_det_compiled([input_image])[face_det_output_layer]
    faces = []
    for result in results[0][0]:
        confidence = result[2]
        if confidence > 0.5:
            xmin = int(result[3] * w)
            ymin = int(result[4] * h)
            xmax = int(result[5] * w)
            ymax = int(result[6] * h)
            faces.append((xmin, ymin, xmax, ymax))

    return faces

# 얼굴 재식별 함수
def reidentify_face(face_img):
    if face_img is None or face_img.size == 0:
        print("Error: 얼굴 이미지가 유효하지 않습니다.")
        return None

    resized_face = cv2.resize(face_img, (face_reid_input_layer.shape[3], face_reid_input_layer.shape[2]))
    transposed_face = resized_face.transpose((2, 0, 1))
    input_image = np.expand_dims(transposed_face, axis=0)

    face_embedding = face_reid_compiled([input_image])[face_reid_output_layer]
    return face_embedding.flatten()

# 두 임베딩 비교 함수
def compare_faces(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        print("Error: 하나 이상의 임베딩이 None입니다.")
        return 0
    cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cosine_similarity * 100

# 기준 얼굴 임베딩 추출 함수
def get_reference_embeddings(image_paths):
    reference_embeddings = {}
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Error: 파일 경로를 찾을 수 없습니다. {image_path}")
            continue

        reference_img = cv2.imread(image_path)
        if reference_img is None:
            print(f"Error: 이미지를 로드할 수 없습니다. {image_path}")
            continue

        faces = detect_faces(reference_img)
        if len(faces) > 0:
            xmin, ymin, xmax, ymax = faces[0]
            face_img = reference_img[ymin:ymax, xmin:xmax]
            embedding = reidentify_face(face_img)
            if embedding is not None:
                reference_embeddings[os.path.basename(image_path)] = embedding
                print(f"Embedding extracted for {os.path.basename(image_path)}.")
        else:
            print(f"No face detected in {image_path}.")
    
    return reference_embeddings

# 기준 얼굴 이미지 경로 설정
reference_image_paths = [
    r"/workspaces/blank-app/image.jpg",
    r"/workspaces/blank-app/image2.jpg"
]

# 기준 얼굴 임베딩 추출
reference_embeddings = get_reference_embeddings(reference_image_paths)

# AI 처리기를 위한 클래스 정의
class VideoProcessor(VideoProcessorBase):
    def __init__(self, reference_embeddings):
        self.reference_embeddings = reference_embeddings

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = detect_faces(img)

        for (xmin, ymin, xmax, ymax) in faces:
            face_img = img[ymin:ymax, xmin:xmax]
            face_embedding = reidentify_face(face_img)

            if face_embedding is None:
                continue

            best_match = None
            highest_similarity = 0
            for ref_name, ref_embedding in self.reference_embeddings.items():
                similarity = compare_faces(face_embedding, ref_embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = ref_name

            # 확장자 제외하고 파일 이름만 표시
            if best_match:
                best_match_name = os.path.splitext(best_match)[0]
                label = f"{best_match_name}: {highest_similarity:.2f}%"
            else:
                label = "No Match"

            color = (0, 255, 0) if highest_similarity > 70 else (0, 0, 255)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit을 통한 웹캠 스트리머 추가
webrtc_streamer(
    key="example",
    video_processor_factory=lambda: VideoProcessor(reference_embeddings),
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)
