import cv2
import datetime
import os

save_dir = "captured_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 2번 카메라 + DirectShow
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# 🔬 해상도를 낮춰서(640x480) 대역폭 부하를 줄입니다.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("🔬 안정화 모드 실행 중! 's'로 캡처, 'q'로 종료")

while True:
    ret, frame = cap.read()
    
    # 튕기는 원인 파악을 위한 로그 출력
    if not ret:
        print("⚠️ 프레임을 읽어오지 못했습니다 (연결 끊김 또는 대역폭 초과).")
        break

    cv2.imshow('Microscope Capture Tool', frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/micro_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"✅ 사진 저장 완료: {filename}")
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()