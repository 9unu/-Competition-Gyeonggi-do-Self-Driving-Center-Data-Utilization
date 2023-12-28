# calcOPticalFlowFarneback 추적 (track_optical_farneback.py)

import cv2, numpy as np

#threshold
def dir_test(x, dx_, dy_):
    if dx_ == 0:
       dx_ = 0.0001

    #threshold
    if (dy_ < -5) and (abs(dy_/dx_) > 0.5):
        return True
    else:
        return False
    
# 플로우 결과 그리기 ---①
def drawFlow(img,flow,step=16):
  h,w = img.shape[:2]
  # 16픽셀 간격의 그리드 인덱스 구하기 ---②
  idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(int)
  indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
  
  for x,y in indices:   # 인덱스 순회
    # 각 그리드 인덱스에 해당하는 플로우 결과 값 (이동 거리)  ---④
    dx,dy = flow[y, x].astype(int)


    if dir_test(x, dx, dy):
        # 각 그리드 인덱스 위치에서 이동한 거리 만큼 선 그리기 ---⑤
        cv2.line(img, (x,y), (x+dx, y+dy), (0,255, 0),2, cv2.LINE_AA)
        

def drawSpeed(img,flow,step=16):
  h,w = img.shape[:2]
  # 16픽셀 간격의 그리드 인덱스 구하기 ---②
  idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(int)
  indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
  dot_sum = 0
  for x,y in indices:   # 인덱스 순회
    # 각 그리드 인덱스에 해당하는 플로우 결과 값 (이동 거리)  ---④
    dx,dy = flow[y, x].astype(int)

    if dir_test(x, dx, dy):
        dot_sum += dy
    
  cv2.putText(img, str(dot_sum), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  return abs(dot_sum)
  
       

# output = None will not save the output video
num = 90
cctv = 3
road = 2
start = 1
for i in range(start, num + 1):
  prev = None # 이전 프레임 저장 변수

  #input
  cap = cv2.VideoCapture("./IO_data/input/video/cctv"+str(cctv)+"-"+str(road)+"/CCTV_" + str(cctv)+ "_"+str(i)+".mp4")
  fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
  delay = int(1000/fps)

  out = None
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  codec = cv2.VideoWriter_fourcc(*"mp4v")

  speed_list = []
  frame_num = 0
  while cap.isOpened():
    frame_num += 1
    speed = 0

    ret,frame = cap.read()
      #중요

    if not ret:
        print('Video has ended or failed!')
        break
    
    #원근변환 좌표
    points = np.array([[1350,209],[964,190],[370,29],[255,47]], dtype=np.float32)
    mag = [1, 4]
    h, w = frame.shape[:2]

    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # 변환 행렬 생성
    M = cv2.getPerspectiveTransform(points, dst)

    frame = cv2.warpPerspective(frame, M, (w, h))
    frame = cv2.resize(frame, None, fx=mag[0], fy=mag[1],
                    interpolation=cv2.INTER_AREA)
                    
    if not ret: break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # 최초 프레임 경우 
    if prev is None: 
      prev = gray # 첫 이전 프레임 --- ⑥
    else:
      # 이전, 이후 프레임으로 옵티컬 플로우 계산 ---⑦
      flow = cv2.calcOpticalFlowFarneback(prev,gray,None,\
                  0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 
      # 계산 결과 그리기, 선언한 함수 호출 ---⑧
      
      

      

      drawFlow(frame,flow)
      
      x = drawSpeed(frame,flow)
      if (frame_num % 10) == 1:
        speed_list.append(x)


      prev = gray

  with open("./IO_data/output/txt/cctv"+str(cctv)+"-"+str(road)+"/CCTV_" + str(cctv)+ "_"+str(i)+".txt", 'w') as f:
      f.write(' '.join(map(str, speed_list)))

  cap.release()