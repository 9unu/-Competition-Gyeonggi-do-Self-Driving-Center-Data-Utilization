# Video processing 폴더

CCTV 영상 전처리에 대한 내용이 정리되어 있습니다.

## 1. `test_video_handling.ipynb` 및 `train_video_handling.ipynb`

### 목적:
- test 및 Train 영상 폴더에서 ROI(관심 영역)를 추출 및 확대

### 입력:
- test 및 Train 영상이 포함 된 폴더.

### 출력:
- ROI 추출된 영상 폴더.
- 해당 영역만 확장된 영상 폴더.

## 2. `split_video.ipynb`

### 목적:
ffmpeg를 사용하여 결합했던 15분 길이의 CCTV별 영상을 다시 90개의 10초 비디오로 분할합니다.

### 입력:
- 확장된 CCTV 비디오.

### 출력:
- 90개의 10초 확장된 비디오 세그먼트가 포함 된 폴더.

## 3. `upscaled_video_handling.ipynb`

### 목적:
확장된 비디오 데이터를 처리하며, ROI를 추출하고 확장합니다.

### 입력:
- 각 CCTV에 대해 90 개의 확장 영상로 나뉜 폴더.
- ROI가 추출된 영상 폴더.

### 출력:
- 해당 확장된 ROI가 있는 영상 폴더.

## 4. `function_file.ipynb`

### 기능:
- `video_folder_shift(folder_path, new_folder_path, file_list)`: test 및 Train 폴더에서 file_list에 선언된 파일들을 각 폴더로 이동합니다.
- `video_ROI(folder_path, pentagon_points, result_folder, file_list=[]`: 지정된 폴더에서 pentagon_points(ROI)만 선택하여 영상를 만듭니다.
- `Enlarge_ROI(folder_path, pentagon_points, result_folder, file_list=[]`: 지정된 폴더의 ROI만 남긴 영상에서 ROI 영역을 확장합니다.

## 5. `image_capture.ipynb`

### 목적:
주어진 비디오에서 초당 프레임들를 jpg 파일로 저장 

-> 도로 배경 이미지, 전이학습 이미지 생성

### 입력:
- 비디오 데이터.

---

## 6. `perspective_acquisition.py`

### 목적:
원근변환을 위한 좌표 획득

### 과정:
1. 원하는 도로의 영상에서 한 프레임을 따온 이미지의 디렉토리를 sbs.jpg의 위치에 입력한다.
- 차가 없고 차선이 잘 보이는 이미지일수록 좋다.
2. points와 mag를 조정한 후 result.jpg를 확인한다.
3. result.jpg가 top-view에 가깝게 나올 때까지 2번을 반복한다.
4. 획득한 points와 mag를 추후 사용할 수 있게끔 기록해둔다.

## 7. `tst.py`

### 목적:
영상으로부터 프레임별로 차량의 좌표가 담긴 csv 획득

### 입력:
- 영상 CCTV_" + str(cctv)+ "_"+str(i)+".mp4"가 담긴 폴더
- 실행 전 bridge_wrapper.py에 파라미터 입력이 필요
  - 6에서 획득한 cctv와 road에 맞는 POINTS와 MAG를 입력해줘야 함
- tst.py 파라미터 입력
  - num, cctv, road, start 파라미터를 입력해줘야 함
  - 모델을 넣어줘야 함
    - 디폴트는 yolov7x.pt와 classes = [2, 3, 5, 7]
    - 전이학습 모델을 사용할 경우 classes와 load_model을 별도로 입력해줘야 함


### 출력:
- 객체를 추적하고 원근변환을 적용한 CCTV_" + str(cctv)+ "_"+str(i)+".mp4이 담긴 폴더
- 10프레임별로 추적된 각 차량의 좌표가 찍힌 CCTV_" + str(cctv)+ "_"+str(i)+".csv

## 8. `car_num.py`

### 목적:
차량 대수가 담긴 csv 파일 생성 
- 이를 기록할 때 직교차선 차량, 반대차선 차량, 중복 인식 차량, 좌회전 차량도 제거함

### 입력:
- 프레임별 차량의 좌표가 찍힌 CCTV_{cctv}_{file_index}.csv가 담긴 폴더
- cctv, road, total_number, OUT_OF_SCREEN, NOT_THIS_LANE, RIGHT_LANE_Y 파라미터를 설정해줘야 함
  - OUT_OF_SCREEN, NOT_THIS_LANE, RIGHT_LANE_Y는 영상과 7의 csv를 대조하며 적합한 값을 찾아야 함

### 출력:
- {cctv}__{road}의 영상별 평균, 최대, 최소 차량 대수가 기록된 {cctv}-{road}.csv

## 9. `distance.py`

### 목적:
차량의 밀집도를 나타낼 수 있는 D가 담긴 csv 파일 생성
- 이를 기록할 때 직교차선 차량, 반대차선 차량, 중복 인식 차량도 제거함

### 입력:
- 프레임별 차량의 좌표가 찍힌 CCTV_{cctv}_{file_index}.csv가 담긴 폴더
- cctv, road, total_number, OUT_OF_SCREEN, NOT_THIS_LANE, RIGHT_LANE_Y 파라미터를 설정해줘야 함
  - OUT_OF_SCREEN, NOT_THIS_LANE, RIGHT_LANE_Y는 영상과 7의 csv를 대조하며 적합한 값을 찾아야 함

### 출력:
- 영상별 평균, 최대, 최소 D가 기록된 {cctv}-{road}.csv

## 10. `optical_flow.py`

### 목적:
Gunner Farneback의 Dense Optical Flow를 이용하여 속도를 추정

### 입력:
- 영상 CCTV_" + str(cctv)+ "_"+str(i)+".mp4"가 담긴 폴더
- num, cctv, road, start 파라미터를 입력해줘야 함
- 6에서 획득한 cctv와 road에 맞는 POINTS와 MAG를 입력해줘야 함

### 출력:
- 각 영상의 10프레임별 속도 합이 담긴 CCTV_" + str(cctv)+ "_"+str(i)+".txt" 파일

## 5. `regress.py`

### 목적:
속도의 증감을 나타낼 수 있는 속도의 회귀계수 획득

### 입력:
- 프레임별 속도가 찍힌 CCTV{cctv}-{road}/CCTV.txt 파일이 담긴 폴더
- num, cctv, road, start 파라미터를 입력해줘야 함

### 출력:
- 각 영상별로 속도를 나타내는 CCTV_{cctv}_{i}.txt

## 전체 사용법:

- **원본 영상 처리 (`test_video_handling.ipynb` 및 `train_video_handling.ipynb`):**
  - 폴더 위치를 지정 (입력, 출력)
  - 출력: ROI 추출 및 확장된 영상 폴더

- **확장된 비디오 처리 (`upscaled_video_handling.ipynb`):**
  - 폴더 위치를 지정 (입력, 출력)
  - 출력: ROI 추출 및 확장된 비디오 폴더

- **원근변환 좌표 획득 (`perspective_acquisition.py`):**
  - sbs.jpg 위치에 이미지 입력
  - result.jpg가 top-view에 가깝게 나올 때까지 points와 mag를 조정

- **프레임별 차량 좌표 획득 (`tst.py`):**
  - 폴더 위치를 지정 (입력, 출력)
  - bridge_wrapper.py에 파라미터 입력
  - yolov7x.pt 다운로드(또는 전이학습 모델 사용)
  - tst.py 파라미터 입력
  - 출력: 프레임별 차량 좌표가 담긴 csv 파일

- **차량 대수 획득 (`car_num.py`):**
  - 폴더 위치를 지정 (입력, 출력)
  - OUT_OF_SCREEN, NOT_THIS_LANE, RIGHT_LANE_Y 값 찾기
  - 파라미터 입력
  - 출력: 평균, 최대, 최소 차량 대수가 담긴 csv 파일

- **차량 밀집도 획득 (`distance.py`):**
  - 폴더 위치를 지정 (입력, 출력)
  - OUT_OF_SCREEN, NOT_THIS_LANE, RIGHT_LANE_Y 값 찾기
  - 파라미터 입력
  - 출력: 평균, 최대, 최소 D가 담긴 csv 파일

- **속도 획득 (`optical_flow.py`):**
  - 폴더 위치를 지정 (입력, 출력)
  - bridge_wrapper.py에 파라미터 입력
  - 파라미터 입력
  - 출력: 10프레임별 속도 합이 담긴 txt 파일

- **속도 회귀계수 획득 (`regress.py`):**
  - 폴더 위치를 지정 (입력, 출력)
  - 파라미터 입력
  - 출력: 속도를 나타내는 txt 파일

- **`car_num.py`에서 획득한 csv를 도로1용 input으로 사용**

- **`regress.py`에서 획득한 회귀계수를 txt를 transpose하여 distance.py의 오른쪽 열에 붙여서 도로2용 input으로 사용**


## 추가 참고 사항:

- pentagon_points는 ROI 지정 픽셀로 좌하단 부터 시계방향으로 지정해야함
- 입출력은 IO_data 폴더에 있음




# modeling 폴더

혼잡도 분류 모델링과 관련된 프로세스별 정보가 포함되어 있습니다.

## 1. create model 폴더

- **목적:**
  - 각 도로별 혼잡도 분류 모델 생성 (`cctv1_road1.ipynb`, `cctv1_road2.ipynb` 등).
- **입력:**
  - 각 도로별 전처리된 train 및 test 데이터
- **출력:**
  - 도로별 혼잡 분류 모델 (pickle)

## 2. run model 폴더 (`model.ipynb`)

- **목적:**
  - 각 도로별 test 데이터 예측 수행 및 제출용.csv 생성
- **입력:**
  - 각 도로별 전처리된 train 및 test 데이터
  - 도로별 혼잡 분류 모델 (pickle)
- **출력:**
  - 제출용.CSV (test 값이 채워진 어노테이션 테이블)

## 3. preprocessed data set

YOLO 객체 감지 및 원근 변환을 통해 생성된 특징 데이터셋

- **도로 1:**
  - 차량 수 (이전 비디오의 최소, 최대, 평균)
- **도로 2:**
  - 차량 속도 증가/감소 계수, 밀도 (최소, 최대, 평균)
