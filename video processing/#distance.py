#distance
import pandas as pd
import numpy as np

#
num = 65
cctv = 1
road = 2
start = 65

#화면밖에 존재한다고 가정하는 좌표
OUT_OF_SCREEN = 2100

#반대차선에 존재한다고 가정하는 좌표
NOT_THIS_LANE = (2700, 1900)

#직교차선의 y좌표
RIGHT_LANE_Y = 100


def right_lane(tupl_):
    coord = tuple(map(float, tupl_.split(',')))
    if coord[0] > RIGHT_LANE_Y:
        return True
    else:
        return False

result = pd.DataFrame(columns=['mean', 'max', 'min'])


for file_index in range(start, num + 1):
    try:
        csv_file = f"./IO_data/output/csv/cctv{cctv}-{road}/CCTV_{cctv}_{file_index}.csv"
        table = pd.read_csv(csv_file, encoding='utf-8')

        # Define condition 'A'

        for column in table.columns:
            # 다 0인 열 제거
            try:
                if all(value == '0,0' for value in table[column]):
                    table = table.drop(column, axis=1)
            except KeyError:
                pass

            # 직교차선 제거
            try:                
                opp = True
                for tupl in table[column]:
                    if right_lane(tupl) and tupl != '0,0':
                        opp = False
                        break

                if opp:
                    try:
                        table = table.drop(column, axis=1)
                    except KeyError:
                        pass

            except KeyError:
                pass

            #정차차량 제거
            try:
                tup_sum = (0,0)
                sum_count = 0
                first_tup = None

                for tupl in table[column]:
                    if tupl != '0,0':
                        coord = tuple(map(float, tupl.split(',')))
                        if first_tup is None:
                            first_tup = tuple(map(float, tupl.split(',')))

                        tup_sum = (tup_sum[0] + coord[0], tup_sum[1] + coord[1])
                        sum_count += 1
                    
                av_tup = (tup_sum[0] / sum_count, tup_sum[1] / sum_count)
                #threshold
                if sum_count > 3 and av_tup[1] < OUT_OF_SCREEN and abs(av_tup[0] - first_tup[0]) < 10  and abs(av_tup[1] - first_tup[1]) < 10:
                    table = table.drop(column, axis=1)
            
            except KeyError:
                pass

        distance_list = []
        coord = None
        for index, row in table.iterrows():
            car_list = []
            for value in row:
                if value != "0,0":
                    coord = tuple(map(float, value.split(',')))
                    found = False
                    for car in car_list:
                        if (abs(coord[0] - car[0]) + abs(coord[1] - car[1])) < 50:
                            found = True
                            break
                    if not found:
                        car_list.append(coord)
            
            car_list = np.array(car_list)


            if len(car_list) == 0 or len(car_list) == 1:
                car_std = 1000000
            else:
                car_std = sum(np.std(car_list, axis=0))/len(car_list)
                
            
            # 최종 평균 계산
            distance_list.append(car_std)

        mean_distance = sum(distance_list) / len(distance_list)
        max_distance = max(distance_list)
        min_distance = min(distance_list)

        # Use file_index as the index for the result DataFrame
        result.loc[file_index] = [mean_distance, max_distance, min_distance]

    except FileNotFoundError:
        result.loc[file_index] = [0, 0, 0]

result.to_csv(f"./IO_data/output/csv/cctv{cctv}-{road}/{cctv}-{road}.csv", encoding='utf-8-sig', index=False)