import pandas as pd
import pingouin as pg

# Create a DataFrame with your data

cctv = 3
road = 2
total_number = 90

output = open(f'./IO_data/output/txt/CCTV{cctv}-{road}/CCTV.txt', 'w', encoding='utf-8')

for i in range(1, total_number + 1):
    try:
        txt = open(f'./IO_data/output/txt/cctv{cctv}-{road}/CCTV_{cctv}_{i}.txt', 'r', encoding='utf-8').read()
        data = [int(x) for x in txt.split()]
        df = pd.DataFrame(data, columns=['y'])
        df['x'] = range(1, len(data) + 1)

        # Calculate the linear regression coefficient
        result = pg.linear_regression(df['x'], df['y'])

        output.write(str(result['coef'].values[1]) + ' ')
    except FileNotFoundError:
        print('FileNotFoundError')
        pass