import os
import numpy as np
import pandas as pd
from config import PREPROCESSING_PARAMS_DIR


def extract(data: list, variable):
    # 1. 평균 구하기
    mean = sum(data) / len(data)

    # 2. 표준편차 구하기
    std = np.std(data)

    # 3. 최댓값과 최솟값 구하기
    maximum = max(data)
    minimum = min(data)

    # 4. 최대 변화율의 절댓값 구하기
    abs_mcr = max([abs(data[idx + 1] - data[idx]) for idx in range(len(data) - 1)])

    # 5. 피크의 개수 구하기
    num_peak = sum([1 if data[idx + 1] > data[idx] and data[idx + 1] > data[idx + 2] else 0
                    for idx in range(len(data) - 2)])

    # 6. Root Mean Square 구하기
    rms = (sum([value ** 2 for value in data]) / len(data)) ** 0.5

    # 7. 추세선의 기울기 구하기
    grad = np.polyfit([idx for idx in range(1, len(data) + 1)], data, 1)[0] * 20

    # 8. n밀리초간 평균 변화율의 최댓값 구하기
    grad_200 = max([abs(data[idx + 4] - data[idx]) / 4 for idx in range(len(data) - 4)])
    grad_400 = max([abs(data[idx + 8] - data[idx]) / 8 for idx in range(len(data) - 8)])
    grad_1000 = max([abs(data[idx + 20] - data[idx]) / 20 for idx in range(len(data) - 20)])

    # 계산한 통계치들을 데이터셋으로 만들기
    features = {'mean': mean, 'std': std, 'max': maximum, 'min': minimum, 'mcr': abs_mcr,
                'peak': num_peak, 'rms': rms, 'grad': grad, 'grad_200': grad_200, 'grad_400': grad_400, 'grad_1000': grad_1000}

    # 딕셔너리의 키 앞에 변수의 약어를 붙여 반환하기
    var_name = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ'][variable]
    return {f'{var_name}_{key}': round(float(value), 5) for key, value in features.items()}


# 1. 모든 컬럼에 대하여, 각 컬럼의 모든 데이터에서 해당 컬럼의 0번째 row의 값을 빼기
def subtract_first_row(df):
    return df - df.iloc[0]


# 2. new_time이라는 새로운 컬럼을 만들기
def add_new_time_column(df, time_interval=0.05):
    max_sampling_time = df['SamplingTime'].max()
    new_time = np.arange(0, max_sampling_time + time_interval, time_interval)
    return pd.DataFrame({'new_time': new_time})


# 3. 선형보간을 통해 AccelerationX, AccelerationY, AccelerationZ, GyroX, GyroY, GyroZ 컬럼 보간
def interpolate_columns(df, new_df, var_list):
    # 기존 데이터프레임의 SamplingTime과 new_time을 기준으로 보간
    for col in var_list:
        new_df[col] = np.interp(new_df['new_time'], df['SamplingTime'], df[col])
    return new_df


# 4. 정가운데 인덱스 기준으로 앞뒤로 100개 데이터(5초 분량)만 남기기
def extract_center_segment(df, window_size=100):
    mid_idx = len(df) // 2
    start_idx = max(0, mid_idx - window_size // 2)
    end_idx = min(len(df), mid_idx + window_size // 2)
    return df.iloc[start_idx:end_idx]


# 사다리꼴 적분으로 속도와 변위를 구하는 함수
def integrate(data: list, delta_t: float, var: int):
    velocity, displacement = [0], [0]

    # 사다리꼴 적분법을 통해 속도와 변위 계산
    for i in range(1, len(data)):
        # 속도 계산 (사다리꼴 적분법)
        v = velocity[-1] + (data[i-1] + data[i]) / 2 * delta_t
        velocity.append(v)

        # 변위 계산 (사다리꼴 적분법)
        d = displacement[-1] + (velocity[i-1] + velocity[i]) / 2 * delta_t
        displacement.append(d)

    var_name = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ'][var]
    to_return = {f'{var_name}_vel': f'{velocity[-1]}', f'{var_name}_loc': f'{displacement[-1]}'}

    return to_return


# dictionary를 바탕으로 표준화를 수행하는 함수
def standardize_new_data(new_df, mean_std_dict):
    for col in new_df.columns:
        if col in mean_std_dict:
            mean = mean_std_dict[col]['mean']
            std = mean_std_dict[col]['std']
            new_df[col] = (new_df[col] - mean) / std
    return new_df

def preprocess_csv(csv_file):
    df = pd.read_csv(csv_file)
    data_feature, current_data = [], {}

    # 기본 데이터 정리
    var_list = ['AccelerationX', 'AccelerationY', 'AccelerationZ', 'GyroX', 'GyroY', 'GyroZ']
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    df.replace('', np.nan, inplace=True)
    df = df.dropna()

    # interpolation으로 시간 간격 조정, 가운데 시간 구간만 추출
    df = subtract_first_row(df)
    df = interpolate_columns(df, add_new_time_column(df, 0.05), var_list)
    df = extract_center_segment(df)
    df = df[['new_time'] + var_list]

    # feature 추출하여 데이터프레임 형식으로 변환
    for var in range(6):
        current_data.update(extract(df[var_list[var]].tolist(), var))
    data_feature.append(current_data)
    df = pd.DataFrame(data_feature)

    # 기존 표준화 데이터셋 활용하여 데이터 정규화
    mean_std_df = pd.read_csv(os.path.join(PREPROCESSING_PARAMS_DIR, 'csv_mean_std_df.csv'), index_col='Unnamed: 0')
    mean_std_df = mean_std_df.to_dict()
    df = standardize_new_data(df, mean_std_df)

    # 66개 변수 중 54개로 축을 축소하기
    axis = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
    stats = ['mcr', 'std', 'max', 'min', 'mcr', 'peak', 'rms', 'grad', 'grad_1000']
    variables = [f'{front}_{back}' for front in axis for back in stats]
    df = df[variables]

    # 주성분분석으로 12개 feature로 축소하기
    components_df = pd.read_csv(os.path.join(PREPROCESSING_PARAMS_DIR, 'csv_PCA_result_11.csv'), index_col=0)
    pca_loadings = components_df.values
    pca_transformed = df.values @ pca_loadings.T
    pca_columns = [f'pca_var_{i}' for i in range(1, 12)]
    df = pd.DataFrame(pca_transformed, columns=pca_columns)
    return df
