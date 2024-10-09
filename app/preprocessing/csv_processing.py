import os
import numpy as np
import pandas as pd
import features


base_dir = os.path.dirname(os.path.abspath(__file__))

def process_csv(csv_file):
    df = pd.read_csv(csv_file)
    data_feature, current_data = [], {}

    # 기본 데이터 정리
    var_list = ['AccelerationX', 'AccelerationY', 'AccelerationZ', 'GyroX', 'GyroY', 'GyroZ']
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    df.replace('', np.nan, inplace=True)
    df = df.dropna()

    # interpolation으로 시간 간격 조정, 가운데 시간 구간만 추출
    df = features.subtract_first_row(df)
    df = features.interpolate_columns(df, features.add_new_time_column(df, 0.05), var_list)
    df = features.extract_center_segment(df)
    df = df[['new_time'] + var_list]

    # feature 추출하여 데이터프레임 형식으로 변환
    for var in range(6):
        current_data.update(features.extract(df[var_list[var]].tolist(), var))
    data_feature.append(current_data)
    df = pd.DataFrame(data_feature)

    # 기존 표준화 데이터셋 활용하여 데이터 정규화
    mean_std_df = pd.read_csv(os.path.join(base_dir, 'src', 'csv_mean_std_df.csv'), index_col='Unnamed: 0')
    mean_std_df = mean_std_df.to_dict()
    df = features.standardize_new_data(df, mean_std_df)

    # 66개 변수 중 54개로 축을 축소하기
    axis = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
    stats = ['mcr', 'std', 'max', 'min', 'mcr', 'peak', 'rms', 'grad', 'grad_1000']
    variables = [f'{front}_{back}' for front in axis for back in stats]
    df = df[variables]

    # 주성분분석으로 12개 feature로 축소하기
    components_df = pd.read_csv(os.path.join(base_dir, 'src', 'csv_PCA_result_11.csv'), index_col=0)
    pca_loadings = components_df.values
    pca_transformed = df.values @ pca_loadings.T
    pca_columns = [f'pca_var_{i}' for i in range(1, 12)]
    df = pd.DataFrame(pca_transformed, columns=pca_columns)
    return df
