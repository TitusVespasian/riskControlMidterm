import re

import lightgbm as lgb
import pandas as pd
from sklearn import preprocessing

import city_and_province_process
import merge_data
def temp():
    train_master = pd.read_csv('./data/train/Master_Training_Cleaned.csv')
    train_master = city_and_province_process.city_process(train_master)
    update_info_pd = pd.read_csv('data/train/userupdate_df.csv')
    log_info_pd = pd.read_csv('data/train/loginfo_df.csv')

    test_master=pd.read_csv('./data/test/Master_Test_Cleaned.csv')
    test_master=city_and_province_process.city_process(test_master)
    test_update_info_pd = pd.read_csv('data/test/test_userupdate_df.csv')
    test_log_info_pd=pd.read_csv('./data/test/test_loginfo_df.csv')

    train_all = merge_data.merge_data(train_master, update_info_pd, log_info_pd)
    test_all=merge_data.merge_data(test_master,test_update_info_pd,test_log_info_pd,if_all=True)
    X=test_all.copy()
    #X = test_all.drop(['target'], axis=1)  # .to_numpy()
    #y = train_master.target  # .to_numpy()
    # new_dict = {key:i for (i,key) in enumerate(X.columns)}
    categorical_ordered = ['UserInfo_1', 'UserInfo_3', 'UserInfo_5', 'UserInfo_6', 'UserInfo_14', 'UserInfo_15',
                           'UserInfo_16', 'SocialNetwork_1', 'SocialNetwork_2', 'SocialNetwork_7', 'SocialNetwork_12']
    categorical_cols = ['UserInfo_2_level', 'UserInfo_4_level', 'UserInfo_6', 'UserInfo_7', 'UserInfo_8_level',
                        'UserInfo_9',
                        'UserInfo_11', 'UserInfo_12', 'UserInfo_13', 'UserInfo_17', 'UserInfo_19', 'UserInfo_20_level',
                        'UserInfo_21', 'UserInfo_22', 'UserInfo_23', 'UserInfo_24', 'Education_Info1',
                        'Education_Info2',
                        'Education_Info3', 'Education_Info4', 'Education_Info5', 'Education_Info6', 'Education_Info7',
                        'Education_Info8',
                        'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21', 'city_sim']

    for icol in categorical_ordered:
        X[icol].astype('category').cat.as_ordered()

    for icol in categorical_cols:
        X[icol].astype('category')

    new_dict = {key: re.sub('[^A-Za-z0-9]+', '', key) + str(i) for (i, key) in enumerate(X.columns)}
    X.rename(columns=new_dict, inplace=True)

    list_fea_str = ['UserInfo720', 'UserInfo921', 'UserInfo1931', 'UserInfo2233', 'UserInfo2334', 'UserInfo2435',
                    'EducationInfo237', 'EducationInfo338', 'EducationInfo439', 'EducationInfo641', 'EducationInfo742',
                    'EducationInfo843', 'WeblogInfo1944', 'WeblogInfo2045', 'WeblogInfo2146']
    dict_fea_encode = {}
    for icol in list_fea_str:
        dict_fea_encode[icol] = preprocessing.LabelEncoder()
        X[icol] = dict_fea_encode[icol].fit_transform(X[icol].astype(str))  # 将提示的包含错误数据类型这一列进行转换
    ID=X['Idx0'].rename('Idx')
    X = X.drop(['Idx0'], axis=1)
    import joblib
    clf = joblib.load('dota_model.pkl')
    Y_res=clf.predict_proba(X)[:,1]
    df=pd.DataFrame(data={"target":Y_res})
    df=df.join(ID)
    df.to_csv("result.csv",index=False)
    print(df.shape)

if __name__=="__main__":
    Ytemp=temp()

