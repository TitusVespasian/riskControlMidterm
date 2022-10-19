#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:22:13 2022

@author: xzytql,zlwtql
"""
import json

import pandas as pd

# 对于城市数据，统一去除“市或省”后缀
def encodingstr_cp(s, appendix):
    if s.find(appendix) != -1:
        s = s[:s.find(appendix)]
    return s
check_none = set()
city_level = {1: ['北京', '上海', '广州', '深圳'],
              1.5: ['成都', '重庆', '杭州', '武汉', '西安', '郑州', '青岛', '长沙', '天津', '苏州', '南京', '东莞', '沈阳', '合肥', '佛山'],
              2: ['昆明', '福州', '无锡', '厦门', '哈尔滨', '长春', '南昌', '济南', '宁波', '大连', '贵阳', '温州', '石家庄', '泉州', '南宁', '金华',
                  '常州', '珠海', '惠州', '嘉兴', '南通', '中山', '保定', '兰州', '台州', '徐州', '太原', '绍兴', '烟台', '廊坊'],
              3: ['渭南', '海口', '汕头', '潍坊', '扬州', '洛阳', '乌鲁木齐', '临沂', '唐山', '镇江', '盐城', '湖州', '赣州', '漳州', '揭阳', '江门',
                  '桂林', '邯郸', '泰州', '济宁', '呼和浩特', '咸阳', '芜湖', '三亚', '阜阳', '淮安', '遵义', '银川', '衡阳', '上饶', '柳州', '淄博',
                  '莆田', '绵阳', '湛江', '商丘', '宜昌。沧州', '连云港', '南阳', '蚌埠', '驻马店', '滁州', '邢台', '潮州', '秦皇岛', '肇庆', '荆州',
                  '周口', '马鞍山', '清远', '宿州', '威海', '九江', '新乡', '信阳', '襄阳', '岳阳', '安庆', '菏泽', '宜春', '黄冈', '泰安', '宿迁',
                  '株洲', '宁德', '鞍山', '南充', '六安', '大庆', '舟山'],
              4: ['黔南', '常德', '渭南湖', '孝感', '丽水', '运城', '德州', '张家口', '鄂尔多斯', '阳江', '泸州', '丹东', '曲靖', '乐山', '许昌',
                  '湘潭', '晋中', '安阳', '齐齐哈尔', '北海', '宝鸡', '抚州', '景德镇', '延安', '三明', '抚顺亳州', '日照', '西宁', '衢州', '拉萨',
                  '淮北', '焦作', '平顶山', '滨州', '吉安', '濮阳', '眉山', '池州', '荆门', '铜仁', '长治', '衡水', '铜陵', '承德', '达州', '邵阳',
                  '德阳', '龙岩', '南平', '淮南', '黄石', '营口', '东营', '吉林', '韶关', '枣庄', '包头', '怀化', '宣城', '临汾', '聊城', '梅州',
                  '盘锦', '锦州', '榆林', '玉林', '十堰', '汕尾', '咸宁', '宜宾', '永州', '益阳', '黔南州', '黔东南', '恩施', '红河', '大理', '大同',
                  '鄂州', '忻州', '吕梁', '黄山', '开封', '郴州', '茂名', '漯河', '葫芦岛', '河源', '娄底', '延边'],
              5: ['汉中', '辽阳', '四平', '内江', '六盘水', '安顺', '新余', '牡丹江', '晋城', '自贡', '三门峡', '赤峰', '本溪', '防城港', '铁岭',
                  '随州', '广安', '广元', '天水', '遂宁', '萍乡', '西双版纳', '绥化', '鹤壁', '湘西', '松原', '阜新', '酒泉', '张家界', '黔西南',
                  '保山', '昭通', '河池', '来宾', '玉溪', '梧州', '鹰潭', '钦州', '云浮', '佳木斯', '克拉玛依', '呼伦贝尔', '贺州', '通化', '阳泉',
                  '朝阳', '百色', '毕节', '贵港', '丽江', '安康', '通辽', '德宏', '朔州', '伊犁', '文山', '楚雄', '嘉峪关', '凉山', '资阳',
                  '锡林郭勒盟', '雅安', '普洱', '崇左', '庆阳', '巴音郭楞(巴州)', '乌兰察布', '白山', '昌吉', '白城', '兴安盟', '定西', '喀什', '白银',
                  '陇南', '巴彦淖尔', '巴中', '鸡西', '乌海', '临沧', '海东', '张掖', '商洛', '黑河', '哈密', '吴忠', '攀枝花', '双鸭山', '阿克苏',
                  '石嘴山', '阿拉善盟', '海西', '平凉', '林芝', '固原', '武威', '儋州', '吐鲁番', '甘孜', '辽源', '临夏', '铜川', '金昌', '鹤岗',
                  '伊春', '中卫', '怒江', '和田', '迪庆', '甘南', '阿坝', '大兴安岭', '七台河', '山南', '日喀则', '塔城', '博尔塔拉', '昌都', '阿勒泰',
                  '玉树', '海南', '黄南', '果洛', '克孜勒苏', '阿里', '海北', '那曲', '三沙']}

with open(r'./data/al_long_tude.json', 'r', encoding='utf-8') as fj:
    city_pos = json.load(fj)
city_pos_dict = {}
sum_lo = 0
sum_al = 0
len_c = len(city_pos)
for icity in city_pos:
    city_pos_dict[encodingstr_cp(icity['name'], '市')] = [
        icity['longitude'], icity['latitude']]
    sum_lo += icity['longitude']
    sum_al += icity['latitude']

avg_lo = sum_lo / len_c
avg_al = sum_al / len_c



def get_featurename(name):
    try:
        rex_str = name[name.rfind('_') + 1:]
    except Exception as err:
        print(err)
        print('Not successful')
        print('name:' + name)
        return name
    return rex_str


# 省份信息处理 UserInfo_19, UserInfo_7
def province_selection(_train_master):
    important_pro_19 = ['山东', '浙江', '福建', '广西壮族自治区', '内蒙古自治区', '天津', '云南', '新疆维吾尔自治区', '黑龙江', '湖南']
    _train_master['UserInfo_19'] = _train_master['UserInfo_19'].apply(lambda x: x if x in important_pro_19
    else 'rest')
    # TODO: ajust 25 greater for xgboost lesser for lr lightgbm ok
    _train_master = _train_master.join(pd.get_dummies(_train_master.UserInfo_19, prefix="UserInfo_19"))
    _train_master.drop("UserInfo_19", axis=1, inplace=True)

    important_pro_7 = ['山东', '云南', '新疆', '浙江']
    _train_master['UserInfo_7'] = _train_master['UserInfo_7'].apply(lambda x: x if x in important_pro_7 else 'rest')
    # TODO: ajust 46 greater for xgboost lesser for lr lightgbm ok
    _train_master = _train_master.join(pd.get_dummies(_train_master.UserInfo_7, prefix="UserInfo_7"))
    _train_master.drop("UserInfo_7", axis=1, inplace=True)

    return _train_master


def get_city_level(x, if_print=False):
    if x == '0' or x == '不详':
        tmp = (1 * 4 + 1.5 * 15 + 2 * 30 + 3 * 70 + 4 *
               90 + 128 * 5) / (4 + 15 + 30 + 70 + 90 + 128)
        return tmp
    x = encodingstr_cp(x, '市')
    for ilevel in city_level.keys():
        if x in city_level[ilevel]:
            return ilevel
    if if_print:
        print(x)
    if len(x) >= 3:
        return 6
    return 5


def get_city_al(x, if_print=False):
    if x == '不详' or x == '0':
        return sum_al
    x = encodingstr_cp(x, '市')
    try:
        return city_pos_dict[x][0]
    except KeyError:
        if if_print:
            print("No city al" + x)
        check_none.add(x)
        return 26.5834


def get_city_lo(x):
    if x == '不详' or x == 0:
        return sum_lo
    x = encodingstr_cp(x, '市')
    try:
        return city_pos_dict[x][1]
    except KeyError:
        # print("No city lo" + x)
        return 107.977


def count_unique(x):
    return len(x.unique())


# 城市信息处理 UserInfo_2, UserInfo_4 UserInfo_8 UserInfo_20
def city_process(_train_master):
    for ifea in ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']:
        _train_master[ifea +
                      '_level'] = _train_master[ifea].apply(get_city_level)
        _train_master[ifea + '_lo'] = _train_master[ifea].apply(get_city_lo)
        _train_master[ifea + '_al'] = _train_master[ifea].apply(get_city_al)

    _train_master['city_sim'] = _train_master[['UserInfo_2',
                                               'UserInfo_4', 'UserInfo_8', 'UserInfo_20']].nunique(axis=1)
    _train_master = _train_master.drop(
        ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20'], axis=1)

    return _train_master


if __name__ == "__main__":

    train_master = pd.read_csv(r"./data/train/Master_Training_Cleaned_expCity.csv")
    y_train = train_master["target"].values
    train_master = province_selection(train_master)
    train_master = city_process(train_master)
    train_master.to_csv(r"./data/train/Master_Training_Modified.csv", index=False, sep=',')
    print(train_master.shape)
