import sys
import pandas
import numpy as np
import datetime
from dateutil import rrule
from tqdm import tqdm

def get_edge_index(length):
    a = []
    b = []
    for i in range(length):
        if i == 0:
            continue
        a.append(0)
        b.append(i)
        a.append(i)
        b.append(0)
    c = []
    c.append(a)
    c.append(b)
    return c


def get_monthes(create_at):
    # a = "2015-09-26"
    create_at = create_at.split(' ')[0]
    date_i = datetime.datetime.strptime(create_at, '%Y-%m-%d').date()
    # today = datetime.datetime.today().date()
    if obj == 'Twitter15':
        today = datetime.date(2022,1,11)
    else:
        today = datetime.date(2022,4,9)

    month_sep = rrule.rrule(rrule.MONTHLY, dtstart=date_i, until=today)
    return month_sep.count()

def main(obj):

    data_1 = pandas.read_csv('../data/'+obj + '/' + obj + '_User_Information.csv',sep='\t')
    data_2 = pandas.read_csv('../data/'+obj + '/' + obj + '_User_Friends.csv',sep='\t')
    data_3 = pandas.read_csv('../data/'+obj + '/' + obj + '_Ego_Relationships.csv',sep='\t')

    total_feature = []
    user_list = data_1['user_id'].unique().tolist()




    for user_id in user_list:
        twitter_result = data_1[data_1['user_id'] == user_id].iloc[0, :]
        if twitter_result['user_status'] == 0:
            root_feature = []
            root_feature.append(user_id)
            for i in range(9):
                root_feature.append(-1)
            root_feature.append(twitter_result['user_status'])
        else:
            root_feature = [user_id, twitter_result['url'], twitter_result['protected'], twitter_result['verified'],
                            twitter_result['followers_count'], twitter_result['friends_count'],
                            twitter_result['listed_count'], twitter_result['favourites_count'],
                            twitter_result['statuses_count'], get_monthes(twitter_result['created_at']),
                            twitter_result['user_status']]
        total_feature.append(root_feature)

    friends_list = data_2[data_2['reason'].isna()]['friend_id'].unique().tolist()

    for friend_id in friends_list:
        twitter_result = data_2[data_2['friend_id'] == friend_id].iloc[0, :]
        if twitter_result['user_status'] == 0:
            root_feature = []
            root_feature.append(friend_id)
            for i in range(9):
                root_feature.append(-1)
            root_feature.append(twitter_result['user_status'])
        else:
            root_feature = [friend_id, twitter_result['url'], twitter_result['protected'], twitter_result['verified'],
                            twitter_result['followers_count'], twitter_result['friends_count'],
                            twitter_result['listed_count'], twitter_result['favourites_count'],
                            twitter_result['statuses_count'], get_monthes(twitter_result['created_at']),
                            twitter_result['user_status']]
        total_feature.append(root_feature)

    data_total = pandas.DataFrame(total_feature,
                                  columns=['user_id', 'url', 'protected', 'verified', 'followers_count', 'friends_count',
                                           'listed_count',
                                           'favourites_count', 'statuses_count',
                                           'created_at', 'user_status'])
    cols = ['url', 'protected', 'verified', 'followers_count', 'friends_count', 'listed_count', 'favourites_count',
            'statuses_count', 'created_at', 'user_status']  # 可以改成自己需要的列的名字
    for item in cols:
        mean_tmp = np.mean(np.array(data_total[item]))
        std_tmp = np.std(np.array(data_total[item]))
        if (std_tmp):
            data_total[item] = data_total[item].apply(lambda x: (x - mean_tmp) / std_tmp)


    user_id_friend = data_3['user_id'].unique().tolist()

    standard_list = []
    for index, row in data_1.iterrows():
        twitter_id = row['twitter_id']
        user_id = row['user_id']
        tree_feature = []
        # 构建root feature
        twitter_result = data_total[data_total['user_id'] == user_id].iloc[0, :]
        root_feature = [float(twitter_result['url']), float(twitter_result['protected']), float(twitter_result['verified']),
                        float(twitter_result['followers_count']), float(twitter_result['friends_count']),
                        float(twitter_result['listed_count']),
                        float(twitter_result['favourites_count']),
                        float(twitter_result['statuses_count']), float(twitter_result['created_at']),
                        float(twitter_result['user_status'])]

        tree_feature.append(root_feature)

        if user_id in user_id_friend:
            friend_ids = data_2[(data_2['user_id'] == user_id) & (data_2['user_status'] == 1)]['friend_id'].tolist()
            for friend_id in friend_ids:
                twitter_result = data_total[data_total['user_id'] == friend_id].iloc[0, :]
                temp_feature = [float(twitter_result['url']), float(twitter_result['protected']),
                                float(twitter_result['verified']),
                                float(twitter_result['followers_count']), float(twitter_result['friends_count']),
                                float(twitter_result['listed_count']),
                                float(twitter_result['favourites_count']),
                                float(twitter_result['statuses_count']), float(twitter_result['created_at']),
                                float(twitter_result['user_status'])]
                tree_feature.append(temp_feature)
            temp_dict = {}
            temp_dict['twitter_id'] = twitter_id
            temp_dict['user_id'] = str(user_id)
            temp_dict['root_feature'] = str(root_feature)
            temp_dict['tree_feature'] = str(tree_feature)
            temp_dict['edge_index'] = str(get_edge_index(len(tree_feature)))
            temp_dict['root_index'] = 0
            temp_dict['tree_len'] = len(tree_feature)
            temp_dict['status'] = True
            standard_list.append(temp_dict)

        else:
            tree_feature.append(root_feature)
            temp_dict = {}
            temp_dict['twitter_id'] = twitter_id
            temp_dict['user_id'] = str(user_id)
            temp_dict['root_feature'] = str(root_feature)
            temp_dict['tree_feature'] = str(tree_feature)
            temp_dict['edge_index'] = str(get_edge_index(len(tree_feature)))
            temp_dict['root_index'] = 0
            temp_dict['tree_len'] = len(tree_feature)
            temp_dict['status'] = False
            standard_list.append(temp_dict)
    print(standard_list[:10])

    print('a')
    for i, dic_sample in tqdm(enumerate(standard_list), total=len(standard_list), desc='Saving .npz files'):
        np.savez('../data/' + obj + '_Ego/'+str(dic_sample['twitter_id'])+'.npz', user_id=dic_sample['user_id'],
                 root_feature=dic_sample['root_feature'], tree_feature=dic_sample['tree_feature'],
                 edge_index=dic_sample['edge_index'], root_index=dic_sample['root_index'],
                 tree_len=dic_sample['tree_len'], status=dic_sample['status'])

if __name__ == '__main__':
    obj = sys.argv[1]
    main(obj)