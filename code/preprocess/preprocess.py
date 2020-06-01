import numpy as np
import pandas as pd
import pickle
import os
import sys

class Preprocessing():

    def __init__(self, kind, extend=False):
        self.kind = kind
        self.load_path = '../intermediate/total_'+self.kind+'.p'
        if os.path.isfile(self.load_path):
            self.total = pd.read_pickle(self.load_path)
        self.load_data(kind)
        self.extend = extend


    def load_data(self,kind):

        path = '../raw/'
        if self.kind == 'train':
            self.label = pd.read_csv(path+kind+'_label.csv')

        if os.path.isfile(self.load_path):
            return

        self.activity = pd.read_csv(path+kind+'_activity.csv')
        self.combat = pd.read_csv(path+kind+'_combat.csv')
        self.pledge = pd.read_csv(path+kind+'_pledge.csv')
        self.trade = pd.read_csv(path+kind+'_trade.csv')
        self.payment = pd.read_csv(path+kind+'_payment.csv')

    def del_server(self):

        if os.path.isfile(self.load_path):
            return

        self.activity = self.activity.drop('server', axis=1)
        self.combat = self.combat.drop('server', axis=1)
        self.trade = self.trade.drop('server', axis=1)
        self.pledge = self.pledge.drop('server', axis=1)

    def del_charid(self):

        if os.path.isfile(self.load_path):
            return

        self.activity = self.activity.drop('char_id', axis=1)
        self.combat = self.combat.drop('char_id', axis=1)
        self.pledge = self.pledge.drop('char_id', axis=1)

    def tradePP_execute(self):

        if os.path.isfile(self.load_path):
            return

        self.trade = self.trade.drop('type', axis=1)
        self.trade = self.trade.drop('source_char_id', axis=1)
        self.trade = self.trade.drop('target_char_id', axis=1)

        trade_keys = list(self.trade.keys())
        sell_keys = trade_keys.copy()
        sell_keys.remove('target_acc_id')
        buy_keys = trade_keys.copy()
        buy_keys.remove('source_acc_id')
        self.sell = self.trade.copy()
        self.sell = self.sell.loc[:,sell_keys]
        self.buy = self.trade.copy()
        self.buy = self.buy.loc[:,buy_keys]

        self.sell = self.tradePP(self.sell.copy(),'source_')
        self.buy = self.tradePP(self.buy.copy(),'target_')

        self.trade = pd.merge(self.sell, self.buy, how = 'outer', on=['day','acc_id'])
        print(self.trade)

    def extractTime(self,y):
        h,m,s = y.split(':')
        return h

    def convertItemType(self, x):
        return self.item_key.index(x)

    def convertTime(self, x):
        return self.time_key.index(x)

    def or_operation(x,y):

        #print("y",y)
        #print("y_values",y.values)
        or_sum = 0
        for value in y.values:
            or_sum = or_sum | value

        return or_sum

    def rename(self,x,add):
            name = '_'+add
            x = x.rename(columns=lambda x: x+name if x != 'day' and x != 'acc_id' else x)
            return x

    def tradePP(self,x,label):
        sum_key = list(x.keys())
        sum_key.remove('time')
        sum_key.remove('item_type')
        trade = x.copy()
        trade = trade.loc[:,sum_key]
        trade_sum = trade.copy()
        trade_sum['average_item_price'] = trade_sum['item_price'] / trade_sum['item_amount']
        trade_sum['trade_amount'] = 1
        trade_sum = trade_sum.groupby(['day',label+'acc_id'], as_index=False).sum()
        print(trade_sum)
        if self.extend:
            trade_max = trade.copy()
            trade_max = trade_max.groupby(['day',label+'acc_id'], as_index=False).max()
            trade_max = trade_max.rename(columns=lambda x: x+'_max' if x != 'day' and x != label+'acc_id' else x)
            print(trade_max)
            trade_min = trade.copy()
            trade_min = trade_min.groupby(['day',label+'acc_id'], as_index=False).min()
            trade_min = trade_min.rename(columns=lambda x: x+'_min' if x != 'day' and x != label+'acc_id' else x)
            print(trade_min)
            #trade_std = trade.copy()
            #trade_std = trade_std.groupby(['day',label+'acc_id'], as_index=False).std()
            #trade_std = trade_std.rename(columns=lambda x: x+'_std' if x != 'day' and x != label+'acc_id' else x)
            #print(trade_std)
            trade_mean = trade.copy()
            trade_mean = trade_mean.groupby(['day',label+'acc_id'], as_index=False).mean()
            trade_mean = trade_mean.rename(columns=lambda x: x+'_mean' if x != 'day' and x != label+'acc_id' else x)
            print(trade_mean)
        unique_time_key = ['day',label+'acc_id','time']
        unique_item_type_key = ['day',label+'acc_id','item_type']
        trade_time = x.copy()
        trade_time = trade_time.loc[:,unique_time_key]
        trade_item_type = x.copy()
        trade_item_type = trade_item_type.loc[:,unique_item_type_key]

        self.time_key = list(trade_time['time'].unique())
        self.item_key = list(trade_item_type['item_type'].unique())

        trade_time['time'] = trade_time['time'].apply(self.extractTime)
        trade_item_type['item_type'] = trade_item_type['item_type'].apply(self.convertItemType)

        time_key = np.asarray(list(trade_time['time'].unique()))
        item_key = np.asarray(list(trade_item_type['item_type'].unique()))
        trade_time['time'] = trade_time['time'].apply(lambda x : int(''.join((((time_key == x)*1).astype(str))),2))
        trade_item_type['item_type'] = trade_item_type['item_type'].apply(lambda x : int(''.join((((item_key == x)*1).astype(str))),2))

        trade_time = trade_time.groupby(['day',label+'acc_id'], as_index=False).agg(self.or_operation)
        trade_item_type = trade_item_type.groupby(['day',label+'acc_id'], as_index=False).agg(self.or_operation)

        trade_time['time'] = trade_time['time'].apply(lambda x : bin(x)[2:])
        trade_item_type['item_type'] = trade_item_type['item_type'].apply(lambda x : bin(x)[2:])

        self.time_key = list(trade_time['time'].unique())
        self.item_key = list(trade_item_type['item_type'].unique())

        trade_time['time'] = trade_time['time'].apply(self.convertTime)
        trade_item_type['item_type'] = trade_item_type['item_type'].apply(self.convertItemType)

        x = pd.merge(trade_time,trade_item_type, how='outer', on=['day',label+'acc_id'])
        print(x)
        x = pd.merge(x,trade_sum, how='outer', on=['day',label+'acc_id'])
        if self.extend:
            print(x)
            x = pd.merge(x,trade_max, how='outer', on=['day',label+'acc_id'])
            print(x)
            x = pd.merge(x,trade_min, how='outer', on=['day',label+'acc_id'])
            #print(x)
            #x = pd.merge(x,trade_std, how='outer', on=['day',label+'acc_id'])
            print(x)
            x = pd.merge(x,trade_mean, how='outer', on=['day',label+'acc_id'])
            print(x)
        name = '_sell' if label == 'source_' else '_buy'
        x = x.rename(columns=lambda x: x+name if x != 'day' and x != label+'acc_id' else x)
        x = x.rename(columns={label+'acc_id' : 'acc_id'})
        return x

    def activityPP(self):

        if os.path.isfile(self.load_path):
            return

        activity = self.activity.copy()
        activity_sum = activity.copy()
        activity_sum = activity_sum.groupby(['day','acc_id'], as_index=False).sum()
        if self.extend:
            activity_max = activity.copy()
            activity_max = activity_max.groupby(['day','acc_id'], as_index=False).max()
            activity_max = activity_max.rename(columns=lambda x: x+'_max' if x != 'day' and x != 'acc_id' else x)
            activity_min = activity.copy()
            activity_min = activity_min.groupby(['day','acc_id'], as_index=False).min()
            activity_min = activity_min.rename(columns=lambda x: x+'_min' if x != 'day' and x != 'acc_id' else x)
            #activity_std = activity.copy()
            #activity_std = activity_std.groupby(['day','acc_id'], as_index=False).std()
            #self.rename(activity_std,'std')
            activity_mean = activity.copy()
            activity_mean = activity_mean.groupby(['day','acc_id'], as_index=False).mean()
            activity_mean = activity_mean.rename(columns=lambda x: x+'_mean' if x != 'day' and x != 'acc_id' else x)

        self.activity = activity_sum
        if self.extend:
            self.activity = pd.merge(self.activity, activity_max, how='outer', on=['day','acc_id'])
            self.activity = pd.merge(self.activity, activity_min, how='outer', on=['day','acc_id'])
            #self.activity = pd.merge(self.activity, activity_std, how='outer', on=['day','acc_id'])
            self.activity = pd.merge(self.activity, activity_mean, how='outer', on=['day','acc_id'])

        print("activity",self.activity)
        print("activity",len(self.activity['acc_id'].unique()))

    def convertClass(self, x):
        return self.class_key.index(x)

    def combatPP(self):

        if os.path.isfile(self.load_path):
            return

        sum_key = list(self.combat.keys())
        sum_key.remove('class')
        combat = self.combat.copy()
        combat = combat.loc[:,sum_key]
        combat_sum = combat.copy()
        combat_sum = combat_sum.groupby(['day','acc_id'], as_index=False).sum()

        mean_key = ['day','acc_id','level']
        combat_mean = self.combat.copy()
        combat_mean = combat_mean.loc[:,mean_key]
        combat_mean = combat_mean.groupby(['day','acc_id'], as_index=False).mean()
        combat_mean = combat_mean.rename(columns={'level':'average_level'})

        if self.extend:
            combat_max = combat.copy()
            combat_max = combat_max.groupby(['day','acc_id'], as_index=False).max()
            combat_max = combat_max.rename(columns=lambda x: x+'_max' if x != 'day' and x != 'acc_id' else x)
            combat_min = combat.copy()
            combat_min = combat_min.groupby(['day','acc_id'], as_index=False).min()
            combat_min = combat_min.rename(columns=lambda x: x+'_min' if x != 'day' and x != 'acc_id' else x)
            #combat_std = combat.copy()
            #combat_std = combat_std.groupby(['day','acc_id'], as_index=False).std()
            #self.rename(combat_std,'std')
            combat_mean = combat.copy()
            combat_mean = combat_mean.groupby(['day','acc_id'], as_index=False).mean()
            combat_mean = combat_mean.rename(columns=lambda x: x+'_mean' if x != 'day' and x != 'acc_id' else x)

        unique_key = ['day','acc_id','class']
        combat_unique = self.combat.copy()
        combat_unique = combat_unique.loc[:,unique_key]
        self.class_key = list(combat_unique['class'])
        combat_unique['class'] = combat_unique['class'].apply(self.convertClass)

        class_key = np.asarray(list(combat_unique['class'].unique()))
        combat_unique['class'] = combat_unique['class'].apply(lambda x : int(''.join((((class_key == x)*1).astype(str))),2))
        combat_unique = combat_unique.groupby(['day','acc_id'], as_index=False).agg(self.or_operation)
        combat_unique['class'] = combat_unique['class'].apply(lambda x : bin(x)[2:])

        self.class_key = list(combat_unique['class'].unique())
        combat_unique['class'] = combat_unique['class'].apply(self.convertClass)

        self.combat = pd.merge(combat_sum, combat_mean, how='outer', on=['day','acc_id'])
        self.combat = pd.merge(self.combat, combat_unique, how='outer', on=['day','acc_id'])
        if self.extend:
            self.combat = pd.merge(self.combat, combat_max, how='outer', on=['day','acc_id'])
            self.combat = pd.merge(self.combat, combat_min, how='outer', on=['day','acc_id'])
            #self.combat = pd.merge(self.combat, combat_std, how='outer', on=['day','acc_id'])
            self.combat = pd.merge(self.combat, combat_mean, how='outer', on=['day','acc_id'])
        print('self.combat',self.combat)

    def pledgePP(self):

        if os.path.isfile(self.load_path):
            return

        sum_key = list(self.pledge.keys())
        sum_key.remove('pledge_id')

        pledge = self.pledge.copy()
        pledge = pledge.loc[:,sum_key]
        pledge_sum = pledge.copy()
        pledge_sum = pledge_sum.groupby(['day','acc_id'], as_index=False).sum()

        max_key = ['day','acc_id','pledge_id']
        pledge_id_max = self.pledge.copy()
        pledge_id_max = pledge_id_max.loc[:,max_key]
        pledge_id_max = pledge_id_max.groupby(['day','acc_id'], as_index=False).max()

        self.pledge = pd.merge(pledge_sum, pledge_id_max, how='outer', on=['day','acc_id'])
        if self.extend:
            pledge_max = pledge.copy()
            pledge_max = pledge_max.groupby(['day','acc_id'], as_index=False).max()
            pledge_max = pledge_max.rename(columns=lambda x: x+'_max' if x != 'day' and x != 'acc_id' else x)
            pledge_min = pledge.copy()
            pledge_min = pledge_min.groupby(['day','acc_id'], as_index=False).min()
            pledge_min = pledge_min.rename(columns=lambda x: x+'_min' if x != 'day' and x != 'acc_id' else x)
            #pledge_std = pledge.copy()
            #pledge_std = pledge_std.groupby(['day','acc_id'], as_index=False).std()
            #self.rename(pledge_std,'std')
            pledge_mean = pledge.copy()
            pledge_mean = pledge_mean.groupby(['day','acc_id'], as_index=False).mean()
            pledge_mean = pledge_mean.rename(columns=lambda x: x+'_mean' if x != 'day' and x != 'acc_id' else x)

            self.pledge = pd.merge(self.pledge, pledge_max, how='outer', on=['day','acc_id'])
            self.pledge = pd.merge(self.pledge, pledge_min, how='outer', on=['day','acc_id'])
            #self.pledge = pd.merge(self.pledge, pledge_std, how='outer', on=['day','acc_id'])
            self.pledge = pd.merge(self.pledge, pledge_mean, how='outer', on=['day','acc_id'])
        print("self.pledge",self.pledge)


    def merge(self):

        if os.path.isfile(self.load_path):
            return

        self.total = pd.merge(self.activity, self.combat, how='outer', on=['day','acc_id'])
        self.total = pd.merge(self.total, self.pledge, how='outer', on=['day','acc_id'])
        self.total = pd.merge(self.total, self.trade, how='outer', on=['day','acc_id'])
        self.total = pd.merge(self.total, self.payment, how='outer', on=['day','acc_id'])
        print("total",self.total)

    def test_merge(self):

        if os.path.isfile(self.load_path):
            return

        self.total = pd.merge(self.activity, self.combat, how='left', on=['day','acc_id'])
        self.total = pd.merge(self.total, self.pledge, how='left', on=['day','acc_id'])
        self.total = pd.merge(self.total, self.trade, how='left', on=['day','acc_id'])
        self.total = pd.merge(self.total, self.payment, how='left', on=['day','acc_id'])
        print("total",self.total)

    def normalize(self):

        if os.path.isfile(self.load_path):
            return

        normalize_key = list(self.total.keys())
        normalize_key.remove('day')
        normalize_key.remove('acc_id')
        for col in normalize_key:
            entry = self.total[col].copy()
            mean = entry.mean()
            std = entry.std()
            self.total[col] = self.total[col].apply(lambda x : (x-mean)/std)

        self.total = self.total.fillna(0)

    def sampling(self):

        path = '../data/'+self.kind
        if os.path.isdir(path):
            return

        os.makedirs(path)
        days = range(1,29)

        acc_id_keys = list(self.total['acc_id'].unique())
        #print('num',len(acc_id_keys))
        for acc_id in acc_id_keys:
            sample = self.total.loc[self.total['acc_id']==acc_id,:].copy().sort_values(by=['day'],axis=0)
            sample.set_index('day',inplace=True)
            sample = sample.reindex(days)
            sample = sample.fillna(0)
            sample = sample.drop('acc_id',axis=1)
            sample = sample.to_numpy()

            with open(os.path.join(path,str(acc_id)+'.p'), 'wb') as f:
                pickle.dump(sample,f)


    def labeling(self):

        path = '../data/'+self.kind
        if os.path.isdir(path):
            return

        os.makedirs(path)

        #print(self.total)
        days = range(1,29)

        acc_id_keys = list(self.total['acc_id'].unique())
        for acc_id in acc_id_keys:
            sample = self.total.loc[self.total['acc_id']==acc_id,:].copy().sort_values(by=['day'],axis=0)
            sample.set_index('day',inplace=True)
            sample = sample.reindex(days)
            sample = sample.fillna(0)
            sample = sample.drop('acc_id',axis=1)
            sample = sample.to_numpy()
            label = self.label.loc[self.label['acc_id']==acc_id,:]
            #print(label)
            label = label.drop('acc_id',axis=1)
            label = label.to_numpy()
            label = tuple((label[0][0],label[0][1]))
            #print(label)
            #print(acc_id)

            with open(os.path.join(path,str(acc_id)+'.p'), 'wb') as f:
                pickle.dump(tuple((sample,label)),f)

            """with open(os.path.join(path,str(acc_id)+'.p'), 'rb') as f:
                entry = pickle.load(f)
                sample = entry[0]
                label = entry[1]
"""

    def execute(self):
        self.del_server()
        self.del_charid()
        self.tradePP_execute()
        self.activityPP()
        self.combatPP()
        self.pledgePP()

        if self.kind == 'train':
            self.merge()
        else:
            self.test_merge()
        self.normalize()

        if not os.path.isdir('../intermediate/total_'+self.kind+'.p'):
            self.total.to_pickle('../intermediate/total_'+self.kind+'.p')

        if self.kind == 'train':
            self.labeling()
        else:
            self.sampling()

if __name__ == '__main__':
    #p1 = Preprocessing('train',extend=True)
    #p1.execute()
    p2 = Preprocessing('test1',extend=True)
    p2.execute()
    p3 = Preprocessing('test2',extend=True)
    p3.execute()
