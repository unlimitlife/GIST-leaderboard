import numpy as np
import pandas as pd
import pickle
import os
import sys

class Preprocessing():

    def __init__(self, kind):
        self.kind = kind
        self.load_path = '../intermediate/total_'+self.kind+'.p'
        if os.path.isfile(self.load_path):
            self.total = pd.read_pickle(self.load_path)
        self.load_data(kind)


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

    def tradePP(self,x,label):
        sum_key = list(x.keys())
        sum_key.remove('time')
        sum_key.remove('item_type')
        trade_sum = x.copy()
        trade_sum = trade_sum.loc[:,sum_key]
        trade_sum['average_item_price'] = trade_sum['item_price'] / trade_sum['item_amount']
        trade_sum['trade_amount'] = 1
        trade_sum = trade_sum.groupby(['day',label+'acc_id'], as_index=False).sum()

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
        x = pd.merge(x,trade_sum, how='outer', on=['day',label+'acc_id'])
        name = '_sell' if label == 'source_' else '_buy'
        x = x.rename(columns=lambda x: x+name if x != 'day' and x != label+'acc_id' else x)
        x = x.rename(columns={label+'acc_id' : 'acc_id'})
        return x

    def activityPP(self):

        if os.path.isfile(self.load_path):
            return

        self.activity = self.activity.groupby(['day','acc_id'], as_index=False).sum()
        print("activity",self.activity)
        print("activity",len(self.activity['acc_id'].unique()))

    def convertClass(self, x):
        return self.class_key.index(x)

    def combatPP(self):

        if os.path.isfile(self.load_path):
            return

        sum_key = list(self.combat.keys())
        sum_key.remove('class')
        combat_sum = self.combat.copy()
        combat_sum = combat_sum.loc[:,sum_key]
        combat_sum = combat_sum.groupby(['day','acc_id'], as_index=False).sum()

        mean_key = ['day','acc_id','level']
        combat_mean = self.combat.copy()
        combat_mean = combat_mean.loc[:,mean_key]
        combat_mean = combat_mean.groupby(['day','acc_id'], as_index=False).mean()
        combat_mean = combat_mean.rename(columns={'level':'average_level'})

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
        print('self.combat',self.combat)

    def pledgePP(self):

        if os.path.isfile(self.load_path):
            return

        sum_key = list(self.pledge.keys())
        sum_key.remove('pledge_id')

        pledge_sum = self.pledge.copy()
        pledge_sum = pledge_sum.loc[:,sum_key]
        pledge_sum = pledge_sum.groupby(['day','acc_id'], as_index=False).sum()

        max_key = ['day','acc_id','pledge_id']
        pledge_max = self.pledge.copy()
        pledge_max = pledge_max.loc[:,max_key]
        pledge_max = pledge_max.groupby(['day','acc_id'], as_index=False).max()

        self.pledge = pd.merge(pledge_sum, pledge_max, how='outer', on=['day','acc_id'])
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

    def binarize(self):
        price_key = ['acc_id','amount_spent']
        self.label_price = self.label.copy()
        self.label_price = self.label_price.loc[:,price_key]

        day_key = ['acc_id','survival_time']
        self.label_day = self.label.copy()
        self.label_day = self.label_day.loc[:,day_key]

        self.label_price['amount_spent'] = self.label_price['amount_spent'].apply(lambda x : int(x == 0))

        self.label_day['survival_time'] = self.label_day['survival_time'].apply(lambda x : int(x == 64))

        #from random import sample as sp
        #self.sampling_keys = sp(list(self.total.keys()).copy(),30)
        #if 'acc_id' in self.sampling_keys:
        #    self.sampling_keys.remove('acc_id')
        #if 'day' in self.sampling_keys:
        #    self.sampling_keys.remove('day')

        #with open('../data_binary/binary_sampling_keys.txt', 'wb') as f:
        #    pickle.dump(self.sampling_keys, f)
        #with open('../data_binary/binary_sampling_keys.txt', 'rb') as f:
            #pickle.load(f)

    def sampling(self):

        path = '../data_binary/'+self.kind
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
            #sample = sample.loc[:,self.sampling_keys]
            sample = sample.to_numpy()

            with open(os.path.join(path,str(acc_id)+'.p'), 'wb') as f:
                pickle.dump(sample,f)


    def labeling(self):
        path_p = '../data_binary/price/'+self.kind
        path_d = '../data_binary/day/'+self.kind
        if os.path.isdir(path_p):
            return
        if os.path.isdir(path_d):
            return

        os.makedirs(path_p)
        os.makedirs(path_d)

        #print(self.total)
        days = range(1,29)
        acc_id_keys = list(self.total['acc_id'].unique())

        for acc_id in acc_id_keys:
            sample = self.total.loc[self.total['acc_id']==acc_id,:].copy().sort_values(by=['day'],axis=0)
            sample.set_index('day',inplace=True)
            sample = sample.reindex(days)
            sample = sample.fillna(0)
            sample = sample.drop('acc_id',axis=1)
            #sample = sample.loc[:,self.sampling_keys]

            sample = sample.to_numpy()
            label_price = self.label_price.loc[self.label['acc_id']==acc_id,:]
            label_day = self.label_day.loc[self.label['acc_id']==acc_id,:]

            #print(label)
            label_price = label_price.drop('acc_id',axis=1)
            label_day = label_day.drop('acc_id',axis=1)
            label_price = label_price.to_numpy()[0]
            label_day = label_day.to_numpy()[0]
            #print(label_price)
            #print(label_day)
            #print(acc_id)

            with open(os.path.join(path_p,str(acc_id)+'.p'), 'wb') as f:
                pickle.dump(tuple((sample,label_price)),f)
            with open(os.path.join(path_d,str(acc_id)+'.p'), 'wb') as f:
                pickle.dump(tuple((sample,label_day)),f)

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
        self.binarize()

        if self.kind == 'train':
            self.labeling()
        else:
            self.sampling()

if __name__ == '__main__':
    p1 = Preprocessing('train')
    p1.execute()
    #p2 = Preprocessing('test1')
    #p2.execute()
    #p3 = Preprocessing('test2')
    #p3.execute()
