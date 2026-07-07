import math
import torch
from colorama import Fore
from Configurations.dynamic_config import save_key, get_float
from utils.report_utils import save_new_data_point
import os

current_dir = os.path.dirname(__file__)
data_directory = os.path.join(current_dir, "data_record")

# data_directory=r'./data_record/'
def truncate(x,k=10000):
    return int(x * k) / k

def confession_mask(label,prediction_,pivot_value=0.5):
    TP_mask = (label > pivot_value) & (prediction_ > pivot_value)
    FP_mask = (label < pivot_value) & (prediction_ > pivot_value)
    FN_mask = (label > pivot_value) & (prediction_ <= pivot_value)
    TN_mask = (label < pivot_value) & (prediction_ <= pivot_value)

    return TP_mask, FP_mask, FN_mask, TN_mask

class ConfessionMatrix:
    def __init__(self,path,TP=0,FP=0,FN=0,TN=0):
        '''confession matrix'''
        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN
        self.epsilon = 0.00001

        self.path=path

        self.TP_MA = get_float('TP_MA_', config_file=self.path)
        self.FP_MA = get_float('FP_MA_', config_file=self.path)
        self.FN_MA = get_float('FN_MA_', config_file=self.path)
        self.TN_MA = get_float('TN_MA_', config_file=self.path)


    def save(self):
        save_key('TP_MA_', self.TP_MA, config_file=self.path)
        save_key('FP_MA_', self.FP_MA, config_file=self.path)
        save_key('FN_MA_', self.FN_MA, config_file=self.path)
        save_key('TN_MA_', self.TN_MA, config_file=self.path)


    def clear(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

    def update_confession_matrix(self,label,prediction_,pivot_value=0.5):
        '''masks'''
        TP_mask,FP_mask,FN_mask,TN_mask=confession_mask(label,prediction_,pivot_value=pivot_value)
        TP = (TP_mask).sum().item()
        FP = (FP_mask).sum().item()
        FN = (FN_mask).sum().item()
        TN = (TN_mask).sum().item()


        self.TP += TP
        self.FP += FP
        self.FN += FN
        self.TN += TN

        c=self.TP_MA+self.FP_MA+self.FN_MA+ self.TN_MA

        alpha=0.99*c

        self.TP_MA=alpha*self.TP_MA+(1-alpha)*TP
        self.FP_MA=alpha*self.FP_MA+(1-alpha)*FP
        self.FN_MA=alpha*self.FN_MA+(1-alpha)*FN
        self.TN_MA=alpha*self.TN_MA+(1-alpha)*TN


        return TP_mask,FP_mask,FN_mask,TN_mask
    @property
    def correct_classification(self):
        return self.TP+self.TN

    @property
    def total_classification(self):
        return self.TP+self.TN+self.FP+self.FN

    @property
    def accuracy(self):
        return self.correct_classification/(self.total_classification+self.epsilon)

    @property
    def recall(self):
        return self.TP/(self.TP+self.FN)

    @property
    def tpr(self):
        return self.recall

    @property
    def fpr(self):
        return self.FP/(self.FP+self.TN)

    @property
    def precision(self):
        return self.TP/(self.TP+self.FP)

    def view(self):
        total=self.total_classification
        print(f'TP={int((self.TP/total)*1000)/10}%, FP={int((self.FP/total)*1000)/10}%, FN={int((self.FN/total)*1000)/10}%, TN={int((self.TN/total)*1000)/10}%')
        total=self.TP_MA + self.TN_MA + self.FP_MA + self.FN_MA
        print(f'TP_MA={int((self.TP_MA/total)*1000)/10}%, FP_MA={int((self.FP_MA/total)*1000)/10}%, FN_MA={int((self.FN_MA/total)*1000)/10}%, TN_MA={int((self.TN_MA/total)*1000)/10}%')

class MovingRate():
    def __init__(self,name='000',decay_rate=0.01,initial_val=0.0,track_history=False,load_last=True):
        self.name=name
        self.path=data_directory+name
        self.decay_rate = decay_rate
        self.counter = 0
        self.moving_rate=initial_val
        self.momentum=0.0
        self.convergence=0.0
        self.var_x=1.

        self.initial_val=initial_val

        '''load latest'''
        if load_last: self.upload(initial_val)
        # print('tes------------------',self.moving_rate)
        # self.set_decay_rate()

        self.last_value=None
        self.track_history=track_history


    @property
    def val(self):
        return self.moving_rate

    @property
    def std(self):
        return math.sqrt(self.var_x)

    def __call__(self, *args, **kwargs):
        return self.moving_rate

    def update(self,value,influence_factor=1.0):
        with torch.no_grad():
            self.moving_rate=(1-self.decay_rate*influence_factor)*self.moving_rate+self.decay_rate*influence_factor*value
            self.momentum=(1-self.decay_rate*influence_factor)*self.momentum+self.decay_rate*influence_factor*(value**2)
            delta=value-self.moving_rate
            self.var_x=(1-self.decay_rate*influence_factor)*self.var_x+self.decay_rate*influence_factor*(delta**2)
            if self.last_value is not None:
                change = value - self.last_value
                self.convergence = self.decay_rate *influence_factor* change + self.convergence * (1 - self.decay_rate*influence_factor)
            self.last_value=value
            self.counter+=1

    def lower_rejection_criteria(self,x,k=2.0,report=False):
        threshold=self.moving_rate-k*math.sqrt(self.var_x)
        if report: print(f'lower criteria for {self.name},',Fore.YELLOW,f' x={x}, moving average= {self.moving_rate}, threshold={threshold}',Fore.RESET)
        return x<threshold

    def upper_rejection_criteria(self,x,k=2.0,report=False):
        threshold=self.moving_rate+k*math.sqrt(self.var_x)
        if report: print(f'Upper criteria for {self.name},',Fore.YELLOW,f' x={x}, moving average= {self.moving_rate}, threshold={threshold}',Fore.RESET)
        return x>threshold

    def save(self):
        save_key('moving_rate_', self.moving_rate, config_file=self.path)
        save_key('counter_', self.counter, config_file=self.path)
        save_key('momentum_', self.momentum, config_file=self.path)
        save_key('convergence_', self.convergence, config_file=self.path)
        save_key('var_x_', self.var_x, config_file=self.path)

        '''append to history records'''
        if self.track_history:
            save_new_data_point(self.moving_rate, self.path + '_moving_rate.txt')
            save_new_data_point(self.counter, self.path + '_counter.txt')
            save_new_data_point(self.momentum, self.path + '_momentum.txt')
            save_new_data_point(self.var_x, self.path + '_var_x.txt')
            save_new_data_point(self.convergence, self.path + '_convergence.txt')


    def upload(self,initial_val):
        try:
            self.moving_rate=get_float('moving_rate_',config_file=self.path,default=initial_val)

            if  math.isnan(self.moving_rate) or math.isinf(self.moving_rate): self.moving_rate=self.initial_val
            else:
                self.counter = get_float('counter_', config_file=self.path)
                self.momentum = get_float('momentum_', config_file=self.path)
                self.convergence = get_float('convergence_', config_file=self.path)
                self.var_x = get_float('_var_x', config_file=self.path)

        except Exception as e:
            print(Fore.RED,f' Error when getting a moving rate of {self.name} : {str(e)}',Fore.RESET)
            self.moving_rate=initial_val

    def view(self):
        self.moving_rate=truncate(self.moving_rate)
        self.momentum=truncate(self.momentum)
        self.convergence=truncate(self.convergence)
        self.var_x=truncate(self.var_x)

        # self.set_decay_rate()
        print(Fore.LIGHTBLUE_EX,end='')
        print(f'{self.name} moving rate = {self.moving_rate}, momentum = {self.momentum}, decay rate = {self.decay_rate}, convergence={self.convergence}, var={self.var_x}',end='')
        print(Fore.RESET)

class TrainingTracker:
    def __init__(self,name='',track_label_balance=False,track_prediction_balance=False,decay_rate=0.01,track_history=False):
        # try:
        self.name=name
        self.path=data_directory+name


        '''confession matrix'''
        self.confession_matrix=ConfessionMatrix(self.path)

        '''balance indicator'''
        self.label_balance_indicator=self.load_label_balance_indicator() if track_label_balance else None
        self.prediction_balance_indicator=self.load_prediction_balance_indicator() if track_prediction_balance else None

        self.loss_moving_average_=self.load_loss_moving_average()
        self.moving_accuracy=self.load_moving_accuracy()

        if math.isnan(self.loss_moving_average_):
            self.loss_moving_average_=.0

        self.convergence=self.load_convergence()
        self.momentum=self.load_momentum()
        if math.isnan(self.convergence):self.convergence=0.
        if math.isnan(self.momentum): self.momentum = 0.

        self.decay_rate=decay_rate
        self.counter=self.load_counter()
        self.last_loss=None

        self.track_history=track_history

        self.tmp_counter=0
        # except Exception as e:
        #     print(str(e))

    @property
    def accuracy(self):
        return self.confession_matrix.accuracy

    @property
    def loss(self):
        return None

    @loss.setter
    def loss(self,value):
        self.loss_moving_average_ = self.decay_rate * value + self.loss_moving_average_ * (1 - self.decay_rate)
        self.momentum = self.decay_rate * (value**2) + self.momentum * (1 - self.decay_rate)

        if self.last_loss is not None:
            change=value-self.last_loss
            self.convergence=self.decay_rate * change + self.convergence * (1 - self.decay_rate)
        self.last_loss=value
        self.counter+=1
        self.tmp_counter+=1


    def update_confession_matrix(self,label,prediction_,pivot_value=0.5):
        with torch.no_grad():
            TP_mask,FP_mask,FN_mask,TN_mask=self.confession_matrix.update_confession_matrix(label,prediction_,pivot_value)
            if self.label_balance_indicator is not None: self.update_label_balance_indicator(label)
            if self.prediction_balance_indicator is not None: self.update_prediction_balance_indicator(prediction_)

            instance_accuracy=(TP_mask.sum()+TN_mask.sum()).item()/(TP_mask.sum()+FP_mask.sum()+FN_mask.sum()+TN_mask.sum()).item()

            self.moving_accuracy = (1 - self.decay_rate) * self.moving_accuracy + self.decay_rate*instance_accuracy

            return TP_mask,FP_mask,FN_mask,TN_mask

    def load_label_balance_indicator(self):
        return get_float('label_balance_indicator',config_file=self.path)

    def load_prediction_balance_indicator(self):
        return get_float('prediction_balance_indicator',config_file=self.path)

    def load_loss_moving_average(self):
        return get_float('loss_moving_average',config_file=self.path)

    def load_moving_accuracy(self):
        return get_float('moving_accuracy',config_file=self.path,default=1.0)

    def load_convergence(self):
        return get_float('convergence',config_file=self.path)

    def load_momentum(self):
        return get_float('momentum',config_file=self.path)

    def load_counter(self):
        return get_float('counter',config_file=self.path)

    def update_label_balance_indicator(self,label,pivot_value=0.5):
        if label>pivot_value:
            self.label_balance_indicator=(1-self.decay_rate)*self.label_balance_indicator+self.decay_rate
        else:
            self.label_balance_indicator = (1 - self.decay_rate) * self.label_balance_indicator - self.decay_rate

    def update_prediction_balance_indicator(self,prediction,pivot_value=0.5):
        if prediction>pivot_value:
            self.prediction_balance_indicator=(1-self.decay_rate)*self.prediction_balance_indicator+self.decay_rate
        else:
            self.prediction_balance_indicator = (1 - self.decay_rate) * self.prediction_balance_indicator - self.decay_rate

    def print(self):

        # self.set_decay_rate()
        print(Fore.LIGHTBLUE_EX,f'statistics for {self.name}')


        self.loss_moving_average_ = truncate(self.loss_moving_average_,k=100000)
        self.convergence = truncate(self.convergence,k=100000)
        self.momentum = truncate(self.momentum,k=100000)
        print(f'Moving average loss= {self.loss_moving_average_},  Convergence = {self.convergence}, momentum = {self.momentum}')

        if self.confession_matrix.total_classification>0:
            self.confession_matrix.view()
            print(f'Moving accuracy = {self.moving_accuracy}')

        if self.label_balance_indicator is not None:
            self.label_balance_indicator = truncate(self.label_balance_indicator)
            print(f'label balance indicator = {self.label_balance_indicator}')

        if self.prediction_balance_indicator is not None:
            self.prediction_balance_indicator = truncate(self.prediction_balance_indicator)
            print(f'prediction balance indicator = {self.prediction_balance_indicator}')

        print(Fore.RESET,'-------------------------------------------------------------------------')

        self.confession_matrix.clear()


    def save(self):
        save_key('label_balance_indicator', self.label_balance_indicator, config_file=self.path)
        save_key('prediction_balance_indicator', self.prediction_balance_indicator, config_file=self.path)
        save_key('loss_moving_average', self.loss_moving_average_, config_file=self.path)
        save_key('convergence', self.convergence, config_file=self.path)
        save_key('momentum', self.momentum, config_file=self.path)
        save_key('counter', self.counter, config_file=self.path)
        save_key('moving_accuracy',self.moving_accuracy,config_file=self.path)

        self.confession_matrix.save()

        if self.track_history:
            '''append to history records'''
            save_new_data_point(self.label_balance_indicator, self.path+'_label_balance_indicator.txt')
            save_new_data_point(self.prediction_balance_indicator, self.path+'_prediction_balance_indicator.txt')
            save_new_data_point(self.loss_moving_average_, self.path+'_loss_moving_average_.txt')
            save_new_data_point(self.convergence, self.path+'_convergence.txt')
            save_new_data_point(self.momentum, self.path+'_momentum.txt')
            save_new_data_point(self.counter, self.path+'_counter.txt')

            save_new_data_point(self.confession_matrix.TP, self.path+'_TP.txt')
            save_new_data_point(self.confession_matrix.FP, self.path+'_FP.txt')
            save_new_data_point(self.confession_matrix.TN, self.path+'_TN.txt')
            save_new_data_point(self.confession_matrix.FN, self.path+'_FN.txt')

if __name__ == '__main__':
    save_new_data_point(torch.Tensor([3, 4, 5, 6]), 'my_file.txt')
