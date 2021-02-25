


import numpy as np
import random
import math
import copy
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from gcs import *

import arcode
import os

def predict_collisions(diff_list):

  diff_list.sort()
  factor=1000.0

  collision_val=0.0

  dict_diff={}

  for i in range(0,len(diff_list)):
    diff_list[i]=closest_integer(diff_list[i]*factor)

    if diff_list[i] in dict_diff:
      dict_diff[diff_list[i]]+=1.0
    else:
      dict_diff[diff_list[i]]=1.0
  # print(dict_diff)

  for item in dict_diff.keys():
    if item<=factor:
      collision_val+=(((factor-item)*dict_diff[item]*1.00)/factor)      

  collision_rate=collision_val*1.00/len(diff_list)

  return collision_rate    


from sklearn import linear_model  

import matplotlib
# matplotlib.use('Agg')
matplotlib.use('GTKAgg')

def get_slope_bias(data_list,cdf_list,bool_recurse):



  if len(data_list)==0:
    return 0.0,0.0

  if len(set(data_list))==1:
    return 0.0,cdf_list[0]  

  X=[]

  for i in range(0,len(data_list)):
    X.append([data_list[i]])  

  
  # huber = HuberRegressor().fit(X, cdf_list)

  # print("Huber coef:",huber.coef_,huber.intercept_)

  # return huber.coef_,huber.intercept_  

  # max_cdf=max(cdf_list)
  # min_cdf=min(cdf_list)
  # max_data=max(data_list)
  # min_data=min(data_list)  
  # temp_slope=(max_cdf-min_cdf)*1.00/(max_data-min_data)
  # temp_bias=min_cdf-min_data*temp_slope

  # return temp_slope,temp_bias  

  # random.shuffle(data_list)

  temp_list=[]

  for i in range(0,len(data_list)):
    temp_list.append(i)

  index_list=random.sample(temp_list, min(10000,len(temp_list)))

  # print("list stats: ",min(data_list),max(data_list),np.percentile(data_list,50))  
  
  operate_list=[]
  cdf_operate_list=[]

  for i in range(0,len(index_list)):
    operate_list.append(data_list[index_list[i]])
    cdf_operate_list.append(cdf_list[index_list[i]])



  x=np.array(operate_list)
  y=np.array(cdf_operate_list)
  n=len(operate_list)

  x_mean = np.mean(x) 
  y_mean = np.mean(y) 
  x_mean,y_mean 

  Sxy=0.0
  Sxx=0.0
  for i in range(0,len(operate_list)):
    Sxy+=(operate_list[i]*cdf_operate_list[i]-x_mean*y_mean)
    Sxx+=(operate_list[i]*operate_list[i]-x_mean*x_mean)
    
  # Sxy = np.sum(x*y)- n*x_mean*y_mean 
  # Sxx = np.sum(x*x)-n*x_mean*x_mean 
    
  slope = Sxy/Sxx 
  bias = y_mean-slope*x_mean   

  # print("slope bias:",slope,bias)

  error_list=[]

  for i in range(0,len(operate_list)):
    error_list.append(abs((slope*operate_list[i]+bias)-cdf_operate_list[i]))

  thershold_error=np.percentile(error_list,90)

  new_operate_list=[]
  new_cdf_operate_list=[]

  for i in range(0,len(operate_list)):
    if error_list[i]<thershold_error:
      new_operate_list.append(operate_list[i])
      new_cdf_operate_list.append(cdf_operate_list[i])

  # print("slope bias:",slope,bias,bool_recurse)    

  if bool_recurse:    
    return get_slope_bias(new_operate_list,new_cdf_operate_list,False)    

  return slope,bias




class RMI:
  

  def print_RMI(self):

    print("RMI architecture: ",self.model_per_level_list)
    print("RMI vals per model: ",self.num_elements_per_level)
    print("RMI others",sum(self.is_RMI))

    # for i in self.RMI_dict.keys():
    #   RMI_dict[i].print_RMI()

    return  

  def initialize_RMI(self,data_list,cdf_list):

    print("rmi training:",len(data_list),len(cdf_list))

    if len(data_list)<3000:
      print(data_list[:10],cdf_list[:10])

    self.num_levels=2
    self.model_per_level_list=[]
    self.num_elements_per_level=[]
    self.slope_list=[]
    self.bias_list=[]
    self.is_RMI=[]
    self.RMI_dict={}

    self.model_per_level_list=[]
    self.model_per_level_list=[1,math.floor(len(data_list)*1.00/100)]

    # print("using",self.model_per_level_list[1],"models")

    temp_slope,temp_bias=get_slope_bias(data_list,cdf_list,True)

    self.slope_list=[[temp_slope],[]]
    self.bias_list=[[temp_bias],[]]

    self.num_elements_per_level=[[len(data_list)],[]]

    if self.model_per_level_list[1]==0:
      return 

    level_2_data={}
    level_2_cdf={}

    for i in range(0,self.model_per_level_list[1]):
      level_2_data[i]=[]
      level_2_cdf[i]=[]
      self.num_elements_per_level[1].append(0)
      self.is_RMI.append(0)

    max_cdf=max(cdf_list)
    min_cdf=min(cdf_list)  

    for i in range(0,len(data_list)):
      cdf_val=(temp_slope*data_list[i]+temp_bias)
      cdf_val=(cdf_val-min_cdf)*1.00/(max_cdf-min_cdf)
      ind_val=closest_integer((cdf_val)*self.model_per_level_list[1]) 
      ind_val=max(ind_val,0)
      ind_val=min(ind_val,self.model_per_level_list[1]-1)

      level_2_data[ind_val].append(data_list[i])
      level_2_cdf[ind_val].append(cdf_list[i])
      self.num_elements_per_level[1][ind_val]+=1.0

    for i in range(0,self.model_per_level_list[1]):
      # print("for child:",i,len(level_2_data[i]))
      if len(level_2_data[i])>500:
        temp_RMI=RMI()
        temp_RMI.initialize_RMI(level_2_data[i],level_2_cdf[i])
        # print("additional RMI for: ",i)
        self.is_RMI[i]=1
        self.RMI_dict[i]=temp_RMI
      temp_slope,temp_bias=get_slope_bias(level_2_data[i],level_2_cdf[i],True)
      self.slope_list[1].append(temp_slope)
      self.bias_list[1].append(temp_bias)

    self.print_RMI()

    
    return 

  def infer(self,data_val):
    temp_slope=self.slope_list[0][0]    
    temp_bias=self.bias_list[0][0]

    # return (temp_slope*data_val+temp_bias)

    ind_val=closest_integer((temp_slope*data_val+temp_bias)*self.model_per_level_list[1]) 
    ind_val=max(ind_val,0)
    ind_val=min(ind_val,self.model_per_level_list[1]-1)

    if self.is_RMI[ind_val]==1:
      return self.RMI_dict[ind_val].infer(data_val)

    temp_slope=self.slope_list[1][ind_val]
    temp_bias=self.bias_list[1][ind_val]

    cdf_ans=(temp_slope*data_val+temp_bias)

    return cdf_ans


def closest_integer(val):

  temp_int=float(math.floor(val))
  temp_int_1=float(math.ceil(val))
  mid=(temp_int_1+temp_int)/2.00

  if val>mid:
    return int(temp_int_1)
  else:
    return int(temp_int)  

def evaluate(our_list,bool_generate,batch_size):


  # print(our_list)  
  if bool_generate:
    num_elements=len(our_list)
    our_list=np.random.uniform(0,1,num_elements)

  num_elements=len(our_list)
  our_list.sort()  

  # min_val=min(our_list)
  # max_val=max(our_list)

  # for i in range(0,len(our_list)):
  #   our_list[i]=(our_list[i]*1.00-min_val)*1.00/(max_val-min_val)

  out_of_range_count=0  

  cdf_list=[]

  for i in range(0,len(our_list)):
    cdf_list.append(i*1.00/num_elements)

  # temp_rmi=RMI()
  # temp_rmi.initialize_RMI(our_list,cdf_list)
  our_list_old=copy.deepcopy(our_list)
  # batch_size=100
  for i in range(0,closest_integer(len(our_list)*1.00/batch_size)):
    # print("i",i,len(our_list),batch_size)
    temp_our=[]
    temp_cdf=[]
    for j in range(0,batch_size):
      temp_our.append(our_list[i*batch_size+j])
      temp_cdf.append(cdf_list[i*batch_size+j])

    slope,bias=get_slope_bias(temp_our,temp_cdf,False) 

    # print("slope bias",slope,bias)
    
    for j in range(0,batch_size):
      our_list[i*batch_size+j]=slope*temp_our[j]+bias

  for i in range(0,len(our_list)):     
    # print(" we infered: ",temp_rmi.infer(our_list[i]))
    # our_list[i]=temp_rmi.infer(our_list[i])
    if our_list[i]>1.0 or our_list[i]<0.0:
        out_of_range_count+=1
    our_list[i]=min(1.0,our_list[i])
    our_list[i]=max(0.0,our_list[i])

  # # plt.hist(our_list, density=False, bins=100, label="g")  # `density=False` would make counts
  # plt.scatter(our_list,our_list_old, label="g",color='green')  # `density=False` would make counts
  # plt.ylabel('Count')
  # plt.xlabel('Score')
  # # plt.yscale('log')
  # plt.title("Score Distribution of negative rows")
  # # plt.title("Score Distribution of Benign URL's")
  # plt.show()    

  # plt.scatter(cdf_list,our_list_old, label="g",color='red')  # `density=False` would make counts
  # plt.ylabel('Count')
  # plt.xlabel('Score')
  # # plt.yscale('log')
  # plt.title("Score Distribution of negative rows")
  # # plt.title("Score Distribution of Benign URL's")
  # plt.show()    

  # y=1/0.0  

  print("out of range percentage: ",out_of_range_count*1.00/len(our_list))  

  mean=1.00/num_elements

  

  print("data size is: ",len(our_list),num_elements)  
  variance_ratio_list=[1.0,0.01,0.1,0.5]

  for vr in variance_ratio_list:

    temp_list=copy.deepcopy(our_list)

    temp_list.sort()

    new_list=[temp_list[0]]

    diff_list=[]

    for i in range(0,len(temp_list)-1):
      diff=temp_list[i+1]-temp_list[i]
      diff=diff-mean
      diff=diff*1.00*vr
      diff=diff+mean
      diff=max(diff,0)
      diff_list.append(diff*num_elements*1.00)
      # print("diff is: ",diff)
      new_list.append(new_list[i]+diff)

    collide_dict={}

    predicted_collision=predict_collisions(diff_list)

    print("predicted collision for vr:",vr,"is:",predicted_collision)
    
    for i in range(0,len(temp_list)):
      collide_dict[i]=0

    for i in tqdm(range(0,len(new_list))):

      val=closest_integer(new_list[i]*num_elements)
      val=min(len(temp_list)-1,val)
      collide_dict[val]=1

    collide_count=0

    for i in tqdm(collide_dict.keys()):
      if collide_dict[i]!=0:
        # print("i",i)
        collide_count+=1.0

    collision_ratio=(len(temp_list)-collide_count)*1.00/len(temp_list)    

    print("for variance ratio: ",vr," collision ratio is: ",collision_ratio)    

    return collision_ratio


def benchmark_binary_stuff(file_name):
  print(file_name)

  if "log_normal" in file_name:
    mu=0.0
    sigma=2.0
    our_list = np.random.lognormal(mu, sigma, 200000000)
    scale=float(len(our_list)*5)
    for i in range(0,len(our_list)):
      our_list[i]=int(our_list[i]*scale)

  else:
    if "genome" in file_name or "longitude" in file_name:
      our_list=[]
      temp_str="kapil"
      with open(file_name) as f:
        for line in f:
          split=line.split() 

          our_list.append(float(split[0]))

    else:  
      if "64" in file_name:
        f = open(file_name, "r")
        our_list = np.fromfile(f, dtype=np.uint64)
      else:
        f = open(file_name, "r")
        our_list = np.fromfile(f, dtype=np.uint32)


  our_list=our_list[1:]   

  print(our_list[0:5])
      
  print("reading data done")  
  print("size of list:",len(our_list))
  our_list=list(set(our_list))  
  print("size of without duplicates list:",len(our_list))
  our_list=our_list[:25000000]
  print("duplicate removal done")
  print("size of list without duplicates:",len(our_list))  

  temp_list=[]
  for i in range(0,len(our_list)):
    if np.isnan(our_list[i]):
      continue
    temp_list.append(float(our_list[i]))  
    # temp_list.append(i)

  print("nan checking done")  

  temp_list=temp_list[:25000000]  

  temp_list.sort()  

  print("sorting done")  

  # batch_size_list=[100000,10000,1000,100]
  batch_size_list=[100000,10000]
  # batch_size_list=[10000]

  collision_list=[]

  for i in tqdm(range(0,len(batch_size_list))):
    temp_send=copy.deepcopy(temp_list)
    collision=evaluate(temp_send,False,batch_size_list[i]) 
    collision_list.append(collision)

  chunk_list=file_name.split('/')  

  # plt.plot(batch_size_list,collision_list, marker='x',label="g",color='green')  # `density=False` would make counts
  # plt.ylabel('collision proportion')
  # plt.xlabel('Number of data points covered per linear model (15mil points) ')
  # plt.xscale('log')
  # plt.title("Collisions with increasing overfitting "+chunk_list[-1])
  # plt.tight_layout()
  
  # plt.savefig(chunk_list[-1]+".png")
  # plt.title("Score Distribution of Benign URL's")
  # plt.show() 






def benchmark_fpr_range(file_name,acceptance_list,reject_list,dataset_name):

  col_name=[]
  if 'dmv' or 'stock' or 'chicago_taxi' in file_name:
    print("Using pandas to read")
    df=pd.read_csv(file_name)
    print("pandas col names:",df.columns)
  else:
    df=pd.read_csv(file_name,header=None,sep=',')  
    for i in range(0,len(df.columns)):
      col_name.append(str(i))

    df.columns=col_name 

  
  col_list=list(df.columns)
  for i in range(0,len(col_list)):
    if i < max_cols_to_consider:
      if "accept_all" in acceptance_list and col_list[i] not in reject_list:
        continue
      if col_list[i] in acceptance_list:
        continue
    df=df.drop(col_list[i], 1) 

  columns_list=list(df.columns)
  
  df.fillna(0.0)

  print(col_list)
  print('df stuff',df.columns)
  matrix=df.values.tolist()

  our_list=[]

  for i in range(0,len(matrix)):
    our_list.append(matrix[i][1])
    # our_list.append(i)

  our_list=list(dict.fromkeys(our_list))  
  
  temp_list=[]
  for i in range(0,len(our_list)):
    if np.isnan(our_list[i]):
      continue
    temp_list.append(our_list[i])  

  temp_list.sort()  

  cut_val=math.floor(len(temp_list)/1000)
  cut_val=int(cut_val*1000)

  temp_list=temp_list[:cut_val]  

  temp_list.sort()  

  print("sorting done for array size",len(temp_list))   

  batch_size_list=[1000,100,10]
  # batch_size_list=[100000]

  collision_list=[]

  for i in tqdm(range(0,len(batch_size_list))):
    temp_send=copy.deepcopy(temp_list)
    collision=evaluate(temp_send,False,batch_size_list[i]) 
    collision_list.append(collision)

  chunk_list=file_name.split('/')  

  plt.plot(batch_size_list,collision_list, marker='x',label="g",color='green')  # `density=False` would make counts
  plt.ylabel('collision proportion')
  plt.xlabel('Number of data points covered per linear model ')
  plt.xscale('log')
  plt.title("Collisions with increasing overfitting "+chunk_list[-1])
  plt.tight_layout()
  
  plt.savefig(chunk_list[-1]+".png")
  # plt.title("Score Distribution of Benign URL's")
  plt.show() 





  # evaluate(temp_list,False)  

def get_fpr_range(actual_list,our_list,range_size,fpr_val,domain,our_domain,higher_ratio):
  
  if range_size>128:
    query_num=1000000
  else:  
    query_num=1000000

  total_neg=0.0
  fp_rosetta=0.0
  fp_ours=0.0

  our_set=set(our_list)
  actual_set=set(actual_list)

  for i in tqdm(range(0,query_num)):

    random_val=np.random.uniform(0,1)
    actual_start=(int)(random_val*domain)
    our_start=(int)(random_val*our_domain)

    if random_val+range_size>=domain:
      continue


    c_neg=0.0

    for j in range(0,range_size):
      if (actual_start+j) in actual_set:
        c_neg+=1.0
        break

    if c_neg!=0:
      continue

    total_neg+=1.0  

    c_rosetta=1.00-math.pow(1.00-(fpr_val*1.00/math.pow(2,higher_ratio)),range_size*1.00)
    c_ours=0.0  

    our_range_size=int(math.ceil(range_size*our_domain*1.00/domain))

    for j in range(0,our_range_size):
      if our_start+j in our_set:
        c_ours=1.0
        break

    fp_ours+=c_ours
    fp_rosetta+=c_rosetta

  fpr_ours=fp_ours*1.00/total_neg
  fpr_rosetta=fp_rosetta*1.00/total_neg  
  print("for range size ",range_size,"fpr is",fpr_ours,"rosetta:",fpr_rosetta)
  print("total negatives:",total_neg," false posiitves are:",fp_rosetta)

  fpr_surf=fpr_ours*0.04/(math.pow(2,higher_ratio-10.0))

  return fpr_ours,fpr_rosetta,fpr_surf  



def get_gcs_size(our_list,our_domain):

  N=int(len(our_list))
  P=int(our_domain*1.00/len(our_list))
  # P=100

  temp_str=[]

  for i in range(0,N*P+1):
    temp_str.append("0")

  print("checking",max(our_list),N*P)  
  for i in range(0,N):
    temp_str[our_list[i]]="1"  

  temp_str=''.join(temp_str)
  # os.remove("encode.out")
  # os.remove("lmao.out")

  try:
    os.remove("encode.out")
  except OSError:
    pass

  try:
    os.remove("lmao.out")
  except OSError:
    pass    

  f_ac=open("encode.out", "w")
  f_ac.write(temp_str)
  f_ac.close()



  ar= arcode.ArithmeticCode(True)
  ar.encode_file("encode.out","lmao.out")

  file_size_ac=os.path.getsize("lmao.out")*8

  # y=1/0.0  

  print("N,P",N,P)

  gcs_local=GCSBuilder(N,P)

  our_list.sort()

  for i in range(0,N):
    # temp_rand=random.uniform(0,1)
    # gcs.add(str(rand_list[i]).encode('utf-8'))
    # gcs_local.add(temp_rand)
    gcs_local.add(our_list[i])


  with open("table1.gcs", "wb") as f:
      gcs_local.finalize(f)
      fsize = f.tell() 

  final_size=fsize*8

  optimal_size=N*math.log(P,2)

  print("size ratio is:",final_size*1.00/optimal_size)
  print("bits per key: ",final_size*1.00/N," optimal bf bits per key:",optimal_size*1.00/N)

  print("size ratio AC is:",file_size_ac*1.00/optimal_size)
  print("AC bits per key: ",file_size_ac*1.00/N," optimal bf bits per key:",optimal_size*1.00/N)
  
  return final_size*1.00/optimal_size,file_size_ac*1.00/optimal_size



def get_data(file_name,batch_size):
  our_list=[]
  if "seq" in file_name:
    for i in range(0,int(math.pow(2,20))):
      our_list.append(i)  
  else:  
    if "64" in file_name:
      f = open(file_name, "r")
      our_list = np.fromfile(f, dtype=np.uint64)
    else:
      f = open(file_name, "r")
      our_list = np.fromfile(f, dtype=np.uint32)

    our_list=our_list[1:]   
      
  print("reading data done")  
  print("size of list:",len(our_list))
  our_list=our_list[:100000]
  our_list=list(set(our_list))  
  print("duplicate removal done")
  print("size of list without duplicates:",len(our_list))  

  temp_list=[]
  for i in range(0,len(our_list)):
    if np.isnan(our_list[i]):
      continue
    temp_list.append(float(our_list[i]))  
    # temp_list.append(i)

  factor=math.floor(len(temp_list)/batch_size)
  temp_list=temp_list[:int(factor*batch_size)]  

  our_list=copy.deepcopy(temp_list)
  
  our_list.sort()

  num_elements=len(our_list)

  cdf_list=[]

  for i in range(0,len(our_list)):
    cdf_list.append(i*1.00/num_elements)
    
  
  for i in range(0,closest_integer(len(our_list)*1.00/batch_size)):
    # print("i",i,len(our_list),batch_size)
    temp_our=[]
    temp_cdf=[]
    for j in range(0,batch_size):
      temp_our.append(our_list[i*batch_size+j])
      temp_cdf.append(cdf_list[i*batch_size+j])

    slope,bias=get_slope_bias(temp_our,temp_cdf,False) 

    # print("slope bias",slope,bias)
    
    for j in range(0,batch_size):
      our_list[i*batch_size+j]=slope*temp_our[j]+bias

  for i in range(0,len(our_list)):     
    # print(" we infered: ",temp_rmi.infer(our_list[i]))
    # our_list[i]=temp_rmi.infer(our_list[i])
    # if our_list[i]>1.0 or our_list[i]<0.0:
    #     out_of_range_count+=1
    our_list[i]=min(1.0,our_list[i])
    our_list[i]=max(0.0,our_list[i])  

  return our_list  



def benchmark_range_filter(file_name):
  batch_size=1000
  our_list=get_data(file_name,batch_size)

  # num_elements=math.pow(2,20)
  # our_list=np.random.uniform(0,1,(int)(num_elements))

  num_elements=len(our_list)

  #10 bits of Surf for 0.04 

  # fpr_val=1.00/math.pow(2,6)

  fpr_val=0.01

  domain=int(len(our_list)*math.pow(2,44))
  our_domain=int(len(our_list)*1.00/fpr_val)


  actual_list=copy.deepcopy(our_list)
  for i in range(0,len(our_list)):
    actual_list[i]=int(our_list[i]*domain)
    our_list[i]=int(our_list[i]*our_domain)

  range_size_list=[1,8,64,256]
  # range_size_list=[1,8,16]
  fpr_our_list=[]
  fpr_rosetta_list=[]
  fpr_surf_list=[]
  fpr_ac_list=[]

  # blocks_size=1000

  higher_ratio,higher_ratio_ac=get_gcs_size(our_list,our_domain)

  for range_size in tqdm(range_size_list):  
    fpr_ours,fpr_rosetta,fpr_surf=get_fpr_range(actual_list,our_list,range_size,fpr_val,domain,our_domain,higher_ratio)  
    fpr_our_list.append(fpr_ours)
    fpr_rosetta_list.append(fpr_rosetta)
    fpr_surf_list.append(fpr_surf)
    fpr_ac_list.append(fpr_ours*math.pow(2,higher_ratio_ac*1.00/higher_ratio))
  
  
  print(range_size_list)
  print(fpr_our_list)
  print(fpr_ac_list)
  print(fpr_surf_list)
  print(fpr_rosetta_list)  


  # fpr_list=[0.1,0.05,0.01,0.005,0.001]  
  # for curr_fpr in tqdm(fpr_list):  
  #   our_size=get_gcs_size(our_list,domain)
  #   surf_size=get_surf_size(our_list,fpr_surf)

  slpit_file=file_name.split('/')

  plt.plot(range_size_list,fpr_our_list, marker='x',label="Our GCS",color='green')  # `density=False` would make counts
  plt.plot(range_size_list,fpr_ac_list, marker='x',label="Our Arithmetic Coding",color='orange')  # `density=False` would make counts
  plt.plot(range_size_list,fpr_rosetta_list, marker='x',label="Rosetta Assuming optimal BF",color='red')  # `density=False` would make counts
  plt.plot(range_size_list,fpr_surf_list, marker='x',label="SuRF Best option",color='blue')  # `density=False` would make counts
  plt.ylabel('FPR acheived')
  plt.xlabel('Range Size')
  plt.xscale('log')
  plt.yscale('log')
  plt.title(""+str(slpit_file[-1])+" ratio is: "+str(round(higher_ratio,3))+","+str(round(higher_ratio_ac,3)))
  plt.legend()
  plt.tight_layout()
  plt.savefig("range_filter_"+str(slpit_file[-1])+".png")
  # plt.show()  

def benchmark_space_range_filter(file_name):
  batch_size=1000
  our_list=get_data(file_name,batch_size)

  num_elements=len(our_list)

  #10 bits of Surf for 0.04 

  fpr_val_list=[0.1,0.05,0.01,0.005]
  domain=int(len(our_list)*math.pow(2,44))
  
  actual_list=copy.deepcopy(our_list)
  for i in range(0,len(our_list)):
    actual_list[i]=int(our_list[i]*domain)
  
  fpr_our_list=[]
  fpr_rosetta_list=[]
  fpr_surf_list=[]
  fpr_ac_list=[]
  space_list=[]

  for fpr_val in tqdm(fpr_val_list):  
    our_domain=int(len(our_list)*1.00/fpr_val)

    our_list_temp=copy.deepcopy(our_list)
    for i in range(0,len(our_list)):
      our_list_temp[i]=int(our_list[i]*our_domain)

    higher_ratio,higher_ratio_ac=get_gcs_size(our_list_temp,our_domain)  

    fpr_ours,fpr_rosetta,fpr_surf=get_fpr_range(actual_list,our_list_temp,1,fpr_val,domain,our_domain,higher_ratio)

    print("data: ",len(our_list),fpr_val,higher_ratio)

    fpr_our_list.append(fpr_ours)
    fpr_rosetta_list.append(fpr_rosetta)
    fpr_surf_list.append(fpr_surf)
    fpr_ac_list.append(fpr_ours*math.pow(2,higher_ratio_ac*1.00/higher_ratio))
    space_list.append(math.log(1.00/fpr_val,2)*higher_ratio)
  
  
  # print(range_size_list)
  print(space_list)
  print(fpr_our_list)
  print(fpr_surf_list)
  print(fpr_rosetta_list)  

  # fpr_list=[0.1,0.05,0.01,0.005,0.001]  
  # for curr_fpr in tqdm(fpr_list):  
  #   our_size=get_gcs_size(our_list,domain)
  #   surf_size=get_surf_size(our_list,fpr_surf)

  split_file=file_name.split('/')
  plt.plot(space_list,fpr_our_list, marker='x',label="Our GCS",color='green')  # `density=False` would make counts
  plt.plot(space_list,fpr_ac_list, marker='x',label="Our Arithmetic Coding",color='orange')  # `density=False` would make counts
  plt.plot(space_list,fpr_rosetta_list, marker='x',label="Rosetta Assuming optimal BF",color='red')  # `density=False` would make counts
  plt.plot(space_list,fpr_surf_list, marker='x',label="SuRF Best option",color='blue')  # `density=False` would make counts
  plt.ylabel('FPR acheived')
  plt.xlabel('Space Used (bits per key)')
  # plt.xscale('log')
  plt.yscale('log')
  plt.title(""+str(split_file[-1])+" ratio is: "+str(round(higher_ratio,3))+","+str(round(higher_ratio_ac,3)))
  plt.legend()
  plt.tight_layout()
  plt.savefig("range_filter_space_"+str(split_file[-1])+".png")
  # plt.show()      
  





# benchmark_range_filter()

# y=1/0.0


dataset=sys.argv[1]
output_file_name=sys.argv[2]
max_row_consider=int(sys.argv[3])
max_cols_to_consider=int(sys.argv[4])

if "stocks" in dataset:
  #Weather
  file_name="../range_filters/small_stocks.csv"
  reject_list=[]
  acceptance_list=["open","close","high","low","adj_close","volume"]
  # acceptance_list=["open","close","high","low"]
  output_file_name_dtree="dtree_"+output_file_name
  output_file_name_dtree_block="dtree_block_"+output_file_name
  output_file_name_vortex="vortex_"+output_file_name
  output_file_name_lb="lb_"+output_file_name
  output_file_name_multi_list="multi_list_"+output_file_name
  output_file_name_karl="karl_"+output_file_name
  output_file_name_data="results/"+dataset
  query_list=[3]



  # benchamrk_bloom_filter_real(file_name,acceptance_list,reject_list,max_cols_to_consider,output_file_name_dtree,output_file_name_dtree_block,max_row_consider)
  # benchmark_vortex(file_name,acceptance_list,reject_list,max_cols_to_consider,output_file_name_vortex)  
  # benchmark_lb(file_name,acceptance_list,reject_list,max_cols_to_consider,output_file_name_lb)
  # benchmark_mult_list(file_name,acceptance_list,reject_list,max_cols_to_consider,output_file_name_multi_list)
  # benchmark_karl(file_name,acceptance_list,reject_list,max_cols_to_consider,output_file_name_karl)
  benchmark_fpr_range(file_name,acceptance_list,reject_list,output_file_name_data) 


if "chicago_taxi" in dataset:
  # print("here")
  #Weather
  file_name="../range_filters/multi_dim_range_filters/data/chicago_taxi_small.csv"
  reject_list=[]
  acceptance_list=["Trip Seconds","Trip Miles","Fare","Tips","Trip Total"]
  # acceptance_list=["Trip Seconds","Trip Total"]
  # acceptance_list=["Trip Seconds","Trip Miles","Trip Total"]
  # acceptance_list=["Fare","Trip Total"]
  output_file_name_dtree="dtree_"+output_file_name
  output_file_name_dtree_block="dtree_block_"+output_file_name
  output_file_name_vortex="vortex_"+output_file_name
  output_file_name_lb="lb_"+output_file_name
  output_file_name_multi_list="multi_list_"+output_file_name
  output_file_name_karl="karl_"+output_file_name
  output_file_name_data="results/"+dataset
  query_list=[3]

  benchmark_fpr_range(file_name,acceptance_list,reject_list,output_file_name_data) 

if "wiki" in dataset:
  # print("here")
  #Weather
  file_name="data/wiki_ts_200M_uint64"

  # benchmark_binary_stuff(file_name)
  benchmark_range_filter(file_name)
  # benchmark_space_range_filter(file_name)

if "fb" in dataset:
  # print("here")
  #Weather
  file_name="data/fb_200M_uint64"

  # benchmark_binary_stuff(file_name)
  benchmark_range_filter(file_name)
  # benchmark_space_range_filter(file_name)  

if "osm" in dataset:
  # print("here")
  #Weather
  file_name="data/osm_cellids_200M_uint64"

  # benchmark_binary_stuff(file_name)
  benchmark_range_filter(file_name)
  # benchmark_space_range_filter(file_name)    

if "books64" in dataset:
  # print("here")
  #Weather
  file_name="data/books_200M_uint64"

  # benchmark_binary_stuff(file_name)
  benchmark_range_filter(file_name)
  # benchmark_space_range_filter(file_name) 

if "books32" in dataset:
  # print("here")
  #Weather
  file_name="data/books_200M_uint32"

  # benchmark_binary_stuff(file_name)
  benchmark_range_filter(file_name)
  # benchmark_space_range_filter(file_name)

if "genome_small" in dataset:
  # print("here")
  #Weather
  file_name="data/genome_data_small.csv"

  # benchmark_binary_stuff(file_name)
  benchmark_range_filter(file_name)
  # benchmark_space_range_filter(file_name)   

if "seq" in dataset:
  # print("here")
  #Weather
  file_name="seq"

  # benchmark_binary_stuff(file_name)
  benchmark_range_filter(file_name)
  # benchmark_space_range_filter(file_name)


if "log_normal" in dataset:
  # print("here")
  #Weather
  file_name="log_normal"

  # benchmark_binary_stuff(file_name)
  benchmark_range_filter(file_name)
  # benchmark_space_range_filter(file_name)       

if "longitude" in dataset:
  print("OSM LONG")
  #Weather
  file_name="data/longitude.csv"

  # benchmark_binary_stuff(file_name)
  benchmark_range_filter(file_name)
  # benchmark_space_range_filter(file_name)         

