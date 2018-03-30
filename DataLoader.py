import csv
import os
import numpy as np
import pandas as pd

class DataLoader(object):
  def __init__(self,dataset_name="",class_index=-1,train=True):

        self.dataset_name=dataset_name
        self.class_index=class_index
        self.train=train
        # call load_data methof
        self. load_data(self.dataset_name,self.class_index,self.train)

  def load_data(self,dataset, class_index,train):

      # read the data
      data=[]
      try:

          dataset_file = csv.reader(open(os.getcwd() + "\\data\\" + dataset + ".csv", 'r'), quoting=csv.QUOTE_NONNUMERIC)
          #str(dataset_file).encode("utf-8").decode("ascii")
          #data = pd.read_csv(filepath_or_buffer=os.getcwd() + "\\data\\" + dataset + ".csv", header=None, dtype={ 'my_column': np.float64 },  na_values=['n/a'])

          index = 0
          i=0
          for row in dataset_file:
                try:
                    data.append(np.float32(row))
                   
                except (TypeError , ValueError) as e:
                    print(e)
                   
      except FileNotFoundError:
          print("File Not Found")

      dataset = np.array(data)
      print("[*] Dataset shape : ", dataset.shape)
      #dataset = dataset.transpose()

      # count the number of clases
      class_list = list(set(dataset[:, class_index]))
      print("[*] Data classes are : ", class_list)

      # separate to cancer_dataset and normal_dataset
      cancer_dataset = []
      normal_dataset = []

      if class_index < 0:
          class_index = len(dataset[0]) + class_index

      for i in range(0, dataset.shape[0]):
          item = pd.DataFrame(dataset[i]).drop(index=class_index)  # remove the class lable from the data

          if dataset[i, class_index] == class_list[0]:

              cancer_dataset.append(item.values.reshape(item.shape[0]))

          else:
              normal_dataset.append(item.values.reshape(item.shape[0]))

      print("[*] Cancer dataset shape = ", np.shape(cancer_dataset))
      print("[*] Normal dataset shape = ", np.shape(normal_dataset))

      
      #case 1:  if the model is used for training
      if train:
          # TODO uses different split vals than .8'
          cancer_training_size = np.int(len(cancer_dataset) * .8)
          cancer_testing_size = len(cancer_dataset) - cancer_training_size
          print("[*] Number of Samples in cancer_dataset: ", len(cancer_dataset), " training=", cancer_training_size,
                " testing=",
                cancer_testing_size)
          # normal dataset
          normal_training_size = np.int(len(normal_dataset) * .8)
          normal_testing_size = len(normal_dataset) - normal_training_size

          print("[*] Number of Samples in normal_dataset: ", len(normal_dataset), " training=", normal_training_size,
                " testing=",
                normal_testing_size)

          # choosing random samples for training and testing files from cancer and test data
          cancer_training_dataset, cancer_testing_dataset = self.chooseRandom(cancer_dataset, cancer_training_size,
                                                                              cancer_testing_size)
          normal_training_dataset, normal_testing_dataset =self. chooseRandom(normal_dataset, normal_training_size,
                                                                              normal_testing_size)

          # prepare for training: Stack the cancer and normal samples
          self.data_train = np.vstack((cancer_training_dataset, normal_training_dataset))
          self.data_test = np.vstack((cancer_testing_dataset, normal_testing_dataset))
          self.cancer_training_size=len(cancer_training_dataset)
          self.cancer_testing_size = len(cancer_testing_dataset)
          #print("[*] cancer trainign size = ", self.cancer_training_size)
          #print("[*] cancer testing size = ", self.cancer_testing_size)

          dir= os.path.join("Representations",self.dataset_name)
          if not os.path.exists(dir):
              os.makedirs(dir)


          np.savetxt(os.path.join(dir,"train_data.csv"),self. data_train, delimiter=",")
          np.savetxt(os.path.join(dir,"test_data.csv"), self.data_test, delimiter=",")
          
          
          #case 2: if the model id used for testing
      else:

          self.data_train = None
          cancer_testing_dataset= np.array(cancer_dataset)
          normal_testing_dataset=np.array(normal_dataset)
          self.data_test = np.vstack((cancer_testing_dataset, normal_testing_dataset))
          self.cancer_training_size = 0
          self.cancer_testing_size = len(cancer_dataset)


  def chooseRandom(self,data, train_Size, test_Size):
      item = range(0, len(data))
      train_ids = np.random.choice(len(data), train_Size, replace=False)
      test_ids = np.array(list(set(item) - set(train_ids)))
      # print("train_ids =", train_ids)
      # print("test_ids =", test_ids)
      data = np.array(data)
      train_dataset = data[train_ids, :]
      test_dataset = data[test_ids, :]

      return train_dataset, test_dataset

