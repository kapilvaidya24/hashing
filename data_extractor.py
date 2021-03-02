count=0

import pandas as pd

# df=pd.read_csv("osm_lat_dataset.csv")
# df.fillna(0.0)
# matrix=df.values.tolist()

# for i in range(0,len(matrix)):
#   print(matrix[i][1])


with open("osm_lat_dataset.csv") as f:
    # content = f.readlines()

    for line in f:
      # print(line)

      # if count>10:
      #   break
      count+=1    

      if count>3680 and count<3690:
        print(count,line)
      else:
        continue  

      # if count%2==0:
      #   continue
      

      # split=line.split()  

      # print(split[0])



