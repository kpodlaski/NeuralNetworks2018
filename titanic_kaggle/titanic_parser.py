import pandas
import numpy as np

def pclassTovector(pclass):
    v = np.zeros(3)
    v[pclass-1]=1
    return v

def sexToVector(sex):
    if sex == 'male' : return [1,0]
    return [0,1]

def ageScale(age):
    return age/max_age

file = open("train.csv",'r')
content = file.readlines()
lines = len(content)
for i in range(0,lines):
    content[i]=content[i].strip().split(",")
    print(content[i])
file.close()
data_frame =pandas.read_csv("train.csv")
print("========")
#print(data_frame)
cols = list(data_frame.columns.values)

# TESTING OPERATIONS
# print(data_frame)
# data_frame.Pclass=data_frame.Pclass.astype(object)
# data_frame.Pclass=data_frame.Pclass.apply(pclassTovector)
# print(data_frame.Pclass)
# data_frame.Sex=data_frame.Sex.astype(object)
# data_frame.Sex=data_frame.Sex.apply(sexToVector)
# print(data_frame.Sex)
#
# data_frame.Age = data_frame.Age.astype(float)
# max_age=data_frame.Age.max()
# data_frame.Age=data_frame.Age.apply(lambda x: x/max_age)
# print(data_frame.Age)

max_age = data_frame.Age.max()
operations = {2:pclassTovector, 4:sexToVector, 5:ageScale}
for col in operations.keys():
    data_frame[cols[col]] = data_frame[cols[col]].astype(object)
    data_frame[cols[col]] = data_frame[cols[col]].apply(operations[col])
print(data_frame.Age)

end_columns = []
end_columns.append(cols.pop(3))
end_columns.append(cols.pop(0))
end_columns.reverse()
data_frame = data_frame[cols + end_columns]

