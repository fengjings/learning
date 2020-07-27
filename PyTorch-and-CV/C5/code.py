# -*- coding: utf-8 -*-

class Student:
    
    student_Count = 0
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        Student.student_Count += 1
    
    def dis_student(self):
        print("Student name:",self.name,"Student age:",self.age)
        
student1 = Student("Tang", "20") #创建第1 个Student 对象
student2 = Student("Wu", "22") #创建第2 个Student 对象

student1.dis_student()
student2.dis_student()
print("Total Student:", Student.student_Count)

class People:
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def dis_name(self):
        print("name is:",self.name)

    def set_age(self, age):
        self.age = age
        
    def dis_age(self):
        print("age is:",self.age)
        
class Student(People):
    def __init__(self, name, age, school_name):
        self.name = name
        self.age = age
        self.school_name = school_name

    def dis_student(self):
        print("school name is:",self.school_name)
        
student = Student("Wu", "20", "GLDZ") #创建一个Student 对象
student.dis_student() #调用子类的方法
student.dis_name() #调用父类的方法
student.dis_age() #调用父类的方法
student.set_age(22) #调用父类的方法
student.dis_age() #调用父类的方法\


class Parent: #定义父类
    
    def __init__(self):
        pass
    
    def print_info(self):
        print("This is Parent.")
        
class Child(Parent): #定义子类
    
    def __init__(self):
        pass

    def print_info(self): #对父类的方法进行重写
        print("This is Child.")
        
child = Child()
child.print_info()


import numpy as np

print(np.array([[1,2,3],[4,5,6]]))

a = np.ones([2,3]) #创建全1 的数组
a[1,2] = 2 #对数组中的元素进行覆盖

print(a)

print(np.zeros([2,3])) #创建全0 的数组

print(np.empty([2, 3])) #创建随机初始化的数组


a = np.ones([2,3]) #创建全1 的数组
print(a.ndim)

print(a.shape)

a = np.ones([2,3]) #创建全1 的数组
print(a.dtype)

a = np.ones([2,3], dtype= np.int32) #创建全1 的数组
print(a.dtype)


import numpy as np
a = np.array([[1,2,3],
[3,2,1]])

print("min of array:",a.min())
print("min of array:",a.min(axis=0))
print("min of array:",a.min(axis=1))
print("max of array:",a.max())
print("max of array:",a.max(axis=0))
print("max of array:",a.max(axis=1))
print("sum of array:",a.sum())
print("sum of array:",a.sum(axis=0))
print("sum of array:",a.sum(axis=1))



import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

np.random.seed(42)
x = np.random.randn(30)
plt.plot(x, "r--o")
plt.savefig("xiantu.jpg") 

##################################################################
plt.clf() # 清图。
a = np.random.randn(30)
b = np.random.randn(30)
c = np.random.randn(30)
d = np.random.randn(30)
plt.plot(a, "r--o", b, "b-*", c, "g-.+", d, "m:x")
plt.savefig("xiantiaoyanse.jpg") 

##################################################################
plt.clf() # 清图。

np.random.seed(42)
x = np.random.randn(30)
y = np.random.randn(30)
plt.title("Example")
plt.xlabel("X")
plt.ylabel("Y")

X, = plt.plot(x, "r--o")
Y, = plt.plot(y, "b-*")
plt.legend([X, Y], ["X", "Y"])

plt.savefig("biaoqiantuli.jpg") 

############################################################
plt.clf() # 清图。
a = np.random.randn(30)
b = np.random.randn(30)
c = np.random.randn(30)
d = np.random.randn(30)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

A, = ax1.plot(a, "r--o")
ax1.legend([A], ["A"])
B, = ax2.plot(b, "b-*")
ax2.legend([B], ["B"])
C, = ax3.plot(c, "g-.+")
ax3.legend([C], ["C"])
D, = ax4.plot(d, "m:x")
ax4.legend([D], ["D"])

plt.savefig("zitu.jpg") 


############################################################
plt.clf() # 清图。
np.random.seed(42)
x = np.random.randn(30)
y = np.random.randn(30)

plt.scatter(x,y, c="g", marker="o", label="(X,Y)")
plt.title("Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc=1)
plt.savefig("sandiantu.jpg") 

#########################################################
plt.clf() # 清图。
np.random.seed(42)
x = np.random.randn(1000)

plt.hist(x,bins=20,color="g")
plt.title("Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("zhifangtu.jpg") 

#########################################################
plt.clf() # 清图。
labels = ['Dos', 'Cats', 'Birds']
sizes = [15, 50, 35]

plt.pie(sizes, explode=(0, 0, 0.1), labels=labels, autopct='%1.1f%%',startangle=90)
plt.axis('equal')
plt.savefig("bingtu.jpg") 




