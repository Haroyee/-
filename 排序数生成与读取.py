import random

def RandomG(num):#生成随机数

    file1= open("待排数组.txt",'w')

    array = [random.randint(1,num) for i in range(num)]

    file1.writelines(str(array))
    file1.close()
def LoadFile():#读取随机数
    filename = "待排数组.txt"
    file = open(filename, 'r')
    List = eval(file.readline())
    file.close()
    return List

# RandomG(100)

# for i in eval(file2):
#     print(i)



