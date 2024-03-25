import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #导入绘图库
import 排序数生成与读取 #导入排序数生成文件
import 排序比较 #导入排序比较文件
import 排序可视化 #导入排序可视化文件
import timeit #导入统计时间的库

pd.set_option('display.unicode.east_asian_width', True) #设置输出右对齐，此代码写入脚本中
pd.set_option('display.unicode.ambiguous_as_wide', True) #实际作用不明，可以不写

Sortname = ["直接插入排序", "折半插入排序", "希尔排序",
            "冒泡排序", "快速排序", "简单选择排序",
            "堆排序", "归并排序", "计数排序",
            "基数排序"]   #排序名称列表

def SortVal(List, name):#可视化调用
    if name == Sortname[0]:
        排序可视化.InsertSort(List)
    elif name == Sortname[1]:
        排序可视化.BInsertSort(List)
    elif name == Sortname[2]:
        排序可视化.ShallSort(List)
    elif name == Sortname[3]:
        排序可视化.BubleSort(List)
    elif name == Sortname[4]:
        ax = plt.subplot()
        排序可视化.QuickSort(List, 0, len(List) - 1, ax, timeit.default_timer())
    elif name == Sortname[5]:
        排序可视化.SelectSort(List)
    elif name == Sortname[6]:
        排序可视化.HeadSort(List)
    elif name == Sortname[7]:
        ax = plt.subplot()
        排序可视化.MergeSort(List, 0, len(List) - 1, ax, timeit.default_timer())
    elif name == Sortname[8]:
        排序可视化.CountSort(List)
    elif name == Sortname[9]:
        排序可视化.RadixSort(List)

    排序可视化.dawnend(List)
    return


def SortTime(List, name):#比较调用
    if name == Sortname[0]:
        step = 排序比较.InsertSort(List)
    elif name == Sortname[1]:
        step = 排序比较.BInsertSort(List)
    elif name == Sortname[2]:
        step = 排序比较.ShallSort(List)
    elif name == Sortname[3]:
        step = 排序比较.BubleSort(List)
    elif name == Sortname[4]:
        step = 排序比较.QuickSort(List, 0, len(List) - 1, [0, 0])
    elif name == Sortname[5]:
        step = 排序比较.SelectSort(List)
    elif name == Sortname[6]:
        step = 排序比较.HeadSort(List)
    elif name == Sortname[7]:
        step = 排序比较.MergeSort(List, 0, len(List) - 1)
    elif name == Sortname[8]:
        step = 排序比较.CountSort(List)
    elif name == Sortname[9]:
        step = 排序比较.RadixSort(List)
    return step


def Meus():#主菜单
    while True:
        while True:
            print()
            print("======================================")
            print("1.排序可视化                  2.排序比较")
            print("3.退出                                ")
            print("======================================")
            choice1 = eval(input("请输入你要执行的操作:"))
            if choice1 == 3:
                return
            elif choice1 <= 0 and choice1 > 3:
                print("输入错误！")
                continue
            else:
                break

        if choice1 == 1:
            while True:
                num = eval(input("请输入要排序的数的个数(限制范围0-50)："))
                if 0 <= num and num <= 50:
                    break
                else:
                    print("输入错误！")
            排序数生成与读取.RandomG(num)

        elif choice1 == 2:
            num = eval(input("请输入要排序的数的个数："))
            排序数生成与读取.RandomG(num)


        while True:
            while True:
                print()
                print("=============================================")
                print("1.插入排序                    2.折半排序")
                print("3.希尔排序                    4.冒泡排序")
                print("5.快速排序                    6.选择排序")
                print("7.堆排序                      8.归并排序")
                print("9.计数排序                    10.基数排序")
                print("11.全部排序                   12.设置排序数个数")
                print("13.返回主菜单                 ")
                print("=============================================")
                choice2 = eval(input("请输入你要执行的操作："))
                if choice2 == 13:
                    break
                elif choice2 == 12 and choice1==1:
                    while True:
                        num = eval(input("请输入要排序的数的个数(限制范围0-50)："))
                        if 0<=num and num <=50:
                            break
                        else:
                            print("输入错误！")
                    排序数生成与读取.RandomG(num)
                elif choice2 == 12 and choice1 == 2:
                    num = eval(input("请输入要排序的数的个数："))
                    排序数生成与读取.RandomG(num)
                elif choice2 > 13 and choice2 < 1:
                    print("输入错误!")
                    continue
                else:
                    break

            if choice2 == 13:
                break

            if choice1 == 1:

                ls = 排序数生成与读取.LoadFile()
                if choice2 != 11:
                    List = ls.copy()
                    SortVal(List, Sortname[choice2 - 1])
                    plt.pause(0)
                    Enter = input("输入回车继续")
                else:
                    for i in range(10):
                        List = ls.copy()
                        SortVal(List, Sortname[i])
                    plt.pause(0)
                    Enter = input("输入回车继续")

            else:

                ls = 排序数生成与读取.LoadFile()
                if choice2 != 11:
                    List = ls.copy()

                    start = timeit.default_timer()
                    step = SortTime(List, Sortname[choice2-1])
                    end = timeit.default_timer()
                    print("\n{}".format(Sortname[choice2-1]))
                    print("比较步数：{}\n移动步数：{}".format(step[0], step[1]))
                    print('时间: {:0.3f} s'.format(end - start))

                    #将排序后的数保存
                    file1 = open("排序结果.txt", 'w')
                    file1.writelines(str(List))
                    file1.close()

                    Enter = input("输入回车继续")
                else:
                    Data = [[], [],[]]
                    for i in range(10):
                        List = ls.copy()

                        start = timeit.default_timer()
                        step = SortTime(List, Sortname[i])
                        end = timeit.default_timer()

                        Data[0].append(step[0])
                        Data[1].append(step[1])
                        Data[2].append('{:0.3f} s'.format(end - start))
                    Columns = ['比较次数', '移动次数', '时间']
                    Index = Sortname
                    df = pd.DataFrame(data=np.array(Data).T, index=Index,columns=Columns)
                    print()
                    print(df)
                    df.to_csv('运行结果.txt')

                    # 将排序后的数保存
                    file1 = open("排序结果.txt", 'w')
                    file1.writelines(str(List))
                    file1.close()

                    Enter = input("输入回车继续")


if __name__ == "__main__":
      Meus()
