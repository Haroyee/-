import matplotlib.pyplot as plt
import timeit

Pause_time = 0.0000001 #绘图每帧显示时间
# 设置RC参数字体，让其支持中文
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def InsertSort(List):#插入排序
    start = timeit.default_timer()
    length = len(List)  # 获取列表长度
    m = max(List)  # 获取最大值
    ax = plt.subplot()
    if length <= 1:
        return 0

    if length <= 1:
        return List
    for i in range(1, length):
        j = i
        temp = List[i]  # 每次待插入的元素

        ax.cla()
        ax.bar(range(len(List)), List, color="steelblue")
        ax.bar(i, List[i], color="red")
        plt.title("直接插入排序")
        end = timeit.default_timer()
        ax.text(0, m, "Time = {:0.3f} s".format(end - start))
        ax.bar(i, List[i], color="red")
        plt.pause(Pause_time)

        while j > 0 and temp < List[j - 1]:
            List[j] = List[j - 1]  # 整体后移,为temp腾空位置
            j -= 1

            ax.cla()
            ax.bar(range(len(List)), List, color="steelblue")

            ax.bar(j, List[j], color="green")
            plt.title("直接插入排序")
            end = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end - start))
            plt.pause(Pause_time)

        List[j] = temp  # 将temp插入到空位

        ax.cla()
        ax.bar(range(len(List)), List, color="steelblue")
        ax.bar(j, List[j], color="red")
        plt.title("直接插入排序")
        end = timeit.default_timer()
        ax.text(0, m, "Time = {:0.3f} s".format(end - start))
        plt.pause(Pause_time)

    return end - start


def BInsertSort(List):#折半插入排序
    start = timeit.default_timer()
    length = len(List)
    m = max(List)  # 获取最大值
    ax = plt.subplot()

    if length <= 1:
        return 0

    if length <= 1:
        return List
    for i in range(1, length):
        j = i
        temp = List[i]  # 每次待插入的元素
        low = 0  # 二分查找最小索引
        high = j - 1  # 二分查找最大索引

        ax.cla()
        ax.bar(range(length), List, color="steelblue")
        ax.bar(i, List[i], color="red")
        ax.bar(low, List[low], color="yellow")
        ax.bar(high, List[high], color="yellow")

        while high >= low:  # 二分查找
            mid = (high + low) // 2  # 取中点

            ax.bar(mid, List[mid], color="green")
            plt.title("折半插入排序")
            end = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end - start))
            plt.pause(Pause_time)

            if temp < List[mid]:
                high = mid - 1  # 范围在前半段，接下来在前半段搜索

                ax.cla()
                ax.bar(range(length), List, color="steelblue")
                ax.bar(i, List[i], color="red")
                ax.bar(low, List[low], color="yellow")
                ax.bar(high, List[high], color="yellow")

            else:
                low = mid + 1  # 范围在后半段，接下来在后半段搜索

                ax.cla()
                ax.bar(range(length), List, color="steelblue")
                ax.bar(i, List[i], color="red")
                ax.bar(low, List[low], color="yellow")
                ax.bar(high, List[high], color="yellow")

        List[high + 2: j + 1] = List[high + 1: j]  # 整体后移，为temp腾位
        # while j > high + 1 :
        #     List[j] = List[j-1]
        #     j -= 1
        List[high + 1] = temp  # 将temp插入到空位

        ax.cla()
        ax.bar(range(length), List, color="steelblue")
        ax.bar(high + 1, List[high + 1], color="red")
        plt.title("折半插入排序")
        end = timeit.default_timer()
        ax.text(0, m, "Time = {:0.3f} s".format(end - start))
        plt.pause(Pause_time)

    return end - start


def ShallSort(List):  # 希尔排序

    start = timeit.default_timer()
    ax = plt.subplot()

    m = max(List)  # 获取最大值

    def ShallInsert(d):  # 以d为递增量进行排序
        length = len(List)
        for i in range(d, length):
            j = i
            temp = List[i]  # 待插入元素

            ax.cla()
            ax.bar(range(length), List, color="steelblue")
            ax.bar(i, List[i], color="red")
            plt.title("希尔排序")
            end = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end - start))
            plt.pause(Pause_time)

            while j - d + 1 > 0 and temp < List[j - d]:  # 整体后移，为temp腾空位置
                List[j] = List[j - d]
                j -= d

                ax.cla()
                ax.bar(range(length), List, color="steelblue")
                ax.bar(j, List[j], color="green")
                plt.title("希尔排序")
                end = timeit.default_timer()
                ax.text(0, m, "Time = {:0.3f} s".format(end - start))
                plt.pause(Pause_time)

            List[j] = temp  # 将temp插入空位

            ax.cla()
            ax.bar(range(length), List, color="steelblue")
            ax.bar(j, List[j], color="red")
            plt.title("希尔排序")
            end = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end - start))
            plt.pause(Pause_time)
        return end - start

    length = len(List)

    if length <= 1:
        return 0

    d = length // 2  # 递增量
    while d >= 1:
        end =ShallInsert(d)
        d = d // 2
    return end - start


def BubleSort(List):#冒泡排序
    start = timeit.default_timer()
    ax = plt.subplot()
    m = max(List)  # 获取最大值
    length = len(List)

    if length <= 1:
        return step

    for i in range(1, length):
        flag = 1
        for j in range(length - i):

            if (List[j] > List[j + 1]):
                temp = List[j]
                List[j] = List[j + 1]
                List[j + 1] = temp
                flag = 0

                ax.cla()
                ax.bar(range(length), List, color="steelblue")
                ax.bar(j + 1, List[j + 1], color="red")
                plt.title("冒泡排序")
                end = timeit.default_timer()
                ax.text(0, m, "Time = {:0.3f} s".format(end - start))
                plt.pause(Pause_time)
            else:
                ax.cla()
                ax.bar(range(length), List, color="steelblue")
                ax.bar(j, List[j], color="red")
                plt.title("冒泡排序")
                end = timeit.default_timer()
                ax.text(0, m, "Time = {:0.3f} s".format(end - start))
                plt.pause(Pause_time)

        if flag: break
    return end - start


def QuickSort(List, start, end, ax,start_time):#快速排序
    length = len(List)
    m = max(List)
    # 分治 一分为二
    # start=end ,证明要处理的数据只有一个
    # start>end ,证明右边没有数据
    if start >= end:
        end_time = timeit.default_timer()
        return end_time
    # 定义两个游标，分别指向0和末尾位置
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    mid = List[left]
    while left < right:
        # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置

        while left < right and List[right] >= mid:
            ax.cla()
            ax.bar(range(length), List, color="steelblue")
            ax.bar(right, List[right], color="green")
            ax.bar(left, List[left], color="red")
            plt.title("快速排序")
            end_time = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end_time - start_time))
            plt.pause(Pause_time)

            right -= 1
        List[left] = List[right]

        ax.cla()
        ax.bar(range(length), List, color="steelblue")
        ax.bar(left, List[left], color="red")
        ax.bar(right, List[right], color="red")
        plt.title("快速排序")
        end_time = timeit.default_timer()
        ax.text(0, m, "Time = {:0.3f} s".format(end_time - start_time))
        plt.pause(Pause_time)

        # 让左边游标往右移动，目的是找到大于mid的值，放到right游标位置
        while left < right and List[left] < mid:
            ax.cla()
            ax.bar(range(length), List, color="steelblue")
            ax.bar(left, List[left], color="green")
            ax.bar(right, List[right], color="red")
            plt.title("快速排序")
            end_time = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end_time - start_time))
            plt.pause(Pause_time)

            left += 1
        List[right] = List[left]

        ax.cla()
        ax.bar(range(length), List, color="steelblue")
        ax.bar(left, List[left], color="red")
        ax.bar(right, List[right], color="red")
        plt.title("快速排序")
        end_time = timeit.default_timer()
        ax.text(0, m, "Time = {:0.3f} s".format(end_time - start_time))
        plt.pause(Pause_time)

    # while结束后，把mid放到中间位置，left=right
    List[left] = mid

    ax.cla()
    ax.bar(range(length), List, color="steelblue")
    ax.bar(left, mid, color="yellow")
    plt.title("快速排序")
    end_time = timeit.default_timer()
    ax.text(0, m, "Time = {:0.3f} s".format(end_time - start_time))
    plt.pause(Pause_time)

    # 递归处理左边的数据
    end_time = QuickSort(List, start, left - 1, ax, start_time)
    # 递归处理右边的数据
    end_time = QuickSort(List, left + 1, end, ax, start_time)
    return end_time


def SelectSort(List):#选择排序
    start = timeit.default_timer()
    ax = plt.subplot()
    length = len(List)
    m = max(List)


    if length <= 1:
        return 0

    for i in range(length):
        k = i

        ax.cla()
        ax.bar(range(length), List, color="steelblue")
        ax.bar(i, List[i], color="red")
        end = timeit.default_timer()
        ax.text(0, m, "Time = {:0.3f} s".format(end - start))
        plt.title("简单选择排序")
        plt.pause(Pause_time)

        for j in range(k + 1, length):
            if List[j] < List[k]:
                k = j

            ax.cla()
            ax.bar(range(length), List, color="steelblue")
            ax.bar(i, List[i], color="red")
            ax.bar(k, List[k], color="yellow")
            ax.bar(j, List[j], color="green")
            end = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end - start))
            plt.title("简单选择排序")
            plt.pause(Pause_time)

        if k != i:
            List[i], List[k] = List[k], List[i]

            ax.cla()
            ax.bar(range(length), List, color="steelblue")
            ax.bar(k, List[k], color="red")
            end = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end - start))
            plt.title("简单选择排序")
            plt.pause(Pause_time)

    return end - start


def HeadSort(List):  # 堆排序
    start = timeit.default_timer()
    Len = len(List)  # 数组长度
    ax = plt.subplot()
    step = 0
    m = max(List)

    def headpify(List, n, Len, start):  # 传入列表,最后一个非叶节点的索引,需要排列的长度
        l = 2 * n + 1  # 最后一个非叶节点的左孩子
        r = 2 * n + 2  # 最后一个非叶节点的右孩子
        maxIndex = n  # 最大值的索引

        if r < Len and List[r] > List[maxIndex]:
            maxIndex = r

        if l < Len and List[l] > List[maxIndex]:
            maxIndex = l

        if maxIndex != n:
            ax.cla()
            ax.bar(range(length), List, color="steelblue")
            ax.bar(n, List[n], color="green")
            ax.bar(0, List[0], color="red")
            plt.title("堆排序")
            end = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end - start))
            plt.pause(Pause_time)

            temp = List[maxIndex]
            List[maxIndex] = List[n]
            List[n] = temp

            ax.cla()
            ax.bar(range(length), List, color="steelblue")
            ax.bar(0, List[0], color="red")
            ax.bar(maxIndex, List[maxIndex], color="green")
            plt.title("堆排序")
            end = timeit.default_timer()
            ax.text(0, m, "Time = {:0.3f} s".format(end - start))
            plt.pause(Pause_time)

            headpify(List, maxIndex, Len, start)  # 如果有子节点，继续交换


    def buildMaxHead(List, Len, step):  # 初始化大顶堆
        n = Len // 2 - 1  # 从最后一个非叶节点向前遍历,使之成为大顶堆
        for i in range(n, -1, -1):
            headpify(List, i, Len, start)

    length = Len = len(List)  # 数组长度

    if length <= 1:
        return step

    buildMaxHead(List, Len, start)

    for i in range(Len - 1, -1, -1):
        temp = List[i]
        List[i] = List[0]
        List[0] = temp

        ax.cla()
        ax.bar(range(length), List, color="steelblue")
        ax.bar(i, List[i], color="red")
        plt.title("堆排序")
        end = timeit.default_timer()
        ax.text(0, m, "Time = {:0.3f} s".format(end - start))
        plt.pause(Pause_time)

        ax.cla()
        ax.bar(range(length), List, color="steelblue")
        ax.bar(0, List[0], color="red")
        plt.title("堆排序")
        end = timeit.default_timer()
        ax.text(0, m, "Time = {:0.3f} s".format(end - start))
        plt.pause(Pause_time)
        Len -= 1
        headpify(List, 0, Len, start)


    return end - start


'''归并排序'''
def merge(List, left, mid, right, ax, start_time):#归并函数
    # List：待排序列表
    # left：排序范围最小索引
    # mid：待排序范围分割点，即中点
    # right：排序范围最大索引
    # ax：画板
    # step：运行步数
    n1 = mid - left + 1  # 计算出上半段长度
    n2 = right - mid  # 计算出下半段长度
    '''创建俩个临时列表'''
    L = []
    R = []
    for i in range(n1):
        L.append(List[left + i])
    for j in range(n2):
        R.append(List[mid + 1 + j])
    i = 0
    j = 0
    '''谁小谁排前面'''
    while i < len(L) and j < len(R):
        if L[i] < R[j]:
            List[left] = L[i]

            ax.cla()
            ax.bar(range(len(List)), List, color="steelblue")
            ax.bar(left, List[left], color="red")
            end_time = timeit.default_timer()
            ax.text(0, max(List), "Time = {:0.3f} s".format(end_time - start_time))
            plt.title("归并排序")
            plt.pause(Pause_time)

            left += 1
            i += 1
        else:
            List[left] = R[j]

            ax.cla()
            ax.bar(range(len(List)), List, color="steelblue")
            ax.bar(left, List[left], color="red")
            end_time = timeit.default_timer()
            ax.text(0, max(List), "Time = {:0.3f} s".format(end_time - start_time))
            plt.title("归并排序")
            plt.pause(Pause_time)

            left += 1
            j += 1
    '''将待排序列表的对应位置替换成将临时列表的剩余元素'''
    while i < len(L):
        List[left] = L[i]

        ax.cla()
        ax.bar(range(len(List)), List, color="steelblue")
        ax.bar(left, List[left], color="red")
        end_time = timeit.default_timer()
        ax.text(0, max(List), "Time = {:0.3f} s".format(end_time - start_time))
        plt.title("归并排序")
        plt.pause(Pause_time)

        i += 1
        left += 1
    while j < len(R):
        List[left] = R[j]

        ax.cla()
        ax.bar(range(len(List)), List, color="steelblue")
        ax.bar(left, List[left], color="red")
        end_time = timeit.default_timer()
        ax.text(0, max(List), "Time = {:0.3f} s".format(end_time - start_time))
        plt.title("归并排序")
        plt.pause(Pause_time)

        left += 1
        j += 1


def MergeSort(List, start, end, ax,start_time):#归并排序

    # List：待排序列表
    # left：排序范围最小索引
    # mid：待排序范围分割点，即中点
    # right：排序范围最大索引
    # ax：画板
    # step：运行步数

    if start < end:
        mid = (start + end) // 2  # 计算分割点，即中点
        MergeSort(List, start, mid, ax ,start_time)
        MergeSort(List, mid + 1, end, ax,start_time)
        merge(List, start, mid, end, ax,start_time)
    return


def CountSort(List):#计数排序
    start = timeit.default_timer()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    length = len(List)

    if length <= 1:
        return 0

    Max = 0
    for i in range(length):
        ax1.cla()
        ax1.bar(range(length), List, color="steelblue")
        ax1.bar(i, List[i], color="green")
        end = timeit.default_timer()
        ax1.text(0, max(List), "Time = {:0.3f} s".format(end - start))
        ax1.text(length // 2, max(List), "计数排序")

        plt.pause(Pause_time)
        if List[i] > Max:
            Max = List[i]

    countlist = [0 for i in range(Max + 1)]

    for i in range(length):
        countlist[List[i]] = countlist[List[i]] + 1
        ax1.cla()
        ax1.bar(range(length), List, color="steelblue")
        ax1.bar(i, List[i], color="green")
        end = timeit.default_timer()
        ax1.text(0, max(List), "Time = {:0.3f} s".format(end - start))
        ax1.text(length // 2, max(List), "计数排序")

        ax2.cla()
        ax2.bar(range(Max + 1), countlist, color="steelblue")
        ax2.bar(List[i], countlist[List[i]], color="red")
        ax2.text(len(countlist) // 2, max(countlist), "计数列表")

        plt.pause(Pause_time)
    j = 0
    for i in range(Max + 1):
        while countlist[i] > 0:
            List[j] = i
            j += 1
            countlist[i] = countlist[i] - 1

            ax1.cla()
            ax1.bar(range(length), List, color="steelblue")
            ax1.bar(j - 1, List[j - 1], color="red")
            end = timeit.default_timer()
            ax1.text(0, max(List), "Time = {:0.3f} s".format(end - start))
            ax1.text(length // 2, max(List), "计数排序")

            ax2.cla()
            ax2.bar(range(Max + 1), countlist, color="steelblue")
            ax2.bar(i, countlist[i], color="green")
            ax2.text(len(countlist) // 2, max(countlist), "计数列表")

            plt.pause(Pause_time)
    return end - start


def RadixSort(List):#基数排序
    start = timeit.default_timer()
    ax = plt.subplot()

    length = len(List)

    if length <= 1:
        return 0

    Max = 0
    for i in range(length):  # 寻找出数组最大的数
        if List[i] > Max:
            Max = List[i]

            ax.cla()
            ax.bar(range(length), List, color="steelblue")
            ax.bar(i,List[i],color="green")
            end = timeit.default_timer()
            ax.text(0, max(List), "Time = {:0.3f} s".format(end - start))
            ax.text(length / 2, Max, "寻找最大值中.....")
            plt.title("基数排序")
            plt.pause(Pause_time)

    k = 1
    count = 0

    while Max // k > 0:  # 求解最大位数
        k = k * 10
        count += 1

    bucket = [[] for i in range(10)]
    for j in range(length):  # 初始化桶
        bucket[List[j] % 10].append(List[j])

        ax.cla()
        ax.bar(range(length), List, color="steelblue")
        ax.bar(j, List[j], color="green")
        end = timeit.default_timer()
        ax.text(0, max(List), "Time = {:0.3f} s".format(end - start))
        plt.title("基数排序")
        plt.pause(Pause_time)

    for i in range(count - 1):  # 初始化时已经排好个位所以count - 1
        temp = [[] for n in range(10)]

        for j in range(len(bucket)):
            for k in range(len(bucket[j])):
                index = bucket[j][k] % pow(10, i + 2) // pow(10, i + 1)  # 求对应位数的大小
                temp[index].append(bucket[j][k])

                ax.cla()
                ax.bar(range(length), List, color="steelblue")
                end = timeit.default_timer()
                ax.text(0, max(List), "Time = {:0.3f} s".format(end - start))
                ax.text(length/2,Max,"排序中.....")
                plt.title("基数排序")
                plt.pause(Pause_time)

        bucket = temp
        pos = 0

        for j in range(len(bucket)):
            for k in range(len(bucket[j])):
                List[pos] = bucket[j][k]

                ax.cla()
                ax.bar(range(length), List, color="steelblue")
                ax.bar(pos, List[pos], color="red")
                end = timeit.default_timer()
                ax.text(0, max(List), "Time = {:0.3f} s".format(end - start))
                plt.title("基数排序")
                plt.pause(Pause_time)

                pos += 1

    if count == 1:
        pos = 0
        for j in range(len(bucket)):
            for k in range(len(bucket[j])):
                List[pos] = bucket[j][k]

                ax.cla()
                ax.bar(range(length), List, color="steelblue")
                ax.bar(pos, List[pos], color="red")
                end = timeit.default_timer()
                ax.text(0, max(List), "Time = {:0.3f} s".format(end - start))
                plt.title("基数排序")
                plt.pause(Pause_time)

                pos += 1

    return end - start


def dawnend(List):#最终结果绘制
    length = len(List)
    ax = plt.subplot()

    ax.bar(range(length), List, color="steelblue")
    plt.pause(0.5)
    for i in range(length):
        ax.bar(i, List[i], color="orange")
        plt.pause(Pause_time)
    plt.pause(1)
