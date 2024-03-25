import timeit

def InsertSort(List):#插入排序
    step = [0,0]
    length = len(List)  # 获取列表长度

    if length <= 1:
        return List
    for i in range(1, length):
        j = i
        temp = List[i]  # 每次待插入的元素
        step[1] += 1

        while j > 0 and temp < List[j - 1]:
            step[0] += 1
            List[j] = List[j - 1]  # 整体后移,为temp腾空位置
            step[1] += 1
            j -= 1
        List[j] = temp # 将temp插入到空位
        step[1] += 1

    return step


def BInsertSort(List):#折半(二分查找)插入排序
    length = len(List)
    step = [0,0]

    if length <= 1:
        return List
    for i in range(1, length):
        j = i
        temp = List[i]  # 每次待插入的元素
        step[1] += 1
        low = 0  # 二分查找最小索引
        high = j - 1  # 二分查找最大索引

        while high >= low:  # 二分查找
            mid = (high + low) // 2  # 取中点
            step[0] += 1
            if temp < List[mid]:
                high = mid - 1  # 范围在前半段，接下来在前半段搜索

            else:
                low = mid + 1  # 范围在后半段，接下来在后半段搜索


        # List[high + 2: j + 1] = List[high + 1: j]  # 整体后移，为temp腾位
        while j > high + 1 :
            step[1] += 1
            List[j] = List[j-1]
            j -= 1
        List[high + 1] = temp  # 将temp插入到空位
        step[1] += 1

    return step


def ShallSort(List):  # 希尔排序
    step = [0,0]
    def ShallInsert(d):  # 以d为递增量进行排序
        length = len(List)
        for i in range(d, length):
            j = i
            temp = List[i]  # 待插入元素
            step[1] += 1

            while j - d + 1 > 0 and temp < List[j - d]:# 整体后移，为temp腾空位置
                step[0] += 1
                List[j] = List[j - d]
                step[1] += 1
                j -= d

            List[j] = temp  # 将temp插入空位
            step[1] += 1

    length = len(List)

    if length <= 1:
        return step

    d = length // 2  # 递增量
    while d >= 1:
        ShallInsert(d)
        d = d // 2
    return step


def BubleSort(List):#冒泡排序
    length = len(List)
    step = [0,0]

    if length <= 1:
        return step

    for i in range(1, length):
        flag = 1
        for j in range(length - i):
            step[0] += 1

            if (List[j] > List[j + 1]):
                step[1] += 3
                temp = List[j]
                List[j] = List[j + 1]
                List[j + 1] = temp
                flag = 0

        if flag: break

    return step


def QuickSort(List, start, end,step):#快速排序
    # 分治 一分为二
    # start=end ,证明要处理的数据只有一个
    # start>end ,证明右边没有数据
    if start >= end:
        return step
    # 定义两个游标，分别指向0和末尾位置
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    mid = List[left]
    step[1] += 1
    while left < right:
        # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置

        while left < right and List[right] >= mid:
            step[0] += 1
            right -= 1
        List[left] = List[right]
        step[1] += 1

        # 让左边游标往右移动，目的是找到大于mid的值，放到right游标位置
        while left < right and List[left] < mid:
            step[0] += 1
            left += 1
        List[right] = List[left]
        step[1] += 1

    # while结束后，把mid放到中间位置，left=right
    List[left] = mid
    step[1] += 1

    # 递归处理左边的数据
    QuickSort(List, start, left - 1,step)
    # 递归处理右边的数据
    QuickSort(List, left + 1, end,step)
    return step


def SelectSort(List):#选择排序
    step=[0,0]
    length = len(List)

    if length <= 1:
        return step

    for i in range(length):
        k = i

        for j in range(k + 1, length):
            step[0]+=1
            if List[j] < List[k]:
                k = j

        if k != i:
            List[i], List[k] = List[k], List[i]
            step[1] += 3

    return step

def HeadSort(List):  # 堆排序
    step = [0,0]

    def headpify(List, n, Len):  # 传入列表,最后一个非叶节点的索引,需要排列的长度
        l = 2 * n + 1  # 最后一个非叶节点的左孩子
        r = 2 * n + 2  # 最后一个非叶节点的右孩子
        maxIndex = n  # 最大值的索引

        if r < Len and List[r] > List[maxIndex]:
            maxIndex = r

        if l < Len and List[l] > List[maxIndex]:
            maxIndex = l
        step[0] += 2
        if maxIndex != n:
            step[1] += 1
            temp = List[maxIndex]
            List[maxIndex] = List[n]
            List[n] = temp
            headpify(List, maxIndex, Len)  # 如果有子节点，继续交换

    def buildMaxHead(List, Len):  # 初始化大顶堆
        n = Len // 2 - 1  # 从最后一个非叶节点向前遍历,使之成为大顶堆
        for i in range(n, -1, -1):
            headpify(List, i, Len)

    Len = len(List)  # 数组长度

    if Len <= 1:
        return step

    buildMaxHead(List, Len)

    for i in range(Len - 1, -1, -1):
        temp = List[i]
        List[i] = List[0]
        List[0] = temp
        step[1] += 3
        Len -= 1
        headpify(List, 0, Len)
    return step


def merge(List, left, mid, right,step):#归并函数
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
        step[0] += 1
        step[1] += 1
        if L[i] < R[j]:
            List[left] = L[i]

            left += 1
            i += 1
        else:
            List[left] = R[j]

            left += 1
            j += 1
    '''将待排序列表的对应位置替换成将临时列表的剩余元素'''
    while i < len(L):
        step[1] += 1
        List[left] = L[i]

        i += 1
        left += 1
    while j < len(R):
        step[1] += 1
        List[left] = R[j]

        left += 1
        j += 1

def MergeSort(List, start, end):#归并排序
    step = [0,0]

    if start < end:
        mid = (start + end) // 2  # 计算分割点，即中点
        MergeSort(List, start, mid)
        MergeSort(List, mid + 1, end)
        merge(List, start, mid, end, step)
    return step

def CountSort(List): #计数排序
    step = [0,0]
    length = len(List)

    if length <= 1:
        return List

    Max = 0
    for i in range(length):
        step[0] += 1
        if List[i] > Max:
            Max = List[i]

    countlist = [0 for i in range(Max + 1)]

    for i in range(length):
        step[1] += 1
        countlist[List[i]] = countlist[List[i]] + 1

    j = 0
    for i in range(Max + 1):
        while countlist[i] > 0:
            step[1] += 1
            List[j] = i
            j += 1
            countlist[i] = countlist[i] - 1

    return step

def RadixSort(List):
    step = [0,0]
    length = len(List)

    if length <= 1:
        return List

    Max = 0
    for i in List: # 寻找出数组最大的数
        step[0] += 1
        if i > Max:
            Max = i
    k = 1
    count = 0

    while Max // k > 0:  # 求解最大位数
        step[0] += 1
        k = k * 10
        count += 1

    bucket = [[] for i in range(10)]
    for j in List:  # 初始化桶
        step[1] += 1
        bucket[j % 10].append(j)

    for i in range(count - 1):  # 初始化时已经排好个位所以count - 1
        temp = [[] for n in range(10)]

        for j in range(len(bucket)):
            for k in range(len(bucket[j])):
                step[1] += 1
                index = bucket[j][k] % pow(10, i + 2) // pow(10, i + 1)  # 求对应位数的大小
                temp[index].append(bucket[j][k])
        bucket = temp
        pos = 0

    for j in range(len(bucket)):
        for k in range(len(bucket[j])):
            step[1] += 1
            List[pos] = bucket[j][k]
            pos += 1
    return step





# print(QuickSort(List,0,len(List)-1))
# print(MergeSort(List,0,len(List)-1))
# List = LoadFile()
# start = timeit.default_timer()
# step = CountSort(List)
# end = timeit.default_timer()
# print("比较步数：{}\n移动步数：{}".format(step[0],step[1]))
# print('Running time: {} Seconds'.format(end - start))
