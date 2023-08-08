import collections
from functools import lru_cache
from heapq import *
import heapq
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # 232
    class MyQueue:
        def __init__(self):
            self.inStack = []
            self.outStack = []

        def push(self, x: int) -> None:
            self.inStack.append(x)

        def inStackToOutStack(self) -> None:
            while self.inStack:
                self.outStack.append(self.inStack.pop())

        def pop(self) -> int:
            if not self.outStack:
                self.inStackToOutStack()
            return self.outStack.pop()

        def peek(self) -> int:
            if not self.outStack:
                self.inStackToOutStack()
            return self.outStack[-1]

        def empty(self) -> bool:
            return not self.inStack and not self.outStack

    # 225
    class MyStack:
        def __init__(self):
            self.queue = collections.deque()

        def push(self, x: int) -> None:
            count = len(self.queue)
            self.queue.append(x)
            while count:
                self.queue.append(self.queue.popleft)
                count -= 1

        def pop(self) -> int:
            return self.queue.popleft()

        def top(self) -> int:
            return self.queue[0]

        def empty(self) -> bool:
            return not self.queue

    # 150
    def evalRPN(self, tokens: List[str]) -> int:
        item = tokens.pop()
        if item not in "+-*/":
            return int(item)
        secondOperand = self.evalRPN(tokens)
        firstOperand = self.evalRPN(tokens)
        match item:
            case "+":
                return firstOperand + secondOperand
            case "-":
                return firstOperand - secondOperand
            case "*":
                return firstOperand * secondOperand
            case "/":
                return int(firstOperand / secondOperand)

    # 155
    class MinStack:
        def __init__(self):
            self.dataStack = []
            self.minStack = []

        def push(self, val: int) -> None:
            self.dataStack.append(val)
            minVal = min(self.minStack[-1], val) if self.minStack else val
            self.minStack.append(minVal)

        def pop(self) -> None:
            self.dataStack.pop()
            self.minStack.pop()

        def top(self) -> int:
            return self.dataStack[-1]

        def getMin(self) -> int:
            return self.minStack[-1]

    # 224*
    def calculate(self, s: str) -> int:
        curStr = ""
        slist = []
        for i in range(len(s)):
            if s[i] == " ":
                continue
            curStr += s[i]
            if i + 1 >= len(s) or not s[i].isdigit() or not s[i + 1].isdigit():
                slist.append(curStr)
                curStr = ""

        bracketIndexMapping = {}
        bracketStack = []
        for i, c in enumerate(slist):
            if c == "(":
                bracketStack.append(i)
            elif c == ")":
                bracketIndexMapping[i] = bracketStack.pop()

        def calculateHelper(start, end) -> int:
            if end - start <= 0:
                return 0
            elif end - start <= 2:
                return int("".join(slist[start:end]))
            match slist[end - 1]:
                case ")":
                    opInd = bracketIndexMapping[end - 1] - 1
                    num1 = calculateHelper(start, opInd)
                    num2 = calculateHelper(opInd + 2, end - 1)
                    return num1 - num2 if slist[opInd] == "-" else num1 + num2
                case _:  # ints
                    opInd = end - 2
                    num1 = calculateHelper(start, opInd)
                    num2 = calculateHelper(opInd + 1, end)
                    return num1 - num2 if slist[opInd] == "-" else num1 + num2

        return calculateHelper(0, len(slist))

    # method 2 from other devlopers, better than mine
    # def calculate(self, s):
    #     def evaluate(i):
    #         res, digit, sign = 0, 0, 1
    #         while i < len(s):
    #             if s[i].isdigit():
    #                 digit = digit * 10 + int(s[i])
    #             elif s[i] in '+-':
    #                 res += digit * sign
    #                 digit = 0
    #                 sign = 1 if s[i] == '+' else -1
    #             elif s[i] == '(':
    #                 subres, i = evaluate(i+1)
    #                 res += sign * subres
    #             elif s[i] == ')':
    #                 res += digit * sign
    #                 return res, i
    #             i += 1

    #         return res + digit * sign

    #     return evaluate(0)

    # monotonic stack
    # 84*
    def largestRectangleArea(self, heights: List[int]) -> int:
        heights.append(0)
        minStack = []
        ans = 0
        for i, height in enumerate(heights):
            while minStack and heights[minStack[-1]] > height:
                popIndex = minStack.pop()
                left = minStack[-1] if minStack else -1
                ans = max(ans, heights[popIndex] * (i - left - 1))
            minStack.append(i)
        return ans

    # 456*
    def find132pattern(self, nums: List[int]) -> bool:
        popNum = float("-inf")
        decreaseStack = []
        for num in reversed(nums):
            if num < popNum:
                return True
            while decreaseStack and decreaseStack[-1] < num:
                popNum = decreaseStack.pop()
            decreaseStack.append(num)
        return False

    # 85*
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        ans = 0
        heights = [0] * n
        for i in range(m):
            heights = [heights[j] + 1 if matrix[i][j] == "1" else 0 for j in range(n)]
            heights.append(0)
            minStack = []
            for j, height in enumerate(heights):
                while minStack and heights[minStack[-1]] > height:
                    popIndex = minStack.pop()
                    left = minStack[-1] if minStack else -1
                    ans = max(ans, heights[popIndex] * (j - left - 1))
                minStack.append(j)
            heights.pop()
        return ans

    # 496
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        mapping = {}
        for i in range(len(nums2) - 1, -1, -1):
            while stack and stack[-1] < nums2[i]:
                stack.pop()
            mapping[nums2[i]] = stack[-1] if stack else -1
            stack.append(nums2[i])
        return [mapping[num] for num in nums1]

    # 739*
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        result = [0] * len(temperatures)
        for i in reversed(range(len(temperatures))):
            while stack and temperatures[stack[-1]] <= temperatures[i]:
                stack.pop()
            if stack:
                result[i] = stack[-1] - i
            stack.append(i)
        return result

    # 503*
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        result = [-1] * len(nums)
        stack = []
        for i in reversed(range(len(nums) * 2 - 1)):
            while stack and stack[-1] <= nums[i % len(nums)]:
                stack.pop()
            if stack:
                result[i % len(nums)] = stack[-1]
            stack.append(nums[i % len(nums)])
        return result

    # 503 second method
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stack = []
        last = None
        for n in nums:
            if last is None or n > last:
                stack.append(n)
                last = n
        stack.reverse()

        result = [-1] * len(nums)
        for i in reversed(range(len(nums))):
            while stack and stack[-1] <= nums[i]:
                stack.pop()
            if stack:
                result[i] = stack[-1]
            stack.append(nums[i])
        return result

    # 1340* O(n)
    # Just dp and recurssion is easier but slower. This solution gets rid of the complexity of d.
    def maxJumps(self, arr: List[int], d: int) -> int:
        arr.append(float("inf"))
        arrL = len(arr)
        stack = []
        dp = [1 for _ in range(arrL)]
        for index, arr_i in enumerate(arr):
            while stack and arr[stack[-1]] < arr_i:
                current = [stack.pop()]
                while stack and arr[stack[-1]] == arr[current[0]]:
                    current.append(stack.pop())
                for sub_index in current:
                    if index - sub_index <= d:
                        dp[index] = max(dp[index], dp[sub_index] + 1)
                    if stack and sub_index - stack[-1] <= d:
                        dp[stack[-1]] = max(dp[stack[-1]], dp[sub_index] + 1)
            stack.append(index)
        return max(dp[: arrL - 1])

    # 1340 second solution
    def maxJumps(self, A, d):
        N = len(A)
        graph = collections.defaultdict(list)

        def jump(iterator):
            stack = []
            for i in iterator:
                while stack and A[stack[-1]] < A[i]:
                    j = stack.pop()
                    if abs(i - j) <= d:
                        graph[j].append(i)
                stack.append(i)

        jump(range(N))
        jump(reversed(range(N)))

        @lru_cache(maxsize=None)
        def height(i):
            return 1 + max(map(height, graph[i]), default=0)

        return max(map(height, range(N)))

    # monotonic deque
    # 239*
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = collections.deque()
        ans = []
        for i in range(len(nums)):
            if i - k >= 0 and queue[0] == i - k:
                queue.popleft()
            while queue and nums[queue[-1]] < nums[i]:
                queue.pop()
            queue.append(i)
            ans.append(nums[queue[0]])
        return ans[k - 1 :]

    # 862*
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        preSums = [0]
        for num in nums:
            preSums.append(num + preSums[-1])
        queue = collections.deque([0])
        shortest = float("inf")
        for i in range(1, len(preSums)):
            while queue and preSums[queue[-1]] >= preSums[i]:
                queue.pop()
            queue.append(i)
            while queue and preSums[queue[-1]] - preSums[queue[0]] >= k:
                shortest = min(shortest, queue[-1] - queue[0])
                queue.popleft()
        return -1 if shortest == float("inf") else shortest

    # 581
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        start, end = -1, -2
        curMax = float("-inf")
        for i in range(len(nums)):
            if nums[i] < curMax:
                end = i
            curMax = max(nums[i], curMax)
        curMin = float("inf")
        for i in reversed(range(len(nums))):
            if nums[i] > curMin:
                start = i
            curMin = min(nums[i], curMin)
        return end - start + 1

    # https://leetcode.com/problems/trapping-rain-water/ 42 with stack
    # 42* TODO
    def trap(self, height: List[int]) -> int:
        stack = []
        res = 0
        for i in range(len(height)):
            stack.append(height[i])

    # https://leetcode.com/problems/subarrays-with-k-different-integers/
    # https://leetcode.com/problems/online-stock-span/
    # https://leetcode.com/problems/sum-of-subarray-minimums/

    # heap
    # 1046*
    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [-stone for stone in stones]
        heapify(stones)
        while len(stones) > 1:
            heappush(stones, -abs(heappop(stones) - heappop(stones)))
        return -stones[0]

    # 23*
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        pq = [(lists[i].val, i, lists[i]) for i in range(len(lists)) if lists[i]]
        heapify(pq)
        ansHead = ListNode()
        ansCurNode = ansHead
        while pq:
            popItem = heappop(pq)
            curNode = popItem[2]
            ansCurNode.next = curNode
            ansCurNode = curNode
            if curNode.next:
                heappush(pq, (curNode.next.val, popItem[1], curNode.next))
        return ansHead.next

    # 621*
    def leastInterval(self, tasks: List[str], n: int) -> int:
        taskCounter = collections.Counter(tasks).values()
        taskCounter = [-frequency for frequency in taskCounter]
        heapify(taskCounter)
        timeQueue = collections.deque()
        time = 0
        while taskCounter or timeQueue:
            time += 1
            if taskCounter:
                freq = heappop(taskCounter) + 1
                if freq < 0:
                    timeQueue.append((freq, time + n))
            if timeQueue and time == timeQueue[0][1]:
                heappush(taskCounter, timeQueue.popleft()[0])
        return time

    # 1851
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        intervals.sort(reverse=True)
        minHeap = []
        ans = {}
        for q in sorted(queries):
            while len(intervals) > 0 and intervals[-1][0] <= q:
                start, end = intervals.pop()
                heapq.heappush(minHeap, (end - start + 1, end))
            while len(minHeap) > 0 and minHeap[0][1] < q:
                heapq.heappop(minHeap)
            ans[q] = minHeap[0][0] if len(minHeap) > 0 else -1
        return [ans[q] for q in queries]

    # 295*


class MedianFinder:
    def __init__(self):
        self.lowerMaxHeap = [float("inf")]
        self.upperMinHeap = [float("inf")]

    def addNum(self, num: int) -> None:
        if self.upperMinHeap[0] < num:
            heappush(self.upperMinHeap, num)
        else:
            heappush(self.lowerMaxHeap, -num)

        diff = len(self.lowerMaxHeap) - len(self.upperMinHeap)
        if diff > 1:
            heappush(self.upperMinHeap, -heappop(self.lowerMaxHeap))
        elif diff < 0:
            heappush(self.lowerMaxHeap, -heappop(self.upperMinHeap))

    def findMedian(self) -> float:
        if len(self.lowerMaxHeap) == len(self.upperMinHeap):
            return (self.upperMinHeap[0] - self.lowerMaxHeap[0]) / 2
        return -self.lowerMaxHeap[0]
