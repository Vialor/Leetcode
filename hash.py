from collections import Counter, defaultdict
from typing import List, Optional


class Node:
    def __init__(self, x: int, next: "Node" = None, random: "Node" = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    # 128*
    def longestConsecutive(self, nums: List[int]) -> int:
        endWithMap = dict()
        startWithMap = dict()
        visited = set()
        for num in nums:
            if num in visited:
                continue
            visited.add(num)
            if num - 1 in visited and num + 1 in visited:
                endWithMap[startWithMap[num - 1]], startWithMap[endWithMap[num + 1]] = (
                    endWithMap[num + 1],
                    startWithMap[num - 1],
                )
            elif num - 1 in visited:
                endWithMap[num], startWithMap[num] = num, startWithMap[num - 1]
                endWithMap[startWithMap[num - 1]] = num
            elif num + 1 in visited:
                endWithMap[num], startWithMap[num] = endWithMap[num + 1], num
                startWithMap[endWithMap[num + 1]] = num
            else:
                startWithMap[num], endWithMap[num] = num, num

        longest = 0
        for num in startWithMap:
            longest = max(endWithMap[num] - startWithMap[num] + 1, longest)
        return longest

    # second method in Java (probably better)
    # public int longestConsecutive(int[] nums) {
    #     Map<Integer, Integer> countForNum = new HashMap<>();
    #     for (int num : nums) {
    #         countForNum.put(num, 1);
    #     }
    #     for (int num : nums) {
    #         forward(countForNum, num);
    #     }
    #     return maxCount(countForNum);
    # }

    # private int forward(Map<Integer, Integer> countForNum, int num) {
    #     if (!countForNum.containsKey(num)) {
    #         return 0;
    #     }
    #     int cnt = countForNum.get(num);
    #     if (cnt > 1) {
    #         return cnt;
    #     }
    #     cnt = forward(countForNum, num + 1) + 1;
    #     countForNum.put(num, cnt);
    #     return cnt;
    # }

    # private int maxCount(Map<Integer, Integer> countForNum) {
    #     int max = 0;
    #     for (int num : countForNum.keySet()) {
    #         max = Math.max(max, countForNum.get(num));
    #     }
    #     return max;
    # }

    # 409
    def longestPalindrome(self, s: str) -> int:
        cntDict = dict()
        for c in s:
            cntDict[c] = cntDict[c] + 1 if c in cntDict else 1
        hasOdd, ans = False, 0
        for k in cntDict:
            if cntDict[k] % 2 == 0:
                ans += cntDict[k]
            else:
                ans += cntDict[k] - 1
                hasOdd = True
        return ans + 1 if hasOdd else ans

    # 697
    def findShortestSubArray(self, nums: List[int]) -> int:
        frequencyMap = {}
        firstIndex, lastIndex = {}, {}
        for i in range(len(nums)):
            num = nums[i]
            if num not in firstIndex:
                firstIndex[num] = i
            lastIndex[num] = i
            frequencyMap[num] = frequencyMap[num] + 1 if num in frequencyMap else 1

        maxFrequencyNum, shortestLength = None, 0
        for num in nums:
            if (
                maxFrequencyNum is None
                or frequencyMap[maxFrequencyNum] < frequencyMap[num]
                or frequencyMap[maxFrequencyNum] == frequencyMap[num]
                and shortestLength > lastIndex[num] - firstIndex[num] + 1
            ):
                maxFrequencyNum = num
                shortestLength = lastIndex[num] - firstIndex[num] + 1
        return shortestLength

    # 49
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groupDict = {}
        for s in strs:
            category = "".join(sorted(s))
            if category not in groupDict:
                groupDict[category] = []
            groupDict[category].append(s)
        ans = []
        for v in groupDict.values():
            ans.append(v)
        return ans

    # 1*
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i in range(len(nums)):
            if target - nums[i] in d:
                return [i, d[target - nums[i]]]
            d[nums[i]] = i
        return [-1, -1]

    # 560*
    def subarraySum(self, nums: List[int], k: int) -> int:
        preSumsCounter = defaultdict(int)
        ans = preSum = 0
        for num in nums:
            preSum += num
            if preSum == k:
                ans += 1
            ans += preSumsCounter[preSum - k]
            preSumsCounter[preSum] += 1
        return ans

    # 930
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        preSumsCount = defaultdict(int)
        curSum = 0
        count = 0
        for num in nums:
            preSumsCount[curSum] += 1
            curSum += num
            count += preSumsCount[curSum - goal]
        return count

    # 1248
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        nums = [num % 2 for num in nums]
        curSum, preSumsCount = 0, defaultdict(int)
        ans = 0
        for num in nums:
            preSumsCount[curSum] += 1
            curSum += num
            ans += preSumsCount[curSum - k]
        return ans

    # 846
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        hand.sort()
        handCounter = Counter(hand)
        for card in hand:
            if handCounter[card] == 0:
                continue
            for i in range(groupSize):
                if handCounter[card + i] < 1:
                    return False
                handCounter[card + i] -= 1
        return True

    # 138
    nodeMap = {None: None}

    def copyRandomList(self, head: "Optional[Node]") -> "Optional[Node]":
        if head in self.nodeMap:
            return self.nodeMap[head]
        newNode = Node(x=head.val)
        self.nodeMap[head] = newNode
        newNode.next = self.copyRandomList(head.next)
        newNode.random = self.copyRandomList(head.random)
        return self.nodeMap[head]
