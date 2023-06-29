from collections import Counter, defaultdict
import math
from typing import List, Optional


class Solution:
    # High & Low
    # 167*
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        a, b = 1, len(numbers)
        while numbers[a - 1] + numbers[b - 1] != target:
            if numbers[a - 1] + numbers[b - 1] < target:
                a += 1
            else:
                b -= 1
        return [a, b]

    # 633
    def judgeSquareSum(self, c: int) -> bool:
        a, b = 0, math.floor(math.sqrt(c))
        while a <= b:
            sum = a**2 + b**2
            if sum < c:
                a += 1
            elif sum > c:
                b -= 1
            else:
                return True
        return False

    # 345
    def reverseVowels(self, s: str) -> str:
        vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}
        result = list(s)
        a, b = 0, len(s) - 1
        while a <= b:
            if result[a] not in vowels:
                a += 1
            elif result[b] not in vowels:
                b -= 1
            else:
                result[a], result[b] = result[b], result[a]
                a += 1
                b -= 1
        return "".join(result)

    # 680
    def validPalindrome(self, s: str) -> bool:
        a, b = 0, len(s) - 1
        while a < b:
            if s[a] != s[b]:
                s1 = s[:a] + s[a + 1 :]
                s2 = s[:b] + s[b + 1 :]
                return s1 == s1[::-1] or s2 == s2[::-1]
            a += 1
            b -= 1
        return True

    # 88
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        index1, index2, indexMerge = m - 1, n - 1, m + n - 1
        while indexMerge >= 0:
            if index1 < 0 or (index2 >= 0 and nums1[index1] < nums2[index2]):
                nums1[indexMerge] = nums2[index2]
                index2 -= 1
            else:
                nums1[indexMerge] = nums1[index1]
                index1 -= 1
            indexMerge -= 1

    # 524
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        def stringMatched(s1: str, s2: str) -> bool:
            a, b = 0, 0
            while a < len(s1) and b < len(s2):
                if s1[a] == s2[b]:
                    b += 1
                a += 1
            return b == len(s2)

        longestWord, longestWordLength = "", 0
        for word in dictionary:
            if stringMatched(s, word) and (
                len(word) > longestWordLength or (len(word) == longestWordLength and word < longestWord)
            ):
                longestWord, longestWordLength = word, len(word)
        return longestWord

    # 11*
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        mostVolumn = 0
        while left < right:
            leftHeight, rightHeight = height[left], height[right]
            if leftHeight < rightHeight:
                mostVolumn = max(mostVolumn, leftHeight * (right - left))
                left += 1
            else:
                mostVolumn = max(mostVolumn, rightHeight * (right - left))
                right -= 1
        return mostVolumn

    # 15*
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []

        nums.sort()
        result = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            a = i + 1
            b = len(nums) - 1
            while a < b:
                if nums[i] + nums[a] + nums[b] < 0:
                    a += 1
                elif nums[i] + nums[a] + nums[b] > 0:
                    b -= 1
                else:
                    result.append([nums[i], nums[a], nums[b]])
                    while a < b and nums[a] == nums[a + 1]:
                        a += 1
                    while a < b and nums[b] == nums[b - 1]:
                        b -= 1
                    a += 1
                    b -= 1
        return result

    # 26
    def removeDuplicates(self, nums: List[int]) -> int:
        a, b = 0, 0
        while b < len(nums):
            while b + 1 < len(nums) and nums[b] == nums[b + 1]:
                b += 1
            nums[a] = nums[b]
            a += 1
            b += 1
        nums = nums[:a]
        return a

    # 18
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        length = len(nums)
        for a in range(length):
            if a > 0 and nums[a] == nums[a - 1]:
                continue
            for b in range(a + 1, length):
                if b > a + 1 and nums[b] == nums[b - 1]:
                    continue
                i, j = b + 1, length - 1
                while i < j:
                    if nums[a] + nums[b] + nums[i] + nums[j] < target:
                        i += 1
                    elif nums[a] + nums[b] + nums[i] + nums[j] > target:
                        j -= 1
                    else:
                        result.append([nums[a], nums[b], nums[i], nums[j]])
                        while i < j and nums[i] == nums[i + 1]:
                            i += 1
                        i += 1
                        while i < j and nums[j] == nums[j - 1]:
                            j -= 1
                        j -= 1
        return result

    # 42
    def trap(self, height: List[int]) -> int:
        lmax, rmax = 0, 0
        l, r = 0, len(height) - 1
        res = 0
        while l < r:
            lmax = max(lmax, height[l])
            rmax = max(rmax, height[r])
            if lmax < rmax:
                res += lmax - height[l]
                l += 1
            else:
                res += rmax - height[r]
                r -= 1
        return res

    # Fast & Slow
    # 1004*
    def longestOnes(self, nums: List[int], k: int) -> int:
        slow = 0
        ans = 0
        for fast in range(len(nums)):
            if nums[fast] == 0:
                k -= 1
            if k < 0:
                if nums[slow] == 0:
                    k += 1
                slow += 1
            ans = max(ans, fast - slow + 1)
        return ans

    # 1493
    def longestSubarray(self, nums: List[int]) -> int:
        deleteCapacity = 1
        ans = 0
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] == 0:
                deleteCapacity -= 1
            if deleteCapacity < 0:
                if nums[slow] == 0:
                    deleteCapacity += 1
                slow += 1
            ans = max(ans, fast - slow)
        return ans

    # 209
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        if sum(nums) < target:
            return 0
        curSum = 0
        slow = 0
        minLen = len(nums)
        for fast in range(len(nums)):
            curSum += nums[fast]
            while curSum >= target:
                minLen = min(minLen, fast - slow + 1)
                curSum -= nums[slow]
                slow += 1
        return minLen

    # 76*
    def minWindow(self, s: str, t: str) -> str:
        needEach, needTotal = Counter(t), len(t)
        ans, ansSize = "", float("inf")
        slow = 0
        for fast in range(len(s)):
            if s[fast] not in needEach:
                continue
            needEach[s[fast]] -= 1
            if needEach[s[fast]] >= 0:
                needTotal -= 1
            while needTotal == 0:
                if fast - slow + 1 < ansSize:
                    ans = s[slow : fast + 1]
                    ansSize = fast - slow + 1
                if s[slow] in needEach:
                    needEach[s[slow]] += 1
                    if needEach[s[slow]] > 0:
                        needTotal += 1
                slow += 1
        return ans

    # 53*
    def maxSubArray(self, nums: List[int]) -> int:
        curSum = 0
        maxSum = nums[0]
        for num in nums:
            curSum += num
            maxSum = max(curSum, maxSum)
            if curSum < 0:
                curSum = 0
        return maxSum

    # 567*
    def checkInclusion(self, s1: str, s2: str) -> bool:
        s1Counter, s1Len, need = Counter(s1), len(s1), len(s1)
        for i in range(len(s2)):
            if s2[i] in s1Counter:
                s1Counter[s2[i]] -= 1
                if s1Counter[s2[i]] >= 0:
                    need -= 1
            if i >= s1Len and s2[i - s1Len] in s1Counter:
                s1Counter[s2[i - s1Len]] += 1
                if s1Counter[s2[i - s1Len]] > 0:
                    need += 1
            if need == 0:
                return True
        return False

    # 424*
    # popular solution:
    # def characterReplacement(self, s: str, k: int) -> int:
    #     countDict, mostFrequency = defaultdict(int), 0
    #     # mostFrequency the most frequncy UP TILL the current window
    #     maxLen = 0
    #     slow = 0
    #     for fast in range(len(s)):
    #         countDict[s[fast]] += 1
    #         mostFrequency = max(mostFrequency, countDict[s[fast]])
    #         if k + mostFrequency < fast - slow + 1:
    #             countDict[s[slow]] -= 1
    #             slow += 1
    #         maxLen = max(fast - slow + 1, maxLen)
    #     return maxLen
    # My version, same complexity, but easier to understand:
    def characterReplacement(self, s: str, k: int) -> int:
        countDict, mostFrequency = defaultdict(int), 0
        # mostFrequency: the most frequncy UP TILL the current window
        slow = 0
        for fast in range(len(s)):
            countDict[s[fast]] += 1
            if mostFrequency < countDict[s[fast]]:
                mostFrequency = countDict[s[fast]]
            elif k > 0:
                k -= 1
            else:
                countDict[s[slow]] -= 1
                slow += 1
        return fast - slow + 1

    # 1248*
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        left, right = 0, 0
        count = 0
        leftChoices = 0
        for right in range(len(nums)):
            if nums[right] % 2 == 1:
                k -= 1
                leftChoices = 0
            while k == 0:
                leftChoices += 1
                if nums[left] % 2 == 1:
                    k += 1
                left += 1
            count += leftChoices
        return count
