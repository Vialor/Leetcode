import bisect
import functools
import math
from typing import List, Optional


class Solution:
    # A. GENERAL
    # 70
    def climbStairs(self, n: int) -> int:
        dp = [0, 1, 2]  # dp[k]: methods to climb k
        for i in range(3, n + 1):
            dp.append(dp[i - 1] + dp[i - 2])
        return dp[n]

    # 746
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0, 0]
        for i in range(2, len(cost) + 1):
            dp.append(min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]))
        return dp[len(cost)]

    # 198
    def rob1(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        dp = [nums[0], max(nums[0], nums[1])]  # dp[k]: largest loot of nums[:k+1]
        for n in nums[2:]:
            dp.append(max(dp[-1], dp[-2] + n))
        return dp[-1]

    # 213*
    def rob2(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        return max(self.rob1(nums[1:]), self.rob1(nums[:-1]))

    # 64*
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                if i == 0:
                    grid[i][j] += grid[i][j - 1]
                elif j == 0:
                    grid[i][j] += grid[i - 1][j]
                else:
                    grid[i][j] += min(grid[i][j - 1], grid[i - 1][j])
        return grid[m - 1][n - 1]

    # 62
    def uniquePaths(self, m: int, n: int) -> int:
        # C(c1, c2)
        c1 = m + n - 2
        c2 = min(m, n) - 1

        contMult = lambda p, q: functools.reduce(lambda a, b: a * b, range(p, q), 1)
        return int(contMult(c1 - c2 + 1, c1 + 1) / contMult(1, c2 + 1))

    # 413*
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        dp = [0, 0]
        for i in range(2, len(nums)):
            if 2 * nums[i - 1] == nums[i] + nums[i - 2]:
                dp.append(dp[i - 1] + 1)
            else:
                dp.append(0)
        return sum(dp)

    # 343*
    def integerBreak(self, n: int) -> int:
        if n == 2:
            return 1
        if n == 3:
            return 2
        dp = [0, 1, 2, 3]
        for i in range(4, n + 1):
            dp.append(0)
            for j in range(1, i // 2 + 1):
                dp[i] = max(dp[j] * dp[i - j], dp[i])
        return dp[n]

    # 91
    def numDecodings(self, s: str) -> int:
        dp = []  # dp[k]: numDecodings(s[:k])
        # 0
        dp.append(1)
        # 1
        dp.append(0 if s[0] == "0" else 1)
        if len(s) == 1:
            return dp[1]
        # more than 1
        for i in range(2, len(s) + 1):
            lastTwo = int(s[i - 2 : i])
            lastOne = int(s[i - 1])
            result = 0
            if 10 <= lastTwo <= 26:
                result += dp[i - 2]
            if 1 <= lastOne <= 9:
                result += dp[i - 1]
            dp.append(result)
        return dp[len(s)]

    # 279
    def numSquares(self, n: int) -> int:
        dp = [0]
        for i in range(1, n + 1):
            r = 1
            minNum = float("inf")
            while i >= r**2:
                minNum = min(minNum, dp[i - r**2] + 1)
                r += 1
            dp.append(minNum)
        return dp[n]

    # 300*
    def lengthOfLIS(self, nums: List[int]) -> int:
        LIS = [1] * len(nums)  # LIS[k]: length of LIS that ends at kth index of nums
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    LIS[i] = max(LIS[i], LIS[j] + 1)
        return max(LIS)

    # 32
    def longestValidParentheses(self, s: str) -> int:
        dp = [0] * (len(s) + 1)
        for i in range(1, len(s)):
            if s[i] == ")":
                if s[i - 1] == "(":
                    dp[i + 1] = dp[i - 1] + 2
                else:
                    prev = i - 1 - dp[i]
                    if prev >= 0 and s[prev] == "(":
                        dp[i + 1] = dp[prev] + dp[i] + 2
        return max(dp)

    # 300* bisect method (this method is really sophisticated and also quicker O(nlog(n)))
    # def lengthOfLIS(self, nums: List[int]) -> int:
    #   LIS = []
    #   for num in nums:
    #       left = bisect.bisect_left(LIS, num)
    #       if left == len(LIS): LIS.append(num)
    #       else: LIS[left] = num
    #   return len(LIS)

    # 1143*
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        # dp[i][j]: longest common subsequence of text1[:i] and text2[:j]
        dp = [[0 for j in range(n + 1)] for i in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    # 354*
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        LIS = []
        for _, height in envelopes:
            left = bisect.bisect_left(LIS, height)
            if left == len(LIS):
                LIS.append(height)
            elif left == len(LIS) - 1:
                LIS[left] = height
        return len(LIS)

    # 376*
    def wiggleMaxLength(self, nums: List[int]) -> int:
        up, down = 1, 1  # up: length of wiggle subsequence that ends with an up-trend
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                up = down + 1
            elif nums[i] < nums[i - 1]:
                down = up + 1
        return max(up, down)

    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        dp = [[0] * (len(nums2) + 1) for i in range(len(nums1) + 1)]
        maxLen = 0
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                if nums1[i] == nums2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                maxLen = max(maxLen, dp[i + 1][j + 1])
        return maxLen

    # 1340 O(nd)
    # There is a monotonic stack solution which is better, but much more difficult
    def maxJumps(self, arr: List[int], d: int) -> int:
        @functools.cache
        def maxJumpsWithStartIndex(startIndex: int) -> int:
            maxSubJump = 0
            for i in range(1, d + 1):
                if startIndex + i >= len(arr) or arr[startIndex + i] >= arr[startIndex]:
                    break
                maxSubJump = max(maxSubJump, maxJumpsWithStartIndex(startIndex + i))
            for i in range(1, d + 1):
                if startIndex - i < 0 or arr[startIndex - i] >= arr[startIndex]:
                    break
                maxSubJump = max(maxSubJump, maxJumpsWithStartIndex(startIndex - i))
            return maxSubJump + 1

        return max(maxJumpsWithStartIndex(i) for i in range(len(arr)))

    # B. 0-1 KNAPSACK
    # 416* 2D
    # def canPartition(self, nums: List[int]) -> bool:
    #   total = sum(nums)
    #   if total % 2 == 1: return False
    #   targetAmount = total // 2
    #   # dp: [capacity][number of items considered]
    #   dp = [[0 for i in range(len(nums) + 1)] for c in range(targetAmount + 1)]

    #   for c in range(1, targetAmount + 1):
    #       for i in range(1, len(nums) + 1):
    #           curValue = nums[i-1]
    #           if c < curValue: dp[c][i] = dp[c][i-1]
    #           else:
    #               dp[c][i] = max(dp[c][i-1], curValue + dp[c-curValue][i-1])
    #   return dp[targetAmount][len(nums)] == targetAmount
    # 416* 1D
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False
        target = total // 2
        # dp[t]: if nums can be partitioned to reach sum t
        dp = [False] * (target + 1)
        dp[0] = True

        for num in nums:
            for t in range(target, num - 1, -1):
                dp[t] = dp[t] or dp[t - num]
        return dp[target]

    # 494
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # sum(P) - sum(N) = target, sum(P) + sum(N) = sum(nums)
        # sum(P) = (target + sum(nums)) // 2
        if (target + sum(nums)) % 2:
            return 0
        if target + sum(nums) < 0:
            return 0
        targetPositiveSum = (target + sum(nums)) // 2
        # dp[t]: number of ways to form P to reach t
        dp = [1] + [0] * (targetPositiveSum)

        for num in nums:
            for t in range(targetPositiveSum, num - 1, -1):
                dp[t] += dp[t - num]
        return dp[targetPositiveSum]

    # 474
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        # dp[p][q]: max size of subset such that it has no more than m 0's, n 1's
        dp = [[0 for q in range(n + 1)] for p in range(m + 1)]
        for s in strs:
            numOf0, numOf1 = s.count("0"), s.count("1")
            for p in range(m, numOf0 - 1, -1):
                for q in range(n, numOf1 - 1, -1):
                    dp[p][q] = max(dp[p - numOf0][q - numOf1] + 1, dp[p][q])
        return dp[m][n]

    # 322*
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp[t]: min size needed to reach t
        dp = [0] + [-1] * amount
        for coin in coins:
            for t in range(coin, amount + 1):
                if dp[t - coin] == -1:
                    pass
                elif dp[t] == -1:
                    dp[t] = dp[t - coin] + 1
                else:
                    dp[t] = min(dp[t], dp[t - coin] + 1)
        return dp[amount]

    # 518
    def change(self, amount: int, coins: List[int]) -> int:
        # dp[t]: num of ways to reach t
        dp = [1] + [0] * amount
        for coin in coins:
            for t in range(coin, amount + 1):
                dp[t] += dp[t - coin]
        return dp[amount]

    # 139*
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # dp[i]: if s[:i] can be made up from wordDict
        dp = [True] + [False] * len(s)
        for i in range(len(word), len(s) + 1):
            for word in wordDict:
                dp[i] = dp[i] or dp[i - len(word)] and s[i - len(word) : i] == word
        return dp[-1]

    # 377
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [1] + [0] * target
        for i in range(target + 1):
            for num in nums:
                if i >= num:
                    dp[i] = dp[i] + dp[i - num]
        return dp[target]

    # C. Finite State Machine
    # 309
    def maxProfit309(self, prices: List[int]) -> int:
        numOfDays = len(prices)
        # bidding => holding, bidding
        # holding => holding, cooldown
        # cooldown => bidding
        # start with bidding
        endWithBidding = [0]
        # float("-inf") is a hardcoded way to say this is an impossible situation
        endWithCooldown = [float("-inf")]
        endWithHolding = [-prices[0]]
        for i in range(1, numOfDays):
            endWithBidding.append(max(endWithBidding[i - 1], endWithCooldown[i - 1]))
            endWithHolding.append(max(endWithBidding[i - 1] - prices[i], endWithHolding[i - 1]))
            endWithCooldown.append(endWithHolding[i - 1] + prices[i])
        return max(endWithBidding[-1], endWithCooldown[-1])

    # 714
    def maxProfit714(self, prices: List[int], fee: int) -> int:
        numOfDays = len(prices)
        # bidding => bidding, holding
        # holding => bidding, holding
        # start with bidding
        endWithBidding = [0]
        endWithHolding = [-prices[0]]
        for i in range(1, numOfDays):
            endWithBidding.append(max(endWithBidding[i - 1], endWithHolding[i - 1] + prices[i] - fee))
            endWithHolding.append(max(endWithBidding[i - 1] - prices[i], endWithHolding[i - 1]))
        return endWithBidding[-1]

    # 123
    def maxProfit123(self, prices: List[int]) -> int:
        # firstBuy => firstSell => secondBuy => secondSell
        firstBuy, firstSell, secondBuy, secondSell = 0, 0, 0, 0
        for price in prices:
            firstBuy = max(-price, firstBuy)
            firstSell = max(price - firstBuy, firstSell)
            secondBuy = max(firstSell - price, secondBuy)
            secondSell = max(secondBuy + price, secondSell)
        return secondSell

    # 188*
    def maxProfit188(self, k: int, prices: List[int]) -> int:
        # endWithHolding[i]: max profit within i transactions while holding the stock
        endWithHolding = [float("-inf")] * (k + 1)
        # endWithWaiting[i]: max profit within i transactions while not holding the stock
        endWithWaiting = [0] + [float("-inf")] * k
        for price in prices:
            for i in range(1, k + 1):
                endWithHolding[i] = max(endWithWaiting[i - 1] - price, endWithHolding[i])
                endWithWaiting[i] = max(endWithWaiting[i], endWithHolding[i] + price)
        return endWithWaiting[-1]

    # 650
    def minSteps(self, n: int) -> int:
        dp = [-1, 0]
        for i in range(2, n + 1):
            dp.append(i)
            for divider in range(2, n):
                if i % divider == 0:
                    dp[i] = dp[divider] + dp[i // divider]
                    break
        return dp[n]

    # 152
    def maxProduct(self, nums: List[int]) -> int:
        minProduct, maxProduct, ans = nums[0], nums[0], nums[0]
        for num in nums[1:]:
            minProduct, maxProduct = min(minProduct * num, maxProduct * num, num), max(
                minProduct * num, maxProduct * num, num
            )
            ans = max(maxProduct, ans)
        return ans

    # 264
    def nthUglyNumber(self, n: int) -> int:
        uglyNums = [1]
        states = [0, 0, 0]
        factors = [2, 3, 5]
        for count in range(1, n):
            nextNum = min(uglyNums[states[0]] * 2, uglyNums[states[1]] * 3, uglyNums[states[2]] * 5)
            uglyNums.append(nextNum)
            states = [
                states[i] + 1 if uglyNums[states[i]] * factors[i] == nextNum else states[i] for i in range(len(states))
            ]
        return uglyNums[-1]
