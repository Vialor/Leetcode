from functools import reduce
from typing import List


class Solution:
    # 461
    def hammingDistance(self, x: int, y: int) -> int:
        return (x ^ y).bit_count()

    # 136*
    def singleNumber(self, nums: List[int]) -> int:
        return reduce(lambda last, num: last ^ num, nums, 0)

    # 268
    def missingNumber(self, nums: List[int]) -> int:
        return reduce(lambda last, i: last ^ i ^ nums[i], range(len(nums)), len(nums))

    # 137*
    def singleNumber(self, nums: List[int]) -> int:
        ones, zeros = 0, ~0
        for num in nums:
            zeros = (zeros ^ num) & ~ones
            ones = (ones ^ num) & ~zeros
        return ones

    # 260*
    def singleNumber(self, nums: List[int]) -> List[int]:
        p, q = reduce(lambda last, num: last ^ num, nums, 0), 0
        for num in nums:
            if num & p & (-p) == 0:
                q ^= num
        return [p ^ q, q]

    # 190
    def reverseBits(self, n: int) -> int:
        ans = 0
        digitNum = 32
        while digitNum > 0:
            ans <<= 1
            ans += n % 2
            n >>= 1
            digitNum -= 1
        return ans

    # 231*
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and n & (n - 1) == 0

    # 342
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and n & (n - 1) == 0 and n & 0b10101010101010101010101010101010 == 0

    # 693*
    def hasAlternatingBits(self, n: int) -> bool:
        a = n ^ (n >> 1)
        return a & (a + 1) == 0

    # 338
    def countBits(self, n: int) -> List[int]:
        ans = [0]
        for i in range(1, n + 1):
            ans.append(ans[i & (i - 1)] + 1)
        return ans

    # 476*
    def findComplement(self, num: int) -> int:
        mask = num | (num >> 1)
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        return num ^ mask

    # 371
    def getSum(self, a: int, b: int) -> int:
        mask = 0xFFFFFFFF
        while (b & mask) > 0:
            carry = (a & b) << 1
            a = a ^ b
            b = carry
        return (a & mask) if b > 0 else a

    # 318
    def maxProduct(self, words: List[str]) -> int:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        letterBitMapping = {alphabet[i]: 1 << i for i in range(len(alphabet))}
        wordsInBit = [reduce(lambda last, x: last | letterBitMapping[x], word, 0) for word in words]
        ans = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if wordsInBit[i] & wordsInBit[j] == 0:
                    ans = max(ans, len(words[i]) * len(words[j]))
        return ans
