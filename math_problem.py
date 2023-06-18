import math
from typing import List, Optional


class Solution:
    # 204
    def countPrimes(self, n: int) -> int:
        definitelyNotPrimes = [False] * (n + 1)
        count = 0
        for num in range(2, n):
            if definitelyNotPrimes[num]:
                continue
            count += 1

            cur = 2 * num
            while cur < n:
                definitelyNotPrimes[cur] = True
                cur += num
        return count

    # 48
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(ceil(n / 2)):
            for j in range(floor(n / 2)):
                curi, curj = i, j
                for time in range(3):
                    matrix[curi][curj], matrix[n - 1 - curj][curi] = matrix[n - 1 - curj][curi], matrix[curi][curj]
                    curi, curj = n - 1 - curj, curi
