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
