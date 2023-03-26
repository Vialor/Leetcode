import functools
import math
from typing import List, Optional


class Solution:
    # 241
    def diffWaysToCompute(self, expression: str) -> List[int]:
        if expression.isdecimal():
            return [int(expression)]

        result = []
        for i in range(1, len(expression)):
            if expression[i] not in "*-+":
                continue
            left = self.diffWaysToCompute(expression[:i])
            right = self.diffWaysToCompute(expression[i + 1 :])
            match expression[i]:
                case "*":
                    result.extend([a * b for a in left for b in right])
                case "-":
                    result.extend([a - b for a in left for b in right])
                case "+":
                    result.extend([a + b for a in left for b in right])
        return result
