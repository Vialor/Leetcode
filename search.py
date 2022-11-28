import math
from typing import List, Optional

class Solution:
  # BFS
  # 1091

  # DFS
  # 695
  def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    def DFS(i, j):
      if (0 <= i < m and 0 <= j <n and grid[i][j]):
        grid[i][j] = 0 # so you do not have to use another grid to keep track of visited nodes
        return 1 + DFS(i-1,j) + DFS(i,j-1) + DFS(i+1,j) + DFS(i,j+1)
      return 0
    areas = [DFS(i, j) for i in range(m) for j in range(n)] 
    return max(areas)

  # 200
  def numIslands(self, grid: List[List[str]]) -> int:
    m, n = len(grid), len(grid[0])
    count = 0
    def DFS(i, j):
      if (0 <= i < m and 0 <= j < n and grid[i][j] == "1"):
        grid[i][j] = 0
        DFS(i-1,j); DFS(i,j-1); DFS(i+1,j); DFS(i,j+1)
        return True
      return False
    for i in range(m):
        for j in range(n):
            if DFS(i, j): count += 1
    return count

  # 212
  def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
    pass
