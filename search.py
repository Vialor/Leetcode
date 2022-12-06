import collections
import math
from typing import List, Optional

class Solution:
  # BFS
  # 1091
  def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
    n = len(grid)
    if grid[0][0] != 0: return -1
    directions = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)]
    pathQueue = collections.deque([(0, 0, 1)]) # location x, y, path length

    while pathQueue: 
      i, j, pathLen = pathQueue.popleft()
      grid[i][j] = 1
      if i == n-1 and j == n-1: return pathLen
      for a, b in directions:
        if 0 <= i+a < n and 0 <= j+b < n and grid[i+a][j+b] == 0:
          pathQueue.append((i+a, j+b, pathLen+1))
    return -1

  # 279
  def numSquares(self, n: int) -> int:
    BFSQueue = collections.deque([n])
    visited = {n: 0}
    while BFSQueue:
      cur = BFSQueue.popleft()
      curCount = visited[cur]
      i = 1
      while i*i <= cur:
        difference = cur - i*i
        if difference == 0: return curCount + 1
        if difference not in visited:
          visited[difference] = curCount + 1
          BFSQueue.append(difference)
        i += 1
  
  # 127
  def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    def differByOneLetter(w1: str, w2: str) -> bool:
      count = 0
      for i in range(len(w1)):
        if w1[i] != w2[i]:
          count += 1
        if count > 1: return False
      return count == 1

    if endWord not in wordList: return 0
    BFSQueue, otherBFSQueue = { beginWord }, { endWord }
    visited, otherVisited = { beginWord }, { endWord }
    length = 2
    while BFSQueue and otherBFSQueue:
      if len(BFSQueue) > len(otherBFSQueue):
        BFSQueue, otherBFSQueue = otherBFSQueue, BFSQueue
        visited, otherVisited = otherVisited, visited
      BFSQueue = { word for word in wordList for curWord in BFSQueue if differByOneLetter(curWord, word) and word not in visited }
      for word in BFSQueue:
        visited.add(word)
        if word in otherVisited:
          return length
      length += 1
    return 0


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
