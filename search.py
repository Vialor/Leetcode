import collections
import math
from typing import List, Optional, Tuple

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

  # 547
  def findCircleNum(self, isConnected: List[List[int]]) -> int:
    cityNum = len(isConnected)
    visitedCities = []
    count = 0
    def DFS(city: int):
      visitedCities.append(city)
      for connectedCity in range(cityNum):
        if connectedCity not in visitedCities and isConnected[city][connectedCity]:
          DFS(connectedCity)
    for city in range(cityNum):
      if city not in visitedCities:
        DFS(city)
        count += 1
    return count

  # 130 double DFS
  # def solve(self, board: List[List[str]]) -> None:
  #   # every grid has 4 states: X O A B
  #   # A: has been visited by DFSBoundary
  #   # B: has been visited, and should be set to O in the final answer
  #   m,n = len(board), len(board[0])

  #   def DFSBoundary(i: int, j: int) -> Tuple[int, int] or None:
  #     # return a grid at the border connected to board[i][j]
  #     if board[i][j] == 'X' or board[i][j] == 'A': return None
  #     board[i][j] = 'A'
  #     if i == m-1 or i == 0 or j == n-1 or j == 0: return (i, j)
  #     return DFSBoundary(i-1, j) or DFSBoundary(i+1, j) or DFSBoundary(i, j-1) or DFSBoundary(i, j+1)

  #   def DFSPaint(i: int, j: int, s: str) -> None:
  #     # set all grids connected to board[i][j] to s
  #     if i >= m or i < 0 or j >= n or j < 0 or board[i][j] == 'X' or board[i][j] == s: return
  #     board[i][j] = s
  #     DFSPaint(i-1, j, s)
  #     DFSPaint(i+1, j, s)
  #     DFSPaint(i, j-1, s)
  #     DFSPaint(i, j+1, s)
      
  #   for i in range(m):
  #     for j in range(n):
  #       if board[i][j] == 'O':
  #         coordinates  = DFSBoundary(i, j)
  #         if coordinates: DFSPaint(coordinates[0], coordinates[1], 'B')
  #         else: DFSPaint(i, j, 'X')
          
  #   for i in range(m):
  #     for j in range(n):
  #       if board[i][j] == 'B':
  #         board[i][j] = 'O'
  # 130 
  def solve(self, board: List[List[str]]) -> None:
    m, n = len(board), len(board[0])
    
    def DFS(i: int, j: int) -> None:
      if i < 0 or i >= m or j < 0 or j >= n: return
      if board[i][j] == 'X' or board[i][j] == 'A': return
      board[i][j] = 'A'
      DFS(i-1, j); DFS(i+1, j); DFS(i, j-1), DFS(i, j+1)
    
    for i in range(m):
      if (board[i][0] == 'O'):
        DFS(i, 0)
      if (board[i][n-1] == 'O'):
        DFS(i, n-1)

    for j in range(n):
      if (board[0][j] == 'O'):
        DFS(0, j)
      if (board[m-1][j] == 'O'):
        DFS(m-1, j)

    for i in range(m):
      for j in range(n):
        if board[i][j] == 'A':
          board[i][j] = 'O'
        elif board[i][j] == 'O':
          board[i][j] = 'X'

  # 212
  def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
    pass
