import collections
import copy
import math
from typing import List, Optional, Set, Tuple

# 212
class TrieNode:
  def __init__(self, parent=None):
    self.children = {}
    self.isEnd = False
    self.parent = parent

  def addWord(self, word: str):
    cur = self
    for c in word:
      if c not in cur.children:
        cur.children[c] = TrieNode(cur)
      cur = cur.children[c]
    cur.isEnd = True
  
  def removeWord(self, word):
    self.isEnd = False
    cur = self
    while cur and word:
      if len(cur.children) > 0:
        break
      else:
        cur = cur.parent
        last, word = word[-1], word[:-1]
        del cur.children[last]

# 257
class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

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

  # 417
  def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    m, n = len(heights), len(heights[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    reach = [['' for j in range(n)] for i in range(m)]
    result = []
    def DFS(i: int, j: int, S: str) -> None:
      reach[i][j] += S
      for x, y in directions:
        nextI, nextJ = i+x, j+y
        if nextI < 0 or nextJ < 0 or nextI >= m or nextJ >= n \
          or heights[nextI][nextJ] < heights[i][j] \
          or S in reach[nextI][nextJ]:
          continue
        DFS(nextI, nextJ, S)

    for i in range(m):
      DFS(i, 0, 'P'); DFS(i, n-1, 'A')
    for j in range(n):
      DFS(0, j, 'P'); DFS(m-1, j, 'A')
    for i in range(m):
      for j in range(n):
        curGrid = reach[i][j]
        if 'A' in curGrid and 'P' in curGrid:
          result.append([i, j])
    return result

  # Backtracking
  # 17
  def letterCombinations(self, digits: str) -> List[str]:
    digitToLetters = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz',
    }
    result = []
    def backtrack(curStr: str):
      if len(digits) == len(curStr):
        result.append(curStr)
        return
      for l in digitToLetters[digits[len(curStr)]]:
        backtrack(curStr + l)
    if digits: backtrack('')
    return result

  # 93
  def restoreIpAddresses(self, s: str) -> List[str]:
    result = []

    def isValidIntString(n: str):
      if not n.isdecimal(): return False
      if n != '0' and n[0] == '0': return False
      nNum = int(n)
      return nNum >= 0 and nNum <= 255

    def backtrack(curStr: str, depth: int, index: int):
      if depth == 4:
        if index >= len(s):
          result.append(curStr)
        return
      for i in [1,2,3]:
        if index+i > len(s): break
        nextStr = s[index:index+i]
        if isValidIntString(nextStr):
          backtrack(curStr + '.' + nextStr if curStr else nextStr, depth+1, index+i)
    
    backtrack('', 0, 0)
    return result
      
  # 79
  def exist(self, board: List[List[str]], word: str) -> bool:  
      ROWS, COLS = len(board), len(board[0])   
      def DFS(r,c,i):
        if i == len(word): return True
        if (r <0  or c <0 or r >= ROWS or c >= COLS or word[i]!= board[r][c]):
          return False
        temp = board[r][c]
        board[r][c] = "#"
        res = (DFS(r+1, c, i+1) or DFS(r-1, c, i+1) or DFS(r, c+1, i+1) or DFS(r, c-1, i+1)) 
        board[r][c] = temp 
        return res   
      for r in range(ROWS):
        for c in range(COLS):
          if board[r][c] == word[0]:
            if DFS(r,c,0):
              return True
      return False

  # 212*
  def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
    result = []
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    ROWS, COLS = len(board), len(board[0])
    trieRoot = TrieNode()
    for word in words:
      trieRoot.addWord(word)

    def DFS(row: int, col: int, trieNode: TrieNode, path: str):
      if row < 0 or row >= ROWS or col < 0 or col >= COLS\
        or board[row][col] not in trieNode.children:
        return
      curChar = board[row][col]
      curNode = trieNode.children[curChar]
      curPath = path + curChar
      if curNode.isEnd:
        result.append(curPath)
        curNode.removeWord(curPath)
      board[row][col] = '#'
      for i, j in directions:
        DFS(row+i, col+j, curNode, curPath)
      board[row][col] = curChar

    for row in range(ROWS):
      for col in range(COLS):
        DFS(row, col, trieRoot, '')
    return result

  # 257
  def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
    result = []
    def DFS(root, path):
      if not root: return
      newPath = f"{path}->{root.val}" if path else str(root.val)
      if not root.left and not root.right:
        result.append(newPath)
      else:
        DFS(root.left, newPath)
        DFS(root.right, newPath)
    DFS(root, '')
    return result

  # 46
  def permute(self, nums: List[int]) -> List[List[int]]:
    result = []
    visited = [ False ] * len(nums)
    def DFS(permutation: List[int]):
      if len(permutation) == len(nums):
        result.append(permutation)
        return
      for i in range(len(nums)):
        if not visited[i]:
          visited[i] = True
          DFS(permutation + [nums[i]])
          visited[i] = False
    DFS([])
    return result

  # 47
  def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    result = []
    visited = [ False ] * len(nums)
    def DFS(permutation: List[int]):
      if len(permutation) == len(nums):
        result.append(permutation)
        return
      for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1] and not visited[i-1]: continue
        if not visited[i]:
          visited[i] = True
          DFS(permutation + [nums[i]])
          visited[i] = False
    DFS([]) 
    return result

  # 77
  def combine(self, n: int, k: int) -> List[List[int]]:
    result = []
    visited = [ False ] * (n + 1)
    def DFS(combination: List[int]):
      if len(combination) == k:
        result.append(combination)
        return
      start = combination[-1] + 1 if combination else 1
      for i in range(start, n+1):
        if not visited[i]:
          visited[i] = True
          DFS(combination + [i])
          visited[i] = False
    DFS([])
    return result

  # 39
  def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    result = []
    def DFS(path: List[int], curSum: int):
      for candidate in candidates:
        if path and candidate < path[-1]: continue
        nextPath = path + [candidate]
        nextSum = curSum + candidate
        if nextSum < target:
          DFS(nextPath, nextSum)
        else:
          if nextSum == target:
            result.append(nextPath)
          break
    DFS([], 0)
    return result
  
  # 70
  def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    visited = [ False ] * len(candidates)
    result = []
    def DFS(path: List[int], curSum: int, startIndex: int):
      for i in range(startIndex, len(candidates)):
        candidate = candidates[i]
        if i > 0 and candidates[i] == candidates[i-1] and not visited[i-1]: continue
        nextPath = path + [candidate]
        nextSum = curSum + candidate
        if nextSum < target:
          visited[i] = True
          DFS(nextPath, nextSum, i+1)
          visited[i] = False
        else:
          if nextSum == target:
            result.append(nextPath)
          break
    DFS([], 0, 0)
    return result

  # 216
  def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    result = []
    def DFS(path: List[int], curSum: int, pathLen: int, start: int):
      for i in range(start, 10):
        nextPath = path + [i]
        nextSum = curSum + i
        nextPathLen = pathLen + 1
        if nextSum == n and nextPathLen == k:
          result.append(nextPath)
          break
        elif nextSum < n and nextPathLen == k:
          continue
        elif nextPathLen < k and nextSum < n:
          DFS(nextPath, nextSum, nextPathLen, i+1)
        else:
          break
    DFS([], 0, 0, 1)
    return result

  # 78
  def subsets(self, nums: List[int]) -> List[List[int]]:
    result = []
    def DFS(path, start):
      result.append(path)
      for i in range(start, len(nums)):
        DFS(path + [nums[i]], i+1)
    DFS([], 0)
    return result
  
  # 90
  def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    visited = [ False ] * len(nums)
    result = []
    def DFS(path, start):
      result.append(path)
      for i in range(start, len(nums)):
        if i > 0 and not visited[i-1] and nums[i] == nums[i-1]: continue
        visited[i] = True
        DFS(path + [nums[i]], i+1)
        visited[i] = False
    DFS([], 0)
    return result

  # 131
  def partition(self, s: str) -> List[List[str]]:
    result = []
    def isPalindrome(string: str):
      return string == string[::-1]
    def DFS(path: List[str], rest: str):
      if not rest: result.append(path)
      for i in range(len(rest)):
        if isPalindrome(rest[:i+1]):
          DFS(path + [rest[:i+1]], rest[i+1:])
    DFS([], s)
    return result

  # 37
  def solveSudoku(self, board: List[List[str]]) -> None:
    allPossibleNumbers = { str(n) for n in range(1, 10) }
    gridPositions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    def getAvailableNumbers(i: int, j: int) -> List[str]:
      alreadyChosenNumbers = set()
      gridI, gridJ = i // 3, j // 3
      for p in gridPositions:
        alreadyChosenNumbers.add(board[gridI*3 + p[0]][gridJ*3 + p[1]])
      for p in range(9):
        alreadyChosenNumbers.add(board[i][p])
        alreadyChosenNumbers.add(board[p][j])
      return allPossibleNumbers - alreadyChosenNumbers

    def getNextEmptyCell(startI: int=-1, startJ: int=-1) -> Tuple[int, int]: # find next empty cell after board[startI][startJ]
      for j in range(startJ+1, 9):
        if board[startI][j] == '.':
          return startI, j
      for i in range(startI+1, 9):
        for j in range(9):
          if board[i][j] == '.':
            return i, j
      return -1, -1

    def DFS(i: int, j: int) -> bool: # return True when solution found
      if i == -1 and j == -1: return True
      nextI, nextJ = getNextEmptyCell(i, j)
      for n in getAvailableNumbers(i, j):
        board[i][j] = n
        if DFS(nextI, nextJ): return True
        board[i][j] = '.'
      return False

    initialI, initialJ = getNextEmptyCell()
    DFS(initialI, initialJ)
  
  # 51
  def solveNQueens(self, n: int) -> List[List[str]]:
    result = []
    board = ['.' * n] * n
    
    def isUnderAttack(row: int, col: int) -> bool:
      for a in range(n):
        for b in range(n):
          if board[a][b] == 'Q' and (a == row or b == col or a-b == row-col or a+b == row+col):
            return True
      return False

    def DFS(row: int):
      if row == n:
        result.append(copy.deepcopy(board))
        return
      for col in range(n):
        if not isUnderAttack(row, col):
          board[row] = board[row][:col] + 'Q' + board[row][col+1:]
          DFS(row + 1)
          board[row] = '.' * n

    DFS(0)
    return result
