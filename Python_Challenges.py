
# Created by: Dmitri Gordeinko
# Taken from LinkedInLeaning Course - Python Code Challenges by Barron Stone
# Link: https://tinyurl.com/yckckz8x

import re
from functools import lru_cache
import time
import random
import pickle
# Library for INT_MAX
import sys
import sched
import winsound as ws
import smtplib as spl
import collections
import secrets
import csv
from itertools import product
import os
import zipfile
import urllib.parse
import urllib.request

# Here is a file just for practicing python.

""" get_prime_factors:
    A function that takes a number and returns all prime factors of this number in a list form
    Create a list to store and return factors
    The divisor will start with 2 and increase whenever the number can be divided by it with no reminder
    Loop through until the divisor is no bigger than the number. The while loop will keep exectueing until
    the number will be 1 that will happen after finding the largets prime factor of the number"""

def get_prime_factors(n):
    factors = list()
    divisor = 2
    while divisor <= n:
        if (n % divisor) == 0:
            factors.append(divisor)
            n = n / divisor
        else:
            divisor += 1
    return factors


""" is_palindrome:
    a fucntion that retruns true or false based 
    on wheather the word is a palindrome or not"""

def is_palindrome(phrase):
    forward = ''.join(re.findall(r'[a-z]+', phrase.lower()))
    backwards = forward[::-1]
    if forward == backwards:
        print("\"" + phrase + "\"" + " is a palindrome.")
        return True
    else:
        print("\"" + phrase + "\"" + " is NOT a palindrom.")
        return False


""" Memoization examples with fibonacci:
    First function will not use memoization
    Second one will"""

#easy implemintation but bad prefermance under n>30
def fib1(n):
    if n == 1:
        return 1
    elif n == 2:
        return 1
    elif n > 2:
        return fib1(n-1) + fib1(n-2)

#For memoization we need a cache, here is an explicit memoization example
fib_cache = {}
def fib2(n):
    # If the requested item is in the cache just return it
    if n in fib_cache:
        return fib_cache[n]
    # Otherwise we compute the Nth term but first store it then return it
    if n == 1:
        val = 1
    elif n == 2:
        val =  1
    elif n > 2:
        val = fib2(n - 1) + fib2(n - 2)
    # Cache the value and then return it
    fib_cache[n] = val
    return val

#Implicit memoization using python build in tool with some error catching.
@lru_cache(maxsize = 1000)
def fibonacci(n):
    # First check that the input is a + integer
    if type(n) != int:
        raise TypeError("input must be positive integer")
    if n < 1:
        raise TypeError("input must be a POSITIVE integer")
    # Compute the fibonacci number
    if n == 1:
        return 1
    elif n == 2:
        return 1
    elif n > 2:
        return fibonacci(n-1) + fibonacci(n-2)

    #To check the time it take a function to execute, first import time
    #Then store the current time in a variable before calling the fucntion
    #After the fucntion print the difference of the current time and the variable
    """ FOR EXAMPLE:
    start_time = time.time()
    for n in range(1, 30):
        print(n, ":", pc1.fib1(n))
    print(" %s seconds" % (time.time() - start_time))
    """

""" Sorting a String:
    First we will split the sentence to individual words. Then to use sort() correctly(to work with capitl letters 
    we need to first append a lower-case copy of the word to the word, and then sort the words"""

def sort_words(str):
    words = str.split()
    # for each word in words take a lower case of the word and append it to the original word, then store it in words
    words = [word.lower() + word for word in words]
    words.sort()
    # take length of word divide by 2 from that point to the end of string and only keep that part (the original)
    words = [word[len(word)//2:] for word in words]
    return ' '.join(words)

""" Finding all list Items:
    This function will index all items in a list
    then search for a indexes that are equal to a given value
    Input: list to search and value to search for. Output: list of indices"""

def find_all_indices(Alist, value):
    indices = list()
    for i in range(len(Alist)):
        if Alist[i] == value:
            indices.append([i])
        elif isinstance(Alist[i], list):
            for index in find_all_indices(Alist[i], value):
                indices.append(([i] + index))
    return indices

""" The Waiting Game:
    The goal here is to make the program that will record the time between
    when the player pressed the enter the first and the second time
    Then it will return the time elapsed and how much they were off from the target"""

def waiting_game():
    target = random.randint(2,4) #target seconds to wait
    print('\n Your tartget time is {} seconds'.format(target))

    input(' ---Press Enter to Begin--- ')
    start = time.perf_counter()

    input('\n...Press Enter again after {} second...'.format(target))
    elapsed = time.perf_counter() - start

    print('\n Elapsed time: {0:.3f} seconds'.format(elapsed))
    if elapsed == target:
        print('(Unbelievable! Perfect timing!)')
    elif elapsed < target:
        print('({0:.3f} seconds too fast)'.format(target - elapsed))
    else:
        print('({0:.3f} seconds too slow)'.format(elapsed - target))

""" Save a py dictionary object to file:
    Input dictonary to save, and the file
    Output file path.
    
    Second function will load the dictonary requested"""

#   In python, "pickling" is a term used for taking an object and turning it to byte stream
def save_dict(dict_to_save, file_path):
    with open(file_path, 'wb') as file: # Write Binary
        pickle.dump(dict_to_save, file)

#   "unpickling" is a term used for bringing out pickled things.
def load_dict(file_path):
    with open(file_path, 'rb') as file: # Read Binary
        return pickle.load(file)

    """ Dijkstra's Algorithm:
        1-set all nodes distance to inf. and starting node to 0 so the alg will start from there.
        2-set all node to false in spt
        3-put min in spt set node to Ture
        4-if (curr dist > 0) and (curr dist > previous dist + the current distance in spt) and (node not visited)
        then Update dist value of the adjacent vertices of the picked vertex
        5-print spt"""

    # Python program for Dijkstra's single
    # source shortest path algorithm. The program is
    # for adjacency matrix representation of the graph

    class Graph():

        def __init__(self, vertices):
            self.V = vertices
            self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

        def printSolution(self, dist):
            print("Vertex true Distance from Source")
            for node in range(self.V):
                print(node, "t", dist[node])

        # A utility function to find the vertex with
        # minimum distance value, from the set of vertices
        # not yet included in shortest path tree
        """ Set all nodes to 0. Init min dist for every node. Search for the SP not in spt."""
        def minDistance(self, dist, sptSet):

            # Initialize minimum distance for next node
            min = sys.maxsize

            # Search not nearest vertex not in the
            # shortest path tree
            for v in range(self.V):
                if dist[v] < min and sptSet[v] == False:
                    min = dist[v]
                    min_index = v

            return min_index

        # Funtion that implements Dijkstra's single source
        # shortest path algorithm for a graph represented
        # using adjacency matrix representation
        def dijkstra(self, src):

            dist = [sys.maxsize] * self.V
            dist[src] = 0
            sptSet = [False] * self.V

            for cout in range(self.V):

                # Pick the minimum distance vertex from
                # the set of vertices not yet processed.
                # u is always equal to src in first iteration
                u = self.minDistance(dist, sptSet)

                # Put the minimum distance vertex in the
                # shortest path tree
                sptSet[u] = True

                # Update dist value of the adjacent vertices
                # of the picked vertex only if the current
                # distance is greater than new distance and
                # the vertex in not in the shortest path tree
                for v in range(self.V):
                    if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]:
                        dist[v] = dist[u] + self.graph[u][v]
        self.printSolution(dist)


""" Set an alarm:
    Play a sound alarm and print a message at the set time"""

def set_alarm(setTime,soundFile,message):
      alarm = sched.scheduler(time.time, time.sleep)
      alarm.enterabs(setTime, 1, print, argument=(message,))
      alarm.enterabs(setTime, 1, ws.PlaySound, argument=(soundFile, ws.SND_FILENAME))
      print('Alarm set for', time.asctime(time.localtime(setTime)))
      alarm.run()

""" Send an email notification:
    Input: Receiver email, address, subject line, message body.
    Using smtplib module-protoco client
    To enable receiving email: by default Gmail will block it, go to Manage your Google Account->
    Securities-> Scroll down to 'less secure app access' and turn ON 'Allow less secure apps' """

SENDER_EMAIL = 'dimanxyou@gmail.com'
SENDER_PASSWORD = ''

def send_email(receiver_email, subject, body):
    message = 'Subject: {}\n\n{}'.format(subject,body)
    with spl.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(SENDER_EMAIL,SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL,receiver_email,message)


""" Simulating Dice:
    A function that determines the probability of certain outcome when rolling dice
    Input: A variable number of arguments for sides of dice"""

def roll_dice(*dice,num_trails = 1_000_000):
    counts = collections.Counter()
    for roll in range(num_trails):
        counts[sum((random.randint(1,side) for side in dice))] += 1

    print('\nOUTCOME\tPROBABILITY')
    for outcome in range(len(dice), sum(dice)+1):
        print('{}\t{:0.2f}%'.format(outcome,counts[outcome]*100/num_trails))

""" Counting Unique Words:
    This function will count how many times a certain word appears.
    Input: Path to text file
    Ouput: Total number of words, top 20 most frequent words, and number of occurrences for the these."""

def count_unique_words(path):
    with open(path, encoding='utf-8') as file:
        all_words = re.findall(r"[0-9a-zA-Z-']+", file.read())
        all_words = [word.upper() for word in all_words]
        print('\nTotal words:', len(all_words))

        word_count = collections.Counter()
        for word in all_words:
            word_count[word] += 1

        print('\nTop 20 words:')
        for word in word_count.most_common(20):
            print('{} \t - \t {}'.format(word[0].lower(), word[1]))

""" Generating a password:
    This function will create a random password with a sequance of numbers followed by a sequance of letters
    Input: Number of words in the password
    Output: String of random words, separated by spaces"""

def generat_password(n_words):
    # need to download the file diceware from docs.python.org/3/library/secrets.html
    # of from the Exercise Files on LinkedInLearning
    with open('diceware.wordlist.asc', 'r') as file:
        lines = file.readlines()[2:7778]
        word_list = [line.split()[1] for line in lines]

    words = [secrets.choice(word_list) for i in range(n_words)]
    return ' '.join(words)

""" BFS:
    This is an algorithm used for tree traversal on graphs or tree data structures
    Pick any node, visit adjacent unvisited vertcies, mark as visited, display it and insert to queue,
    If there are no remaining adj vertices, remover the first vertex in the queue.
    Repeat these 2 steps until queue is empty or desired node was found.
    In other words this algorithm tells us which nodes are reachable from a given node.
    Complexity O(V+E)"""

def bfs(graph, node):
    visited = []  # list to keep track of visited nodes.
    queue = []  # create a queue
    visited.append(node)
    queue.append(node)

    while queue:
        s = queue.pop(0)
        print(s, end=" ")

        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

""" Merfe CSV files:
    Comma-Separated Values are plain text file format for tabular data
    This function will merge multiple CSV files to one
    Input: list of input files
    Output: file path"""

def merge_CSV(pathFrom, outputPath):
    # build a list with all fieldnames
    fieldName = list()
    for file in pathFrom:
        with open(file, 'r') as input_csv:
            fn = csv.DictReader(input_csv).fieldnames
            fieldName.extend(x for x in fn if x not in fieldName)

    # write data to output file based on field names
    with open(pathFrom, 'w', newline='') as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=fieldName)
        writer.writeheader()
        for file in pathFrom:
            with open(file, 'r') as input_csv:
                reader = csv.DictReader(input_csv)
                for row in reader:
                    writer.writerow(row)

""" Sove a Sudoku:
    representing in 2D list of list (9 lists with 9 elements each)
    Input: Partially filled puzzle
    Ouput: solved puzzle
    This will be implemented using the backtracking approach (DFS), every time the alg gets stuck
    it takes a step back and tries another path."""

def solve_sudoku(puzzle):
    for (row, col) in product(range(0,9), repeat=2):
        if puzzle[row][col] == 0: # find unassigned cell
            for num in range(1,10):
                allowed = True # check if num is allwed in row/col/box
                for i in range(0,9):
                    if (puzzle[i][col] == num) or (puzzle[row][i] == num):
                        allowed = False;
                        break
                for (i,j) in product(range(0,3), repeat=2):
                    if puzzle[row-row%3+i][col-col%3+j == num]:
                        allowed = False;
                        break
                if allowed:
                    puzzle[row][col] = num
                    if trial := solve_sudoku(puzzle):
                        return trial
                    else:
                        puzzle[row][col] = 0
            return False # could not place a number in this cell
    return puzzle

''' PROBELM in line 380 if statement: 'bool' object is not subscriptable '''

def print_sudoku(puzzle):
    # replace the zeros with Starts and make everything more visible.
    puzzle = [['*' if num == 0 else num for num in row] for row in puzzle]
    print()
    for row in range(0,9):
        if ((row % 3 == 0) and (row != 0)):
            print('-' * 30) # draw a horizontal lines
        for col in range(0 ,9):
            if ((col % 3 == 0) and (col != 0)):
                print(' | ', end='') # draw vertaical lines
            print('', puzzle[row][col], '', end='')
        print()
    print()

""" Build a zip archive:
    The zip archive should maintain folder structure relative to top level dictornery.
    Input: directory path, list of extentions, output file path
    Output: A zip file"""

def zip_archives(search_dir, extension_list, output_path):
    with zipfile.ZipFile(output_path, 'w') as output_zip:
        for root, dirs, file in os.walk(search_dir):
            rel_path = os.path.relpath(root, search_dir)
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() in extension_list:
                    output_zip.write(os.path.join(root,file),
                                     arcname=os.path.join(rel_path,file))

""" Sequantially download numbered files:
    The function will download and save a sequence of files
    Input: URL for first item
    Output: output directory path"""

def download_files(first_url, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    url_head, url_tail = os.path.split(first_url)
    first_index = re.findall(r'[0-9]+', url_tail)[-1]
    index_count, error_count = 0, 0
    while(error_count < 5):
        next_index = str(int(first_index) + index_count)
        if first_index[0] == '0':   # zero padded
            next_index = '0' * (len(first_index) - len(next_index)) + next_index
        next_url = urllib.parse.urljoin(url_head, re.sub(first_index, next_index, url_tail))
        try:
            output_file = os.path.join(output_dir, os.path.basename(next_url))
            urllib.request.urlretrieve(next_url, output_file)
            print("Successfully downloaded {}".format(os.path.basename(next_url)))
        except IOError:
            print("Could not retrieve {}".format(next_url))
            error_count += 1
        index_count += 1


    ################## End of python code challenges ##################