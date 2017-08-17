from csp import *

# jobs is a list of jobs
# a job is a list of operations
# an operation is [name, list of alternative resources that could be
# used, a release time (earliest start time (usually 0), and a duration.

# This example is from "Variable And Value Ordering Heuristics For The
# Job Shop Scheduling Constraint Satisfaction Problem", Norman
# M. Sadeh and Mark S. Fox.  A deadline of 15 should work.

jobs4 = [ [['o_1_1', [1], 0, 3], ['o_1_2', [2], 0, 3], ['o_1_3', [3], 0, 3]],
          [['o_2_1', [1], 0, 3], ['o_2_2', [2], 0, 3]],
          [['o_3_1', [3], 0, 3], ['o_3_2', [1], 0, 3], ['o_3_3', [2], 0, 3]],
          [['o_4_1', [4], 0, 3], ['o_4_2', [2], 0, 3]] ]

# A more compact specification, assuming a single resource per
# operation and start times of 0

# Each row specifies a job by X pairs of consecutive numbers. Each
# pair of numbers defines one task of the job, which represents the
# processing of a job on a machine. For each pair, the first number
# identifies the machine it executes on, and the second number is the
# duration. The order of the X pairs defines the sequence of the
# tasks for a job.

# Simple example from www.columbia.edu/~cs2035/courses/ieor4405.S03/jobshop.doc
# A deadline of 31 should work.
j3 = [
    [1, 10, 2, 8, 3, 4],
    [2, 8, 1, 3, 4, 5, 3, 6],
    [1, 4, 2, 7, 4, 3]
    ]

# http://yetanothermathprogrammingconsultant.blogspot.sg/2012_04_01_archive.html
#            job1  job2  job3  job4  job5  job6  job7  job8  job9  job10 
#  machine1   4     2     1     5     4     3     5     2     1     8 
#  machine2   2     5     8     6     7     4     7     3     6     2 
#  machine3   2     8     6     2     2     4     2     7     1     8
# A deadline of 52 should work.

m3x10 = [
[ 4,     2,     1,     5,     4,     3,     5,     2,     1,     8 ],
[ 2,     5,     8,     6,     7,     4,     7,     3,     6,     2 ],
[ 2,     8,     6,     2,     2,     4,     2,     7,     1,     8 ]]

# This is the transpose of what we want
j10x3 = [[1, 0, 2, 0, 3, 0] for i in range(10)]
for mi, m in enumerate(m3x10):
    for ji, duration in enumerate(m):
        j10x3[ji][mi*2+1] = duration
print j10x3

# This example is a (very hard) job shop scheduling problem from Lawrence
# (1984). This test is also known as LA19 in the literature, and its
# optimal makespan is known to be 842 (Applegate and Cook,
# 1991). There are 10 jobs (J1-J10) and 10 machines (M0-M9). Every job
# must be processed on each of the 10 machines in a predefined
# sequence. The objective is to minimize the completion time of the
# last job to be processed, known as the makespan.

j10x10 = [
[2,  44,  3,   5,  5,  58,  4,  97,  0,   9,  7,  84,  8,  77,  9,  96,  1,  58,  6,  89],
[4,  15,  7,  31,  1,  87,  8,  57,  0,  77,  3,  85,  2,  81,  5,  39,  9,  73,  6,  21],
[9,  82,  6,  22,  4,  10,  3,  70,  1,  49,  0,  40,  8,  34,  2,  48,  7,  80,  5,  71],
[1,  91,  2,  17,  7,  62,  5,  75,  8,  47,  4,  11,  3,   7,  6,  72,  9,  35,  0,  55],
[6,  71,  1,  90,  3,  75,  0,  64,  2,  94,  8,  15,  4,  12,  7,  67,  9,  20,  5,  50],
[7,  70,  5,  93,  8,  77,  2,  29,  4,  58,  6,  93,  3,  68,  1,  57,  9,   7,  0,  52],
[6,  87,  1,  63,  4,  26,  5,   6,  2,  82,  3,  27,  7,  56,  8,  48,  9,  36,  0,  95],
[0,  36,  5,  15,  8,  41,  9,  78,  3,  76,  6,  84,  4,  30,  7,  76,  2,  36,  1,   8],
[5,  88,  2,  81,  3,  13,  6,  82,  4,  54,  7,  13,  8,  29,  9,  40,  1,  78,  0,  75],
[9,  88,  4,  54,  6,  64,  7,  32,  0,  52,  2,   6,  8,  54,  5,  82,  3,   6,  1,  26],
]

def parse_jobs(jobs):
    parsed = []
    for ji, j in enumerate(jobs):
        job = []
        for i in range(len(j)/2):
            job.append(['o_%d_%d'%(ji,i), [j[i*2]], 0, j[i*2+1]])
        parsed.append(job)
    return parsed

#####################################################
# The example jobs defined above are
# jobs4 - defined at the top of the file
jobs3 = parse_jobs(j3)
jobs10x10 = parse_jobs(j10x10)
jobs10x3 = parse_jobs(j10x3)
#####################################################
