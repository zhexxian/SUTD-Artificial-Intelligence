import numpy as np
from asciiart import AsciiArt
from sklearn.preprocessing import normalize
from stats import *


class GameDisasters:

	def __init__(self):
		self.normalized_disaster_probability_array = []
		self.disaster_probability_array = []
		self.disaster_array = []

	#add new disaster type, ensure that total probability is still 1
	def add_disaster(self, Disaster):
		self.disaster_probability_array.append(Disaster.prob)
		self.disaster_array.append(Disaster)
		self.normalize_probability_array()
		
	def normalize_probability_array(self):
		self.normalized_disaster_probability_array = [x/float(sum(self.disaster_probability_array)) for x in self.disaster_probability_array]

	def generate_disaster_random(self, trials):
		out = np.random.choice(self.disaster_array, trials, p=self.normalized_disaster_probability_array)
		return out

	def generate_disaster_multinomial(self, trials):
		disaster_array = []
		for i in xrange(0, trials):
			output_disaster = np.random.multinomial(1, self.normalized_disaster_probability_array, size=1)
			idx = 0
			for x in output_disaster[0]:
				if (x > 0):
					disaster_array.append(self.disaster_array[idx])
					break
				else:
					idx = idx + 1

		return disaster_array


class Disaster(object):
	def __init__(self, id):
		self.id = id
		print id + ' created'

class OceanGhost(Disaster):
	def __init__(self, prob = 0.1):
		super(self.__class__, self).__init__("OceanGhost")
		self.prob = prob
		self.art = AsciiArt.ghost
		self.attributes = Attributes([0,3,0,0,1,0,3])

class Shark(Disaster):
	def __init__(self, prob = 0.1):
		super(self.__class__, self).__init__("Shark")
		self.prob = prob
		self.art = AsciiArt.shark
		self.attributes = Attributes([3,0,0,0,1,1,0])

class Pirate(Disaster):
	def __init__(self, prob = 0.1):
		super(self.__class__, self).__init__("Pirate")
		self.prob = prob
		self.art = AsciiArt.pirate
		self.attributes = Attributes([3,2,0,0,2,1,0])

class Tornado(Disaster):
	def __init__(self, prob = 0.1):
		super(self.__class__, self).__init__("Tornado")
		self.prob = prob
		self.art = AsciiArt.tornado		
		self.attributes = Attributes([3,1,0,0,1,1,0])	

class Siren(Disaster):
	def __init__(self, prob = 0.1):
		super(self.__class__, self).__init__("Siren")
		self.prob = prob
		self.art = AsciiArt.siren	
		self.attributes = Attributes([0,0,2,3,1,1,0])

class SeaMonster(Disaster):
	def __init__(self, prob = 0.1):
		super(self.__class__, self).__init__("Sea Monster")
		self.prob = prob
		self.art = AsciiArt.monster	
		self.attributes = Attributes([3,2,1,0,2,0,0])

class Battleship(Disaster):
	def __init__(self, prob = 0.1):
		super(self.__class__, self).__init__("Battleship")
		self.prob = prob
		self.art = AsciiArt.battleship	
		self.attributes = Attributes([3,0,3,0,1,2,0])

class Devil(Disaster):
	def __init__(self, prob = 0.1):
		super(self.__class__, self).__init__("Devil")
		self.prob = prob
		self.art = AsciiArt.devil	
		self.attributes = Attributes([0,0,3,3,1,2,1])

class Exam(Disaster):
	def __init__(self, prob = 0.1):
		super(self.__class__, self).__init__("Exam")
		self.prob = prob
		self.art = AsciiArt.exam	
		self.attributes = Attributes([1,0,1,0,1,0,1])


