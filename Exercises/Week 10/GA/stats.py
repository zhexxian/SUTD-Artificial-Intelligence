from copy import deepcopy

class Attributes:
	def __init__(self, strength):
		self.attribute_points_limit = 15
		self.attribute_types = 7
		try:
			assert(sum(strength) <= self.attribute_points_limit)
		except:
			print "Error: total attribute points > 15"
			exit()

		for i in xrange(0,self.attribute_types):
			assert(strength[i] < 4)

		## this is a private variable, access this through the methods below
		self.__attribute_level = strength

		self.attribute_name = ["Strength", "Agility", "Intelligence", "Charm", "Vitality", "Stamina", "Spirit"]
	
	def print_attributes(self):
		sentence = "Attributes are "
		for index in xrange(0, self.attribute_types):
			sentence += self.attribute_name[index] + ":" + str(self.__attribute_level[index]) + " "
		print sentence

	## method to change one of the attribute's skill points
	def change_attribute(self, index, value):
		assert (value < 4)
		self.__attribute_level[index] = value

	## method to read one of the attribute's skill points
	def get_attribute(self, index):
		return self.__attribute_level[index]

	## method to read all of the attributes' skill points
	def get_all_attributes(self):
		return deepcopy(self.__attribute_level)