import numpy as np
from disasters import *
from player import *

class GamePlay:

	def __init__(self):
		self.players = ["Player", "Disaster"]
		print "Initializing Game..."
		self.game_disasters = GameDisasters()

		##Adding all the disasters into the game
		self.game_disasters.add_disaster(Siren(prob=0.1))
		self.game_disasters.add_disaster(Tornado(prob=0.1))
		self.game_disasters.add_disaster(Pirate(prob=0.1))
		self.game_disasters.add_disaster(Shark(prob=0.1))
		self.game_disasters.add_disaster(OceanGhost(prob=0.1))
		self.game_disasters.add_disaster(SeaMonster(prob=0.05))
		self.game_disasters.add_disaster(Battleship(prob=0.05))
		self.game_disasters.add_disaster(Devil(prob=0.05))
		self.game_disasters.add_disaster(Exam(prob=0.15))

	def generate_disaster_random(self, days):
		## generate disasters with non-uniform probability distribution
		return self.game_disasters.generate_disaster_random(days)

	def generate_disaster_multinomial(self, days):
		return self.game_disasters.generate_disaster_multinomial(days)

	def play(self, player, days, probability_game = True, use_multinomial = True, printout = False):
		disaster_array = []

		if (use_multinomial):
			disaster_array = self.generate_disaster_multinomial(days)			
		else:
			disaster_array = self.generate_disaster_random(days)

		days_survived = 0
		final_disaster = None
		died = False

		for disaster in disaster_array:
			output = self.fight(player.attributes, disaster.attributes, probability_game)
			if (output):
				days_survived = days_survived + 1
			else:
				final_disaster = disaster
				died = True
				break

		if (printout):
			if (died):
				print final_disaster.art 
				print player.id + " survived for : " + str(days_survived) + " days out of " + str(days) + " days, and then killed by a " + final_disaster.id 

			else:
				print "VICTORY!"
				print "Player survived for : " + str(days_survived) + " days."

		return days_survived


	def fight(self, player_attr, disaster_attr, probability_game = True):
		player_win_count = 0
		disaster_win_count = 0

		for idx in xrange(0, disaster_attr.attribute_types):
			attribute_to_fight = [player_attr.get_attribute(idx),disaster_attr.get_attribute(idx)]
			try:
				attribute_to_fight_norm = [i/float(sum(attribute_to_fight)) for i in attribute_to_fight]
			except:
				attribute_to_fight_norm = [0.5, 0.5]

			if (probability_game):
				result = np.random.choice(self.players, 1, p=attribute_to_fight_norm)
				if result[0] == self.players[0]:
					player_win_count += 1
				else:
					disaster_win_count += 1

			else: 
				if (attribute_to_fight[0] > attribute_to_fight[1]):
					player_win_count += 1
				else:
					disaster_win_count += 1


		if (player_win_count >= disaster_win_count):
			# print "Player Survived"
			return True
		else:
			# print "Player Died"
			return False


if __name__ == '__main__':

	game = GamePlay()

	student = Player("Student")
	student.randomize_attribute()

	days_survived = game.play(student,30, probability_game = True, use_multinomial = False, printout = True)



