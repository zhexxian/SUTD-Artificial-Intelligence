from stats import *
import random
from random import randint

class Player:
    def __init__(self, name):
        self.id = name
        ##["Strength", "Agility", "Intelligence", "Charm", "Vitality", "Stamina", "Spirit"]
        self.attributes = Attributes([0,0,0,0,0,0,0])
        self.randomize_attribute()
        self.fitness_value = 0

    #used to initialize attributes random;
    def randomize_attribute(self):
        maxAttribute = self.attributes.attribute_points_limit
        index = 0
        while (maxAttribute > 0) and (index < self.attributes.attribute_types):
            attribute_point = randint(0,3)
            assert(attribute_point < 4)
            if (maxAttribute - attribute_point >= 0) and (index < self.attributes.attribute_types):
                self.attributes.change_attribute(index, attribute_point)
                maxAttribute -= attribute_point
                index += 1

    #change one of the attributes 
    def mutate(self):
        ## TODO ##
        ## Implement a method to change any one of the attribute's skill point ##
        ## Can be done randomly or deterministically ## 
        ## Can also be done by swapping the skill point of one random attribute category to another ##
        
        # mutate here is:
        # if sum(self.attributes.attribute_level) <= self.attributes.attribute_points_limit
        # then one can increase one single attribute
        # otherwise, one swaps the stats of two random attributes
        # in the first case: mutate cannot increase an attribute to more than 4
        # in the first case: mutate cannot decrease an attribute to less than zero               
        if (sum(self.attributes.get_all_attributes()) <= self.attributes.attribute_points_limit):
            index = random.randint(0,6)
            if(self.attributes.get_attribute(index)<=2):
                self.attributes.change_attribute(index, self.attributes.get_attribute(index)+1)
        else:
            index1 = random.randint(0,6)
            index2 = random.randint(0,6)
            point1 = self.attributes.get_attribute(index1)
            point2 = self.attributes.get_attribute(index2)
            self.attributes.change_attribute(index1, point2)  
            self.attributes.change_attribute(index2, point1)  
        return None

    #replace one's attribute with partner's
    def marry(self, significant_other):
        
        ## TODO ##
        ## Change half of the attributes with the attributes' of the significant_other ##
        
        # there are 7 attributes
        #important: after a swap of any attribute it must hold: 
        #sum(self.attributes.attribute_level) <= self.attributes.attribute_points_limit
        # if this does not hold, then undo the swap!        
        # stop swapping if 5 attributes were swapped, or if one has gone through all 7 attributes
        swap_count = 0
        for i in range(7):
              if (swap_count >= 5):
                  return None
              point_self = self.attributes.get_attribute(i)
              point_other = significant_other.attributes.get_attribute(i)
              self.attributes.change_attribute(i, point_other)  
              significant_other.attributes.change_attribute(i, point_self)
              if ((sum(self.attributes.get_all_attributes()) <= self.attributes.attribute_points_limit) or (sum(significant_other.attributes.get_all_attributes()) <= significant_other.attributes.attribute_points_limit)):
                  self.attributes.change_attribute(i, point_self)  
                  significant_other.attributes.change_attribute(i, point_other)      
              else:
                  swap_count += 1       
        return None



            



        
