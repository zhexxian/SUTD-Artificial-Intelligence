from search import Problem

WORDS = set(i.lower().strip() for i in open('words2.txt'))
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'



def is_valid_word(word): 
    return word.lower() in WORDS


class Word_Ladder(Problem):

    def actions(self, state):
        newWords = []
        oldWord = state
        for index in range (len(oldWord)):
            oldLetter = oldWord[index].lower()
            for newLetter in ALPHABET:
                if (newLetter != oldLetter):
                    newWord = oldWord.replace(oldLetter, newLetter, index+1)
                    if((newWord is not None) and (is_valid_word(newWord))):
                        newWords.append(newWord)
        #print newWords
        return newWords

    def result(self, state, action):
        pass
    def value(self, state):
        pass



'''
You may ﬁnd the constants and functions in Python's string module helpful;
you need to import string if you want to use it. Here are some simple test cases:
"cars" to "cats"
"cold" to "warm"
"best" to "math"
'''

if __name__=='__main__':
    print ('starting word ladder')
    if(is_valid_word('Rome')):
        print ('yes it is a valid word')
    else:
        print ('no it is not a valid word')
    Word_Ladder(object).actions('cold')