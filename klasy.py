class Person:
    #__y=3   # private variable with __

    def __init__(self, name): #self can be changed to this or different name
        self.name = name    
    
    def desc(self):
        return "OPIS "+str(self.name)
    def set_name(self, name):
        #print(self.__y)
        self.name=name
    

p = Person("Ala")
print(p.desc())
print(p.name)
p.set_name("Anna")
print(p.name)
p.x=12
print(p.x)
#p.__y=7
#print(p.__y)
#p.set_name("Anna")
