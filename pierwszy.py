def metoda(x):
    #print (x)
    if isinstance(x, int):
        print("LICZBA" + str(x))
    else :
        print("NIE LICZBA" + x)
    

def tabliczka_mnozenia():
    for x in range(0,7):
        linia= ""
        for y in range(0,7):
            linia+=str(x*y)+" "
        print(linia)

print("Ala ma kota")
x=13
x+=3
print(x)
x="Olaf"
print (x)
metoda(22)
metoda("XYZ")
tabliczka_mnozenia()
