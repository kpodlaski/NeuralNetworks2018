lista = []
lista2 = ['a', "b", 'c']
lista.append(2)
lista.append(3)
lista.insert(1,"bb")

for x in lista:
    print (x)

print (lista2)
print (lista[2])
a= lista.pop()
print (lista)
a ="Ala ma Kota"
print(a[4])
print(a[4::])
print(a[4::6])
print(a[4:-2:])

def tuple_test(x):
    a,b = x, x*x
    return a,b

w = tuple_test(3)
print (w)
print (w[0])
v = (2,4)
print (w+v)
x = float ('inf')
print(x*3)
print(3//x)

c = complex(2,7)
print(c)
c= c.conjugate()
print(c**2);
print(str(c.real) + " "+ str(c.imag))
