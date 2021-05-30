test: main.o
	gcc -o test main.o

main.o: main.c main.h
	gcc -ansi -c main.c main.h

clean:
	rm test *.o
